"""Tests for the agent MCP server: protocol, tool outputs, and error handling."""
import io
import json
import subprocess
import sys

import numpy as np
import pytest

from finlab.agent import server as srv
from finlab.agent import tools as tl
from finlab.bars import cusum_filter
from finlab.fracdiff import frac_diff_ffd
from finlab.hrp import hrp_weights
from finlab.stats import deflated_sharpe_ratio, sharpe_ratio


def _rpc(method, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return srv.handle_message(msg)


def _call(name, arguments):
    reply = _rpc("tools/call", {"name": name, "arguments": arguments})
    result = reply["result"]
    return json.loads(result["content"][0]["text"]), result["isError"]


def test_initialize_reports_protocol_and_tools_capability():
    reply = _rpc("initialize", {})
    assert reply["result"]["protocolVersion"] == srv.PROTOCOL_VERSION
    assert "tools" in reply["result"]["capabilities"]
    assert reply["result"]["serverInfo"]["name"] == "finlab"


def test_tools_list_schemas_are_well_formed():
    tools = _rpc("tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"sharpe_ratio", "deflated_sharpe_ratio", "probability_of_backtest_overfitting",
            "hrp_weights", "cusum_filter", "cpcv_split_summary"} <= names
    for t in tools:
        assert t["inputSchema"]["type"] == "object"
        assert set(t["inputSchema"]["required"]) <= set(t["inputSchema"]["properties"])
        assert t["description"]


def test_sharpe_matches_library():
    r = np.random.default_rng(0).normal(0.001, 0.01, 300)
    out, err = _call("sharpe_ratio", {"returns": r.tolist()})
    assert not err
    assert out["sharpe"] == pytest.approx(sharpe_ratio(r), rel=1e-12)


def test_deflated_sharpe_matches_library():
    out, err = _call("deflated_sharpe_ratio",
                     {"sr_hat": 0.3, "n_trials": 40, "var_sr": 0.02, "n_obs": 250})
    assert not err
    assert out["dsr"] == pytest.approx(deflated_sharpe_ratio(0.3, 40, 0.02, 250), rel=1e-12)


def test_cusum_and_fracdiff_match_library():
    x = np.cumsum(np.random.default_rng(1).normal(size=200))
    out, _ = _call("cusum_filter", {"series": x.tolist(), "threshold": 1.5})
    assert out["event_positions"] == [int(i) for i in cusum_filter(x, 1.5)]
    out, _ = _call("frac_diff_ffd", {"series": x.tolist(), "d": 0.4})
    np.testing.assert_allclose(np.array(out["values"], dtype=float),
                               frac_diff_ffd(x, 0.4), rtol=1e-12, equal_nan=True)


def test_hrp_matches_library():
    cov = [[1.0, 0.3, 0.1], [0.3, 2.0, 0.2], [0.1, 0.2, 0.5]]
    out, _ = _call("hrp_weights", {"cov": cov})
    np.testing.assert_allclose(out["weights"], hrp_weights(np.array(cov)), rtol=1e-12)
    assert sum(out["weights"]) == pytest.approx(1.0)


def test_cpcv_summary_counts():
    out, err = _call("cpcv_split_summary",
                     {"n_samples": 120, "n_groups": 6, "n_test_groups": 2, "horizon": 5})
    assert not err
    assert out["n_splits"] == 15 and out["n_paths"] == 5


def test_bad_arguments_are_reported_not_raised():
    out, err = _call("sharpe_ratio", {"returns": []})
    assert err and "non-empty" in out["error"]
    out, err = _call("sharpe_ratio", {"returns": [1.0, float("nan")]})
    assert err
    out, err = _call("frac_diff_ffd", {"series": [1, 2, 3], "d": 1.5})
    assert err and "[0, 1]" in out["error"]
    out, err = _call("sharpe_ratio", {"returns": [1, 2], "annualize": False})
    assert not err and out["sharpe"] == pytest.approx(sharpe_ratio(np.array([1.0, 2.0]), annualize=False))


def test_unknown_tool_and_method():
    out, err = _call("no_such_tool", {})
    assert err and "unknown tool" in out["error"]
    reply = _rpc("resources/list")
    assert reply["error"]["code"] == -32601


def test_size_limit_enforced(monkeypatch):
    monkeypatch.setattr(tl, "MAX_ELEMENTS", 10)
    out, err = _call("sharpe_ratio", {"returns": list(range(1, 50))})
    assert err and "limit" in out["error"]


def test_notifications_get_no_reply():
    assert srv.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert srv.handle_message({"jsonrpc": "2.0", "id": 9, "method": "tools/list"}) is not None


def test_stdio_loop_handles_parse_errors_and_batches_of_lines():
    lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        "{not json",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "sharpe_ratio", "arguments": {"returns": [0.01, 0.02]}}}),
    ]) + "\n"
    out = io.StringIO()
    srv.main(io.StringIO(lines), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert len(replies) == 3  # ping, parse error, tools/call; the notification gets no reply
    assert replies[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert replies[1]["error"]["code"] == -32700
    assert replies[2]["id"] == 2
    assert not replies[2]["result"]["isError"]


def test_subprocess_round_trip():
    """The real entry point, over real pipes, as an agent would run it."""
    req = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                      "params": {"name": "sharpe_ratio", "arguments": {"returns": [0.01, -0.005, 0.02]}}})
    proc = subprocess.run([sys.executable, "-m", "finlab.agent"], input=req + "\n",
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    reply = json.loads(proc.stdout.splitlines()[0])
    assert reply["id"] == 7 and not reply["result"]["isError"]


def test_call_tool_unknown_name_returns_error_shape():
    result, is_error = tl.call_tool("no_such_tool", {})
    assert is_error is True
    assert result == {"error": "unknown tool 'no_such_tool'"}


def test_tool_error_is_reported_by_call_tool():
    assert issubclass(tl.ToolError, ValueError)
    with pytest.raises(tl.ToolError, match="non-empty"):
        tl.TOOLS["sharpe_ratio"].handler({"returns": []})
    result, is_error = tl.call_tool("sharpe_ratio", {"returns": []})
    assert is_error is True
    assert "non-empty" in result["error"]
    # Missing arguments (None) are treated as an empty dict, not a crash.
    result, is_error = tl.call_tool("sharpe_ratio", None)
    assert is_error is True and "error" in result


def test_list_tools_entries_have_name_description_and_input_schema():
    entries = tl.list_tools()
    assert [e["name"] for e in entries] == list(tl.TOOLS.keys())
    assert entries, "registry must not be empty"
    for e in entries:
        assert set(e) == {"name", "description", "inputSchema"}
        assert isinstance(e["name"], str) and e["name"]
        assert isinstance(e["description"], str) and e["description"]
        assert e["inputSchema"]["type"] == "object"
        assert isinstance(e["inputSchema"]["properties"], dict)
def test_non_string_method_is_invalid_request_not_crash():
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 3, "method": 5})
    assert reply["id"] == 3 and reply["error"]["code"] == -32600


def test_list_params_is_invalid_params_not_crash():
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": [1]})
    assert reply["id"] == 4 and reply["error"]["code"] == -32602


def test_list_arguments_reported_as_tool_error():
    out, err = _call("sharpe_ratio", [1])
    assert err and "object" in out["error"]


def test_stdio_loop_survives_malformed_messages():
    lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": 5}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": [1]}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "sharpe_ratio", "arguments": [1]}}),
        "[1, 2]",
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping"}),
    ]) + "\n"
    out = io.StringIO()
    srv.main(io.StringIO(lines), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, 2, 3, None, 4]
    assert replies[0]["error"]["code"] == -32600
    assert replies[1]["error"]["code"] == -32602
    assert replies[2]["result"]["isError"]
    assert replies[3]["error"]["code"] == -32600
    assert replies[4] == {"jsonrpc": "2.0", "id": 4, "result": {}}


def test_stdio_loop_survives_unexpected_exception(monkeypatch):
    real = srv.handle_message

    def boom_on_method(msg):
        if msg.get("method") == "boom":
            raise RuntimeError("boom")
        return real(msg)

    monkeypatch.setattr(srv, "handle_message", boom_on_method)
    lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "boom"}),
        json.dumps({"jsonrpc": "2.0", "method": "boom"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
    ]) + "\n"
    out = io.StringIO()
    srv.main(io.StringIO(lines), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2]
    assert replies[0]["error"]["code"] == -32603
    assert replies[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_deeply_nested_line_is_parse_error_and_server_survives():
    lines = "[" * 100_000 + "\n" + json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"}) + "\n"
    out = io.StringIO()
    srv.main(io.StringIO(lines), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1] == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_tool_overflow_is_reported_as_tool_error(monkeypatch):
    def handler(args):
        raise OverflowError("int too large")

    monkeypatch.setitem(tl.TOOLS, "boom", tl.Tool("boom", "raises", {"type": "object", "properties": {}}, handler))
    result, is_error = tl.call_tool("boom", {})
    assert is_error is True
    assert result["error"].startswith("OverflowError")


def test_oversized_integer_id_is_parse_error_and_server_survives():
    big = "1" * 5000
    lines = "\n".join([
        '{"jsonrpc":"2.0","id":' + big + ',"method":"ping"}',
        json.dumps({"jsonrpc": "2.0", "id": 5, "method": "ping"}),
    ]) + "\n"
    out = io.StringIO()
    srv.main(io.StringIO(lines), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1] == {"jsonrpc": "2.0", "id": 5, "result": {}}


def _strict_loads(text):
    def reject(name):
        raise ValueError(f"non-standard constant {name}")

    return json.loads(text, parse_constant=reject)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_constant_id_is_parse_error_and_reply_is_strict_json(token):
    line = '{"jsonrpc":"2.0","id":' + token + ',"method":"ping"}'
    out = io.StringIO()
    srv.main(io.StringIO(line + "\n"), out)
    reply = _strict_loads(out.getvalue().splitlines()[0])
    assert reply["error"]["code"] == -32700
    assert reply["id"] is None


def test_string_boolean_is_rejected_not_truthy():
    out, err = _call("sharpe_ratio", {"returns": [1, 2, 3], "annualize": "false"})
    assert err and "'annualize' must be a boolean" in out["error"]
    out, err = _call("cusum_filter", {"series": [1, 2, 3], "threshold": 1, "is_returns": "false"})
    assert err and "'is_returns' must be a boolean" in out["error"]
    out, err = _call("sharpe_ratio", {"returns": [1, 2], "annualize": False})
    assert not err and out["sharpe"] == pytest.approx(sharpe_ratio(np.array([1.0, 2.0]), annualize=False))


def test_nested_array_is_rejected_for_flat_input():
    out, err = _call("sharpe_ratio", {"returns": [[0.1, 0.2], [0.3, 0.4]]})
    assert err and "'returns' must be a flat array of numbers" in out["error"]


def test_boolean_elements_are_rejected_as_numbers_and_symbols():
    out, err = _call("sharpe_ratio", {"returns": [True, False, True, True]})
    assert err and "'returns' element 0" in out["error"] and "bool" in out["error"]
    out, err = _call("kontoyiannis_entropy", {"message": ["a", True]})
    assert err and "'message' element 1" in out["error"]
    out, err = _call("plug_in_entropy", {"message": [False, 1]})
    assert err and "'message' element 0" in out["error"]


def test_matrix_bool_and_string_elements_are_rejected():
    rows = [[True, False], [False, True], [True, True], [False, False]]
    out, err = _call("probability_of_backtest_overfitting", {"performance_matrix": rows, "n_partitions": 2})
    assert err and "'performance_matrix' element (0, 0) must be a number, got bool" in out["error"]
    rows = [["0.1", "0.2"], ["0.3", "-0.1"], ["0.2", "0.5"], ["0.0", "0.4"]]
    out, err = _call("probability_of_backtest_overfitting", {"performance_matrix": rows, "n_partitions": 2})
    assert err and "'performance_matrix' element (0, 0) must be a number, got str" in out["error"]
    rows = [[0.1, 0.2], [0.3, -0.1], [0.2, 0.5], [0.0, 0.4]]
    out, err = _call("probability_of_backtest_overfitting", {"performance_matrix": rows, "n_partitions": 2})
    assert not err and "pbo" in out


def test_non_numeric_element_names_argument_index_and_type():
    out, err = _call("sharpe_ratio", {"returns": [0.1, "x"]})
    assert err and "element 1" in out["error"] and "str" in out["error"]
    assert "'returns'" in out["error"]
    out, err = _call("sharpe_ratio", {"returns": ["0.1", "x"]})
    assert err and "element 0" in out["error"] and "str" in out["error"]


def test_int_argument_keeps_large_integers_exact():
    assert tl._int({"seed": 9007199254740993}, "seed") == 9007199254740993
    assert type(tl._int({"seed": 9007199254740993}, "seed")) is int
    assert tl._int({"seed": 3.0}, "seed") == 3
    corr = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]]
    out, err = _call("onc_clusters", {"corr": corr, "seed": 9007199254740993})
    assert not err


def test_oversized_integers_are_tool_errors_naming_the_argument():
    huge = int("1" + "0" * 399)
    out, err = _call("sharpe_ratio", {"returns": [0.1, 0.2, -0.1, 0.3], "periods_per_year": huge})
    assert err and "'periods_per_year'" in out["error"]
    out, err = _call("onc_clusters", {"corr": [[1.0, 0.9], [0.9, 1.0]], "seed": huge})
    assert err and "'seed'" in out["error"]
    out, err = _call("onc_clusters", {"corr": [[1.0, 0.9], [0.9, 1.0]], "max_k": huge})
    assert err and "'max_k'" in out["error"]


def test_fracdiff_thres_out_of_range_is_rejected():
    out, err = _call("frac_diff_ffd", {"series": [1, 2, 3], "d": 0.4, "thres": 2})
    assert err and "'thres'" in out["error"]
    out, err = _call("frac_diff_ffd", {"series": [1, 2, 3], "d": 0.4, "thres": 0})
    assert err and "'thres'" in out["error"]


def test_onc_max_k_must_be_integer():
    corr = [[1.0, 0.9, 0.1, 0.0], [0.9, 1.0, 0.0, 0.1], [0.1, 0.0, 1.0, 0.8], [0.0, 0.1, 0.8, 1.0]]
    out, err = _call("onc_clusters", {"corr": corr, "max_k": 2.7})
    assert err and "'max_k' must be an integer" in out["error"]
    out, err = _call("onc_clusters", {"corr": corr, "max_k": 2})
    assert not err and out["n_clusters"] <= 2


@pytest.mark.parametrize("params", [[], 0, "x", [1], False])
def test_tools_call_params_that_are_not_an_object_are_invalid_params(params):
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": params})
    assert reply["id"] == 11
    assert reply["error"] == {"code": -32602, "message": "invalid params: expected an object"}


@pytest.mark.parametrize("params", [[], 0, "", False, "x", [1]])
def test_methods_that_ignore_params_accept_any_params(params):
    ping = srv.handle_message({"jsonrpc": "2.0", "id": 13, "method": "ping", "params": params})
    assert ping == {"jsonrpc": "2.0", "id": 13, "result": {}}
    listed = srv.handle_message({"jsonrpc": "2.0", "id": 14, "method": "tools/list", "params": params})
    assert "error" not in listed
    assert {t["name"] for t in listed["result"]["tools"]} >= {"sharpe_ratio"}


@pytest.mark.parametrize("params", [{}, {"arguments": {}}, {"name": None}, {"name": 5}])
def test_tools_call_without_string_name_says_missing_tool_name(params):
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": params})
    assert reply["id"] == 12
    assert reply["error"]["code"] == -32602
    assert "missing tool name" in reply["error"]["message"]


def test_cpcv_rejects_non_positive_n_samples():
    out, err = _call("cpcv_split_summary", {"n_samples": -5, "n_groups": 6, "n_test_groups": 2})
    assert err and "n_samples must be a positive integer, got -5" in out["error"]


def test_cpcv_rejects_negative_horizon():
    out, err = _call("cpcv_split_summary",
                     {"n_samples": 120, "n_groups": 6, "n_test_groups": 2, "horizon": -1})
    assert err and "'horizon'" in out["error"]


def test_cpcv_summary_refuses_split_count_beyond_cap():
    # C(20, 10) = 184756 splits must be refused, not all materialized (38 s, 1.6 GB).
    out, err = _call("cpcv_split_summary",
                     {"n_samples": 1000, "n_groups": 20, "n_test_groups": 10})
    assert err and "splits" in out["error"]


def test_cpcv_summary_huge_group_count_refused_without_computing_binomial():
    out, err = _call("cpcv_split_summary",
                     {"n_samples": 1000, "n_groups": 10**18, "n_test_groups": 5 * 10**17})
    assert err and "splits" in out["error"]


def test_notification_method_with_id_gets_a_reply() -> None:
    # A request with an id sent to a notifications/* method must get a reply; a client waiting
    # on that id would otherwise hang. Only messages without an id may be silent.
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 5, "method": "notifications/initialized"})
    assert reply is not None and reply["id"] == 5
    assert reply["error"]["code"] == -32601
    assert srv.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_symbol_list_rejects_non_integer_floats():
    # Non-integer floats such as 1.5 and 2.5 must be refused as symbols, since the schema says
    # string or integer. NaN and infinity are left to the MCP number-handling checks.
    out, err = _call("kontoyiannis_entropy", {"message": [1.5, 2.5, 1.5, 2.5, 1.5, 2.5]})
    assert err and "string or integer" in out["error"]
    out, err = _call("kontoyiannis_entropy", {"message": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]})
    assert not err


def test_cpcv_summary_bounds_samples_times_splits_not_splits_alone() -> None:
    # The cap must cover samples times splits: n = 100000 with 4950 splits would materialize
    # about 3.9 GB of index arrays in one request.
    out, err = _call("cpcv_split_summary", {"n_samples": 100000, "n_groups": 100, "n_test_groups": 2})
    assert err and "n_samples" in out["error"]


BATCH_ERROR = {"jsonrpc": "2.0", "id": None,
               "error": {"code": -32600, "message": "batch requests are not supported"}}


def test_batch_array_gets_one_error_and_no_element_is_processed():
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "ping"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    out = io.StringIO()
    srv.main(io.StringIO(json.dumps(batch) + "\n"), out)
    replies = out.getvalue().splitlines()
    assert [json.loads(x) for x in replies] == [BATCH_ERROR]


def test_empty_batch_array_gets_the_batch_error():
    assert srv.handle_message([]) == BATCH_ERROR


@pytest.mark.parametrize("response", [
    {"jsonrpc": "2.0", "id": 5, "result": {"tools": []}},
    {"jsonrpc": "2.0", "id": 6, "error": {"code": -32601, "message": "method not found"}},
])
def test_client_response_gets_no_reply(response):
    assert srv.handle_message(response) is None


def test_stdio_serves_ping_and_tools_list_around_batches_and_responses():
    lines = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        json.dumps([{"jsonrpc": "2.0", "id": 2, "method": "ping"}]),
        json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 4, "error": {"code": -32000, "message": "x"}}),
        json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/list"}),
    ]) + "\n"
    out = io.StringIO()
    srv.main(io.StringIO(lines), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 5]
    assert replies[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert replies[1] == BATCH_ERROR
    assert {t["name"] for t in replies[2]["result"]["tools"]} >= {"sharpe_ratio"}


def test_cpcv_split_summary_description_states_the_work_limit() -> None:
    # The work cap (n_samples x splits <= 20,000,000) refuses some requests, so the tool schema
    # must say so. The cap stays (it bounds server work); the limit is stated in the tool
    # description, so the refusal is not a surprise.
    from finlab.agent.tools import MAX_CPCV_WORK, list_tools

    tool = next(t for t in list_tools() if t["name"] == "cpcv_split_summary")
    assert f"{MAX_CPCV_WORK:,}" in tool["description"]


@pytest.mark.parametrize("bad_id", [{"a": 1}, [1], True, False])
def test_server_rejects_request_id_that_is_not_string_number_or_null(bad_id) -> None:
    # Ids of other types must be refused: the reply echoed any type, so a client could not match it.
    # JSON-RPC allows a string, a number or null; anything else is an invalid request.
    from finlab.agent.server import handle_message

    reply = handle_message({"jsonrpc": "2.0", "id": bad_id, "method": "ping"})
    assert reply["error"]["code"] == -32600
    assert reply["id"] is None


def test_server_still_echoes_valid_request_ids() -> None:
    from finlab.agent.server import handle_message

    assert handle_message({"jsonrpc": "2.0", "id": "x", "method": "ping"})["id"] == "x"
    assert handle_message({"jsonrpc": "2.0", "id": 7, "method": "ping"})["id"] == 7
    assert handle_message({"jsonrpc": "2.0", "id": None, "method": "ping"})["id"] is None


def test_ragged_matrix_gets_the_named_rectangular_error() -> None:
    # A ragged matrix must get the named rectangular error, not numpy's inhomogeneous-shape message.
    # _mat already names the rectangular rule for the same input, and the tool error names it too.
    from finlab.agent.tools import call_tool

    result, is_error = call_tool("hrp_weights", {"cov": [[1, 0], [0]]})
    assert is_error is True
    assert "rectangular" in result["error"]
    assert "inhomogeneous" not in result["error"]


def test_id_that_parses_to_infinity_is_refused_and_reply_is_strict_json():
    # 1e400 is a JSON number that Python reads as inf. _valid_id must refuse it, since the reply
    # would write id Infinity, which strict JSON readers reject. It is an invalid id (id null).
    import json as _json

    msg = _json.loads('{"jsonrpc":"2.0","id":1e400,"method":"ping"}')
    reply = srv.handle_message(msg)
    assert reply["error"]["code"] == -32600
    assert reply["id"] is None
    out = io.StringIO()
    srv.main(io.StringIO('{"jsonrpc":"2.0","id":1e400,"method":"ping"}\n'), out)
    assert _strict_loads(out.getvalue().splitlines()[0])["id"] is None


def test_batch_of_only_notifications_gets_no_reply() -> None:
    # JSON-RPC sends no response for a batch made only of notifications. The
    # batch error is kept for batches that contain requests, and for an empty batch.
    import io

    from finlab.agent import server

    out = io.StringIO()
    server.main(stdin=io.StringIO('[{"jsonrpc":"2.0","method":"notifications/initialized"}]\n'), stdout=out)
    assert out.getvalue() == ""
    assert server.handle_message([]) is not None
    reply = server.handle_message([{"jsonrpc": "2.0", "id": 1, "method": "ping"}])
    assert reply is not None and reply["error"]["code"] == -32600


def test_plug_in_entropy_tool_description_names_bits_per_symbol() -> None:
    # The tool must not say "bits per word": plug_in_entropy divides by word_length, so the value
    # is bits per symbol (the library docstring). An agent that reads the schema gets the unit
    # wrong by a factor of word_length. The description matches the value.
    from finlab.agent.tools import list_tools
    from finlab.entropy import plug_in_entropy

    tool = next(t for t in list_tools() if t["name"] == "plug_in_entropy")
    assert "bits per symbol" in tool["description"]
    assert "bits per word" not in tool["description"]
    # The value is the per-symbol figure the description states (1.9502 / 2 for word_length 2).
    seq = [1, 2, 3, 4, 1, 2, 3, 4]
    assert plug_in_entropy(seq, word_length=2) == pytest.approx(0.9751060324573735)


def test_psr_and_dsr_tool_descriptions_state_per_period_input() -> None:
    # The sharpe_ratio tool annualises by default, so chaining it unchanged inflates sr_hat.
    from finlab.agent.tools import list_tools

    for name in ("probabilistic_sharpe_ratio", "deflated_sharpe_ratio"):
        tool = next(t for t in list_tools() if t["name"] == name)
        assert "per period" in tool["description"]
        assert "per period" in tool["inputSchema"]["properties"]["sr_hat"]["description"]


def test_unknown_key_is_refused_by_name():
    # 'kurtosis' is not declared by deflated_sharpe_ratio (it is 'kurt'), and
    # 'annualise' is not declared by sharpe_ratio ('annualize'). Both must be refused, not
    # dropped silently, since a dropped key returned isError false.
    out, err = _call("deflated_sharpe_ratio",
                     {"sr_hat": 0.3, "n_trials": 40, "var_sr": 0.02, "n_obs": 250, "kurtosis": 6})
    assert err
    assert "kurtosis" in out["error"]
    assert "kurt" in out["error"] and "n_trials" in out["error"]
    out, err = _call("sharpe_ratio", {"returns": [0.01, -0.005, 0.02], "annualise": False})
    assert err
    assert "annualise" in out["error"]
    assert "annualize" in out["error"] and "returns" in out["error"]


def test_declared_keys_still_work():
    out, err = _call("deflated_sharpe_ratio",
                     {"sr_hat": 0.3, "n_trials": 40, "var_sr": 0.02, "n_obs": 250,
                      "skew": -0.5, "kurt": 6})
    assert not err
    assert out["dsr"] == pytest.approx(deflated_sharpe_ratio(0.3, 40, 0.02, 250, -0.5, 6.0), rel=1e-12)
    r = [0.01, -0.005, 0.02, 0.004]
    out, err = _call("sharpe_ratio", {"returns": r, "annualize": False})
    assert not err
    assert out["sharpe"] == pytest.approx(sharpe_ratio(np.array(r), annualize=False), rel=1e-12)


def _pbo_request(matrix, req_id=1):
    args = {"performance_matrix": matrix.tolist(), "n_partitions": 8}
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": "probability_of_backtest_overfitting", "arguments": args},
    }


def test_pbo_equity_curve_warning_reaches_reply():
    rng = np.random.default_rng(21)
    returns = rng.normal(0, 0.01, (240, 12)).round(6)
    equity = np.cumsum(returns, axis=0).round(6)  # cumulative curves, not per-period returns
    reply = srv.handle_message(_pbo_request(equity))
    assert reply["result"]["isError"] is False
    body = json.loads(reply["result"]["content"][0]["text"])
    assert "pbo" in body and "n_combinations" in body
    assert body["warnings"]
    assert any("cumulative" in w or "equity" in w for w in body["warnings"])


def test_clean_reply_has_no_warnings_key():
    rng = np.random.default_rng(21)
    returns = rng.normal(0, 0.01, (240, 12)).round(6)
    reply = srv.handle_message(_pbo_request(returns))
    assert reply["result"]["isError"] is False
    body = json.loads(reply["result"]["content"][0]["text"])
    assert "warnings" not in body
    assert body["pbo"] == pytest.approx(0.5142857142857142, rel=1e-12)


def test_version_strings_match_pyproject():
    import re
    from pathlib import Path

    import finlab

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert finlab.__version__ == declared
    result = _rpc("initialize")["result"]
    assert result["serverInfo"]["version"] == declared


def test_nan_in_arguments_parse_error_reply_keeps_request_id():
    line = (
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"sharpe_ratio",'
        '"arguments":{"returns":[0.01,NaN,0.02,-0.01]}}}'
    )
    out = io.StringIO()
    srv.main(io.StringIO(line + "\n"), out)
    reply = _strict_loads(out.getvalue().splitlines()[0])
    assert reply["id"] == 7
    assert reply["error"]["code"] == -32700
    assert reply["error"]["message"] == "parse error: non-standard constant NaN"


def _strict_call(name, arguments):
    """Like _call, but the reply text must be strict JSON: no NaN or Infinity tokens."""
    reply = _rpc("tools/call", {"name": name, "arguments": arguments})
    result = reply["result"]
    return _strict_loads(result["content"][0]["text"]), result["isError"]


def test_hrp_and_bet_size_replies_turn_non_finite_values_into_null(monkeypatch):
    # The library output is replaced here, since no input found so far makes these calls return a
    # non-finite value. Each non-finite float must still come back as null, and the text must be strict JSON.
    monkeypatch.setattr(tl, "hrp_weights", lambda cov, method="single": np.array([np.nan, 0.25, np.inf]))
    monkeypatch.setattr(tl, "prob_bet_size", lambda p, side, k: np.array([-np.inf, np.nan, 0.5]))
    cov = [[1.0, 0.3, 0.1], [0.3, 2.0, 0.2], [0.1, 0.2, 0.5]]
    out, err = _strict_call("hrp_weights", {"cov": cov})
    assert not err and out == {"weights": [None, 0.25, None]}
    out, err = _strict_call("bet_size_from_probability", {"probability": [0.6, 0.7, 0.8]})
    assert not err and out == {"bet_size": [None, None, 0.5]}


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("arguments", [
    '{"returns":[0.01,TOKEN,0.02,-0.01]}',
    '{"returns":[0.01,0.02,-0.01],"periods_per_year":TOKEN}',
])
def test_non_finite_literal_in_arguments_takes_the_nan_parse_error_path(token, arguments):
    # The whole line fails to parse, so the position of the literal does not change the reply,
    # and every constant gets the same reply shape with the request id kept.
    line = ('{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"sharpe_ratio","arguments":'
            + arguments.replace("TOKEN", token) + '}}')
    out = io.StringIO()
    srv.main(io.StringIO(line + "\n"), out)
    replies = out.getvalue().splitlines()
    assert len(replies) == 1
    assert _strict_loads(replies[0]) == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32700, "message": f"parse error: non-standard constant {token}"},
    }


@pytest.mark.parametrize("arguments, message", [
    ('{"returns":[0.01,1e400,0.02]}', "'returns' must contain only finite numbers"),
    ('{"returns":[0.01,0.02,-0.01],"periods_per_year":1e400}', "'periods_per_year' must be a finite number"),
])
def test_literal_that_overflows_to_infinity_is_a_tool_error(arguments, message):
    # 1e400 is valid JSON that Python reads as inf. It passes the parser and the argument checks
    # refuse it, so the reply is a tool error that names the argument.
    line = ('{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"sharpe_ratio","arguments":'
            + arguments + '}}')
    out = io.StringIO()
    srv.main(io.StringIO(line + "\n"), out)
    reply = _strict_loads(out.getvalue().splitlines()[0])
    assert reply["id"] == 8 and reply["result"]["isError"] is True
    assert message in json.loads(reply["result"]["content"][0]["text"])["error"]
