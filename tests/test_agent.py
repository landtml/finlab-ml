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


def test_non_numeric_element_names_argument_index_and_type():
    out, err = _call("sharpe_ratio", {"returns": [0.1, "x"]})
    assert err and "element 1" in out["error"] and "str" in out["error"]
    assert "'returns'" in out["error"]
    out, err = _call("sharpe_ratio", {"returns": ["0.1", "x"]})
    assert err and "element 0" in out["error"] and "str" in out["error"]


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


@pytest.mark.parametrize("params", [[], 0, "x", [1]])
def test_params_that_are_not_an_object_are_invalid_params(params):
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 11, "method": "ping", "params": params})
    assert reply["id"] == 11
    assert reply["error"] == {"code": -32602, "message": "invalid params: expected an object"}


@pytest.mark.parametrize("params", [{}, {"arguments": {}}, {"name": None}, {"name": 5}])
def test_tools_call_without_string_name_says_missing_tool_name(params):
    reply = srv.handle_message({"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": params})
    assert reply["id"] == 12
    assert reply["error"]["code"] == -32602
    assert "missing tool name" in reply["error"]["message"]


def test_cpcv_rejects_non_positive_n_samples():
    out, err = _call("cpcv_split_summary", {"n_samples": -5, "n_groups": 6, "n_test_groups": 2})
    assert err and "n_samples must be a positive integer, got -5" in out["error"]
