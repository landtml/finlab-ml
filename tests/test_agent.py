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
