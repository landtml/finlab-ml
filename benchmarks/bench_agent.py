"""Agent server overhead: one tool call through the JSON-RPC layer vs the direct call.

Run: PYTHONPATH=../src python3 benchmarks/bench_agent.py   (from the repo root: PYTHONPATH=src)
"""
import json

import numpy as np

from _timing import best_of
from finlab.agent.server import handle_message
from finlab.stats import sharpe_ratio


def main():
    r = np.random.default_rng(0).normal(0.001, 0.01, 1000)
    payload = r.tolist()
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "sharpe_ratio", "arguments": {"returns": payload}}}
    handle_message(req)  # warm-up
    rpc = best_of(lambda: json.loads(json.dumps(handle_message(req))), repeats=5)
    direct = best_of(lambda: sharpe_ratio(r), repeats=5)
    print(f"n=1000 returns: direct {direct * 1e6:.1f} us, via JSON-RPC {rpc * 1e6:.1f} us "
          f"({rpc / direct:.0f}x, dominated by JSON encoding of the input)")


if __name__ == "__main__":
    main()
