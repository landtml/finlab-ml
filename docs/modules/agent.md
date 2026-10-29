# `quantowl.agent`: Agent access: the quantowl-mcp server

An MCP server that exposes the main computations as typed tools with compact JSON
output. Full guide, client configuration and limits are in [AGENTS.md](../AGENTS.md).

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import json
from quantowl.agent.server import handle_message

reply = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "sharpe_ratio",
                                   "arguments": {"returns": [0.01, -0.005, 0.02]}}})
print(json.loads(reply["result"]["content"][0]["text"]))
```

## Tools

Eighteen tools, grouped here by the module each one wraps. The table with chapters is in
[AGENTS.md](../AGENTS.md#tools).

| Module | Tools |
|---|---|
| `quantowl.stats` | `sharpe_ratio`, `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `min_track_record_length` |
| `quantowl.strategy_risk` | `binomial_sharpe`, `implied_precision`, `implied_frequency`, `probability_of_failure` |
| `quantowl.pbo` | `probability_of_backtest_overfitting` |
| `quantowl.fracdiff`, `quantowl.bars`, `quantowl.structural_breaks` | `frac_diff_ffd`, `cusum_filter`, `sadf` |
| `quantowl.entropy` | `kontoyiannis_entropy`, `plug_in_entropy` |
| `quantowl.hrp`, `quantowl.onc`, `quantowl.bet_sizing`, `quantowl.cv` | `hrp_weights`, `onc_clusters`, `bet_size_from_probability`, `cpcv_split_summary` |

## Notes

See [AGENTS.md](../AGENTS.md).

Full signatures: [API reference](../API.md).
