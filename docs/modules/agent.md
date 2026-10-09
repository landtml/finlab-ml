# `finlab.agent`: Agent access: the finlab-mcp server

An MCP server that exposes the main computations as typed tools with compact JSON
output. Full guide, client configuration and limits are in [AGENTS.md](../AGENTS.md).

## Example

Runs as written; `tests/test_doc_examples.py` executes it on every test run.

```python
import json
from finlab.agent.server import handle_message

reply = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "sharpe_ratio",
                                   "arguments": {"returns": [0.01, -0.005, 0.02]}}})
print(json.loads(reply["result"]["content"][0]["text"]))
```

## Notes

See [AGENTS.md](../AGENTS.md).

Full signatures: [API reference](../API.md).
