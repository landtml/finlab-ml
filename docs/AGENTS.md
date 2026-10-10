# Using finlab from an AI agent

finlab ships an MCP server (`finlab-mcp`) that exposes its main computations as
typed tools. An agent calls a tool with JSON arguments and gets compact JSON back.
The server needs no extra dependency and runs locally over stdio.

## Install and register

```bash
pip install -e .            # provides the finlab-mcp command
```

Add it to an MCP client's configuration, for example:

```json
{
  "mcpServers": {
    "finlab": { "command": "finlab-mcp" }
  }
}
```

Or run it directly: `python -m finlab.agent` (reads JSON-RPC lines on stdin).

## Tools

| Tool | Purpose | Chapter |
|---|---|---|
| `sharpe_ratio` | Sharpe ratio of returns | 14 |
| `probabilistic_sharpe_ratio` | P(true SR > benchmark) | 14 |
| `deflated_sharpe_ratio` | SR corrected for n trials | 14 |
| `min_track_record_length` | Observations needed for a given confidence | 14 |
| `probability_of_backtest_overfitting` | PBO by CSCV on a returns matrix | 11 |
| `frac_diff_ffd` | Fixed-window fractional differentiation | 5 |
| `cusum_filter` | Symmetric CUSUM event positions | 2 |
| `sadf` | Supremum ADF statistic (bubble detection) | 17 |
| `kontoyiannis_entropy` | Entropy rate of a symbol sequence | 18 |
| `plug_in_entropy` | Plug-in entropy of a symbol sequence | 18 |
| `hrp_weights` | Hierarchical risk parity weights | 16 |
| `onc_clusters` | Clusters from a correlation matrix (ONC) | not AFML |
| `bet_size_from_probability` | Bet sizes in [-1, 1] | 10 |
| `cpcv_split_summary` | CPCV split counts and sizes (no indices) | 7 |

Call `tools/list` for the full JSON schemas.

## Example exchange

Request (one line on stdin):

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"deflated_sharpe_ratio","arguments":{"sr_hat":0.3,"n_trials":40,"var_sr":0.02,"n_obs":250}}}
```

Response (one line on stdout):

```json
{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{\"dsr\":0.9...}"}],"isError":false}}
```

Bad input returns `isError: true` with an `error` message, not a protocol error, so
the agent can read the reason and retry.

## Limits and efficiency

- **Size guard.** Any single array above 5,000,000 numbers is refused before it is
  allocated.
- **Compact output.** Results contain only the answer. `cpcv_split_summary` returns
  counts, not index arrays, for that reason.
- **JSON transport.** Arguments travel as JSON text, which is slow for very large
  arrays. For big inputs the computation itself is fast, but the parsing dominates.
  A file-path input is not implemented yet.
- **No ADF in the server.** `find_min_d` needs an injected ADF test and is not
  exposed, because no ADF test is bundled.
- **First call is slower.** numba compiles each kernel on its first use and caches it
  on disk.

## Verification

`tests/test_agent.py` checks each tool's output against a direct library call, the
error paths, and a real subprocess round trip over stdio.
