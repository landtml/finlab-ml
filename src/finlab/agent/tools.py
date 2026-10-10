"""Tool registry for the finlab agent server.

Every tool is a pure function from JSON arguments to a JSON-serialisable dict. Inputs
are checked against a size limit before any array is allocated, so one request
cannot exhaust memory. Output contains only the answer, not the input echoed back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..bars import cusum_filter
from ..bet_sizing import prob_bet_size
from ..cv import CombinatorialPurgedCV, make_t1
from ..entropy import kontoyiannis_entropy, plug_in_entropy
from ..fracdiff import frac_diff_ffd
from ..hrp import hrp_weights
from ..onc import onc
from ..pbo import probability_of_backtest_overfitting
from ..stats import (
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from ..structural_breaks import sadf

MAX_ELEMENTS = 5_000_000  # guard: refuse inputs larger than this many numbers


class ToolError(ValueError):
    """Bad arguments for a tool; reported to the agent as an error result."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


TOOLS: dict[str, Tool] = {}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]):
    def wrap(fn: Callable[[dict[str, Any]], dict[str, Any]]):
        TOOLS[name] = Tool(
            name=name,
            description=description,
            schema={"type": "object", "properties": properties, "required": required},
            handler=fn,
        )
        return fn

    return wrap


def _vec(args: dict[str, Any], key: str) -> np.ndarray:
    v = args.get(key)
    if not isinstance(v, list) or not v:
        raise ToolError(f"'{key}' must be a non-empty array of numbers")
    if len(v) > MAX_ELEMENTS:
        raise ToolError(f"'{key}' has {len(v)} elements; the limit is {MAX_ELEMENTS}")
    arr = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ToolError(f"'{key}' must contain only finite numbers")
    return arr


def _mat(args: dict[str, Any], key: str) -> np.ndarray:
    v = args.get(key)
    if not isinstance(v, list) or not v or not all(isinstance(r, list) for r in v):
        raise ToolError(f"'{key}' must be a non-empty array of rows")
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 2 or arr.size > MAX_ELEMENTS:
        raise ToolError(f"'{key}' must be a rectangular matrix with at most {MAX_ELEMENTS} entries")
    if not np.all(np.isfinite(arr)):
        raise ToolError(f"'{key}' must contain only finite numbers")
    return arr


def _num(args: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in args:
        if default is None:
            raise ToolError(f"'{key}' is required")
        return float(default)
    v = args[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ToolError(f"'{key}' must be a finite number")
    return float(v)


def _int(args: dict[str, Any], key: str, default: int | None = None) -> int:
    v = _num(args, key, default)
    if v != int(v):
        raise ToolError(f"'{key}' must be an integer")
    return int(v)


def _clean(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


# --------------------------------------------------------------------------- #
# Backtest statistics (AFML ch. 14, 11-12)                                    #
# --------------------------------------------------------------------------- #
@_tool(
    "sharpe_ratio",
    "Sharpe ratio of a return series. Annualised by default.",
    {
        "returns": {"type": "array", "items": {"type": "number"}},
        "periods_per_year": {"type": "number", "default": 252},
        "annualize": {"type": "boolean", "default": True},
    },
    ["returns"],
)
def _sharpe(args):
    r = _vec(args, "returns")
    sr = sharpe_ratio(r, _num(args, "periods_per_year", 252), annualize=bool(args.get("annualize", True)))
    return {"sharpe": _clean(float(sr))}


@_tool(
    "probabilistic_sharpe_ratio",
    "Probability that the true Sharpe ratio exceeds a benchmark, given sample size, skew and kurtosis.",
    {
        "sr_hat": {"type": "number"},
        "sr_benchmark": {"type": "number", "default": 0},
        "n_obs": {"type": "integer"},
        "skew": {"type": "number", "default": 0},
        "kurtosis": {"type": "number", "default": 3},
    },
    ["sr_hat", "n_obs"],
)
def _psr(args):
    p = probabilistic_sharpe_ratio(
        _num(args, "sr_hat"), _num(args, "sr_benchmark", 0), _int(args, "n_obs"),
        _num(args, "skew", 0), _num(args, "kurtosis", 3),
    )
    return {"psr": _clean(float(p))}


@_tool(
    "deflated_sharpe_ratio",
    "Probability the Sharpe ratio is genuine after correcting for n_trials tried strategies.",
    {
        "sr_hat": {"type": "number"},
        "n_trials": {"type": "integer"},
        "var_sr": {"type": "number", "description": "variance of Sharpe ratios across trials"},
        "n_obs": {"type": "integer"},
        "skew": {"type": "number", "default": 0},
        "kurt": {"type": "number", "default": 3},
    },
    ["sr_hat", "n_trials", "var_sr", "n_obs"],
)
def _dsr(args):
    d = deflated_sharpe_ratio(
        _num(args, "sr_hat"), _int(args, "n_trials"), _num(args, "var_sr"), _int(args, "n_obs"),
        _num(args, "skew", 0), _num(args, "kurt", 3),
    )
    return {"dsr": _clean(float(d))}


@_tool(
    "min_track_record_length",
    "Minimum number of observations for the Sharpe ratio to exceed the benchmark with the given probability.",
    {
        "sr_hat": {"type": "number"},
        "sr_benchmark": {"type": "number", "default": 0},
        "skew": {"type": "number", "default": 0},
        "kurt": {"type": "number", "default": 3},
        "prob": {"type": "number", "default": 0.95},
    },
    ["sr_hat"],
)
def _minTRL(args):
    v = min_track_record_length(
        _num(args, "sr_hat"), _num(args, "sr_benchmark", 0), _num(args, "skew", 0),
        _num(args, "kurt", 3), _num(args, "prob", 0.95),
    )
    return {"min_track_record_length": _clean(float(v))}


@_tool(
    "probability_of_backtest_overfitting",
    "PBO by combinatorially symmetric cross-validation. Input: rows = time, columns = trial strategies' returns.",
    {
        "performance_matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "n_partitions": {"type": "integer", "default": 8},
    },
    ["performance_matrix"],
)
def _pbo(args):
    m = _mat(args, "performance_matrix")
    res = probability_of_backtest_overfitting(m, n_partitions=_int(args, "n_partitions", 8))
    return {"pbo": _clean(float(res.pbo)), "n_combinations": int(res.n_combinations)}


# --------------------------------------------------------------------------- #
# Features, filters, structural breaks, entropy                              #
# --------------------------------------------------------------------------- #
@_tool(
    "frac_diff_ffd",
    "Fixed-window fractional differentiation of a series. Keeps memory while making it closer to stationary.",
    {
        "series": {"type": "array", "items": {"type": "number"}},
        "d": {"type": "number", "description": "differencing order in [0, 1]"},
        "thres": {"type": "number", "default": 1e-5},
    },
    ["series", "d"],
)
def _fracdiff(args):
    s = _vec(args, "series")
    d = _num(args, "d")
    if not 0.0 <= d <= 1.0:
        raise ToolError("'d' must be in [0, 1]")
    out = frac_diff_ffd(s, d, thres=_num(args, "thres", 1e-5))
    return {"values": [_clean(float(x)) for x in np.asarray(out)]}


@_tool(
    "cusum_filter",
    "Symmetric CUSUM event filter: positions where cumulative change exceeds the threshold.",
    {
        "series": {"type": "array", "items": {"type": "number"}},
        "threshold": {"type": "number"},
        "is_returns": {"type": "boolean", "default": False},
    },
    ["series", "threshold"],
)
def _cusum(args):
    s = _vec(args, "series")
    idx = cusum_filter(s, _num(args, "threshold"), is_returns=bool(args.get("is_returns", False)))
    return {"event_positions": [int(i) for i in np.asarray(idx)]}


@_tool(
    "sadf",
    "Supremum augmented Dickey-Fuller statistic over expanding windows; large values indicate explosive behaviour (bubbles).",
    {
        "series": {"type": "array", "items": {"type": "number"}},
        "min_length": {"type": "integer"},
        "lags": {"type": "integer", "default": 1},
    },
    ["series", "min_length"],
)
def _sadf(args):
    s = _vec(args, "series")
    out = sadf(s, min_length=_int(args, "min_length"), lags=_int(args, "lags", 1))
    arr = np.asarray(out, dtype=np.float64).ravel()
    return {"sadf": _clean(float(np.nanmax(arr))) if arr.size else None, "n_points": int(arr.size)}


@_tool(
    "kontoyiannis_entropy",
    "Kontoyiannis entropy rate estimate of a symbol sequence (bits per symbol).",
    {"message": {"type": "array", "items": {"type": ["string", "integer"]}}},
    ["message"],
)
def _kont(args):
    msg = args.get("message")
    if not isinstance(msg, list) or not msg:
        raise ToolError("'message' must be a non-empty array of symbols")
    if len(msg) > MAX_ELEMENTS:
        raise ToolError(f"'message' is longer than {MAX_ELEMENTS}")
    return {"entropy": _clean(float(kontoyiannis_entropy(msg)))}


@_tool(
    "plug_in_entropy",
    "Plug-in (maximum-likelihood) entropy of a symbol sequence, in bits per word.",
    {
        "message": {"type": "array", "items": {"type": ["string", "integer"]}},
        "word_length": {"type": "integer", "default": 1},
    },
    ["message"],
)
def _plugin(args):
    msg = args.get("message")
    if not isinstance(msg, list) or not msg:
        raise ToolError("'message' must be a non-empty array of symbols")
    if len(msg) > MAX_ELEMENTS:
        raise ToolError(f"'message' is longer than {MAX_ELEMENTS}")
    return {"entropy": _clean(float(plug_in_entropy(msg, word_length=_int(args, "word_length", 1))))}


# --------------------------------------------------------------------------- #
# Portfolio, clustering, bet sizing, validation                              #
# --------------------------------------------------------------------------- #
@_tool(
    "hrp_weights",
    "Hierarchical risk parity portfolio weights from a covariance matrix.",
    {
        "cov": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "method": {"type": "string", "enum": ["single", "complete", "average", "weighted"], "default": "single"},
    },
    ["cov"],
)
def _hrp(args):
    cov = _mat(args, "cov")
    if cov.shape[0] != cov.shape[1]:
        raise ToolError("'cov' must be square")
    w = hrp_weights(cov, method=str(args.get("method", "single")))
    return {"weights": [float(x) for x in np.asarray(w)]}


@_tool(
    "onc_clusters",
    "Cluster variables from a correlation matrix, choosing the number of clusters by silhouette (ONC).",
    {
        "corr": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "max_k": {"type": "integer"},
        "seed": {"type": "integer", "default": 0},
    },
    ["corr"],
)
def _onc(args):
    c = _mat(args, "corr")
    max_k = args.get("max_k")
    res = onc(c, max_k=None if max_k is None else int(max_k), seed=_int(args, "seed", 0))
    return {
        "labels": [int(x) for x in res.labels],
        "n_clusters": int(res.n_clusters),
        "silhouette": _clean(float(res.silhouette)),
    }


@_tool(
    "bet_size_from_probability",
    "Bet sizes in [-1, 1] from predicted probabilities (AFML 10.3).",
    {
        "probability": {"type": "array", "items": {"type": "number"}},
        "side": {"type": "array", "items": {"type": "number"}},
        "num_classes": {"type": "integer", "default": 2},
    },
    ["probability"],
)
def _bet(args):
    p = _vec(args, "probability")
    side = args.get("side")
    side_arr = None if side is None else _vec({"side": side}, "side")
    return {"bet_size": [float(x) for x in prob_bet_size(p, side_arr, _int(args, "num_classes", 2))]}


@_tool(
    "cpcv_split_summary",
    "Combinatorial purged CV geometry for a sample: number of splits, paths, and training sizes after purge and embargo. Does not return indices.",
    {
        "n_samples": {"type": "integer"},
        "n_groups": {"type": "integer"},
        "n_test_groups": {"type": "integer"},
        "horizon": {"type": "integer", "default": 1, "description": "label horizon in bars"},
        "embargo_pct": {"type": "number", "default": 0},
    },
    ["n_samples", "n_groups", "n_test_groups"],
)
def _cpcv(args):
    n = _int(args, "n_samples")
    if n > MAX_ELEMENTS:
        raise ToolError(f"n_samples above {MAX_ELEMENTS}")
    horizon = _int(args, "horizon", 1)
    import pandas as pd

    index = pd.RangeIndex(n)
    t1 = make_t1(index, horizon) if horizon >= 1 else None
    cv = CombinatorialPurgedCV(
        n_groups=_int(args, "n_groups"),
        n_test_groups=_int(args, "n_test_groups"),
        embargo_pct=_num(args, "embargo_pct", 0),
        t1=t1,
        min_train_size=0,
    )
    train_sizes = []
    test_sizes = []
    for tr, te in cv.split(index):
        train_sizes.append(int(len(tr)))
        test_sizes.append(int(len(te)))
    return {
        "n_splits": cv.get_n_splits(),
        "n_paths": cv.get_n_paths(),
        "train_size_min": min(train_sizes),
        "train_size_max": max(train_sizes),
        "test_size_min": min(test_sizes),
        "test_size_max": max(test_sizes),
    }


def list_tools() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "inputSchema": t.schema}
        for t in TOOLS.values()
    ]


def call_tool(name: str, arguments: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """Run a tool. Returns (result, is_error). Never raises for bad input."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool '{name}'"}, True
    try:
        return tool.handler(arguments or {}), False
    except ToolError as exc:
        return {"error": str(exc)}, True
    except (ValueError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}, True
