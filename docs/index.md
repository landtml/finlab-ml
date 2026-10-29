# quantowl

quantowl is a toolkit for financial machine learning that implements the methods of Marcos López de Prado's *Advances in Financial Machine Learning* (AFML), organised by chapter. Hot loops are compiled to machine code with [numba](https://numba.pydata.org/); there is no scikit-learn dependency. Most methods come with tests that cross-check them against a naive reference, and where the book states a result, a proof note in the Proof notes section marks each claim as proved here, checked by test, measured, or claimed from the book only.

## Install

```bash
pip install "git+https://github.com/landtml/quantowl@v0.1.5"
```

Requires Python 3.10 to 3.14. From a clone of the repository:

```bash
pip install -e .            # core library
pip install -e ".[dev]"     # + pytest, ruff and mypy, to run the test suite and lint
pip install -e ".[plot]"    # + plotly, for the figures in quantowl.plot
pip install -e ".[notebooks]"  # + Jupyter tooling, to run the notebooks in examples/
pytest                      # run the suite
```

## Quick start

```python
import numpy as np
import pandas as pd
from quantowl.cv import CombinatorialPurgedCV, make_t1
from quantowl.stats import sharpe_ratio, deflated_sharpe_ratio
from quantowl.bet_sizing import prob_bet_size, discretize_signal
from quantowl.hrp import hrp_weights

rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=500, freq="B")

# 1. Leakage-free cross-validation for 10-bar forward labels (AFML ch. 7)
t1 = make_t1(idx, horizon=10)
cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01, t1=t1)
X = pd.DataFrame(rng.normal(size=(500, 3)), index=idx)
print(sum(1 for _ in cv.split(X)), "purged splits")        # 15 purged splits

# 2. Backtest statistics: Sharpe and deflated Sharpe over 50 trials (AFML ch. 14)
r = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)
sr = sharpe_ratio(r, annualize=False)
print(round(deflated_sharpe_ratio(sr, n_trials=50, var_sr=0.01, n_obs=500,
                                  skew=0, kurt=3), 4))

# 3. Bet sizes from predicted probabilities, discretised (AFML ch. 10)
print(prob_bet_size([0.7, 0.2]).round(3), discretize_signal([0.33], 0.2))

# 4. Hierarchical risk parity weights (AFML ch. 16)
cov = pd.DataFrame(np.cov(rng.normal(size=(200, 5)).T),
                   columns=list("abcde"), index=list("abcde"))
print(hrp_weights(cov).round(3).to_dict())
```

## Find a method

Each row gives a method name, the chapter of the book it comes from, and the module page.

| Method | AFML chapter | Module page |
|---|---|---|
| Purged k-fold and combinatorial purged cross-validation | 7 | [`quantowl.cv`](modules/cv.md) |
| Triple-barrier labels and meta-labeling | 3 | [`quantowl.labeling`](modules/labeling.md) |
| Fractional differentiation | 5 | [`quantowl.fracdiff`](modules/fracdiff.md) |
| Sequential bootstrap and average uniqueness | 4 | [`quantowl.weights`](modules/weights.md) |
| Deflated Sharpe ratio and probabilistic Sharpe ratio | 14 | [`quantowl.stats`](modules/stats.md) |
| Probability of backtest overfitting (CSCV) | 11 | [`quantowl.pbo`](modules/pbo.md) |
| Hierarchical risk parity | 16 | [`quantowl.hrp`](modules/hrp.md) |
| SADF and other structural break tests | 17 | [`quantowl.structural_breaks`](modules/structural_breaks.md) |
| VPIN and microstructure features | 19 | [`quantowl.microstructure`](modules/microstructure.md) |
| Entropy features | 18 | [`quantowl.entropy`](modules/entropy.md) |
| Bet sizing | 10 | [`quantowl.bet_sizing`](modules/bet_sizing.md) |
| Information-driven bars | 2 | [`quantowl.bars`](modules/bars.md) |
