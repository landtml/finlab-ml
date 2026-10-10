# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deflated Sharpe ratio after many trials
#
# A researcher who runs many backtests and keeps the best one reports a Sharpe
# ratio that the selection has inflated. This notebook shows the effect with
# strategies that have no edge at all: zero-mean daily returns.
#
# The steps are:
#
# 1. Register 40 pure-noise strategies in a `TrialRegistry`.
# 2. Pick the best one and compare its Sharpe ratio with the naive PSR against zero.
# 3. Deflate it for the 40 trials with `registry.deflated_sharpe()`, where N comes from the registry.
# 4. Register 40 more pure-noise strategies and deflate again.
# 5. Round-trip the registry through JSON and check that nothing changed.
#
# Run it from a checkout with `finlab` installed (`pip install -e .`).

# %% [markdown]
# ## Setup
#
# The seed is fixed, so every run gives the same numbers.

# %%
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from finlab.stats import expected_max_sharpe, probabilistic_sharpe_ratio, sharpe_ratio
from finlab.trials import TrialRegistry

SEED = 20261010
T = 500  # daily return observations per strategy
N_FIRST = 40  # strategies registered in step 1
N_SECOND = 40  # strategies added in step 4
SIGMA = 0.01  # daily volatility of the noise; the mean is zero

rng = np.random.default_rng(SEED)
returns = rng.normal(0.0, SIGMA, size=(T, N_FIRST + N_SECOND))

# %% [markdown]
# ## 1. Register 40 strategies of pure noise
#
# Each strategy is one column of `returns`. The registry keeps the name, the
# parameters and the return series of every trial. `n_trials` is the N used by
# the deflated Sharpe ratio.

# %%
reg = TrialRegistry()
for i in range(N_FIRST):
    reg.record(f"noise-{i:02d}", {"draw": i}, returns[:, i])

print("registered trials:", reg.n_trials)


# %% [markdown]
# ## 2. The best trial
#
# The selected trial is the one with the highest non-annualised Sharpe ratio,
# which is the rule `TrialRegistry.deflated_sharpe` applies. The annualised
# value is shown for reference.


# %%
def non_annualised_srs(registry):
    return np.array([sharpe_ratio(rec.returns, annualize=False) for rec in registry.records])


srs = non_annualised_srs(reg)
best = int(np.argmax(srs))
best_returns = np.asarray(reg.records[best].returns)
sr_hat = float(srs[best])
g3 = float(skew(best_returns))
g4 = float(kurtosis(best_returns, fisher=False))

print(f"best trial:              {reg.records[best].name}")
print(f"best annualised Sharpe:  {sharpe_ratio(best_returns):.4f}")
print(f"best non-annualised SR:  {sr_hat:.4f}")

# %% [markdown]
# ## 3. Naive PSR against zero and the deflated Sharpe ratio
#
# The naive probabilistic Sharpe ratio tests the best trial against zero skill
# and ignores the other 39 trials. The deflated Sharpe ratio tests it against
# the expected maximum Sharpe ratio of N zero-skill trials, SR*. Both use the
# same observed Sharpe ratio, sample length, skewness and kurtosis.

# %%
n_trials = reg.n_trials
psr_vs_zero = probabilistic_sharpe_ratio(sr_hat, 0.0, T, g3, g4)
var_sr = float(np.var(srs, ddof=1))
sr_star = expected_max_sharpe(n_trials, var_sr)
dsr = reg.deflated_sharpe()

# The registry's DSR is the PSR evaluated at SR*, with the same inputs.
assert abs(dsr - probabilistic_sharpe_ratio(sr_hat, sr_star, T, g3, g4)) < 1e-12
assert dsr < psr_vs_zero

print(f"N trials:                {n_trials}")
print(f"expected max SR, SR*:    {sr_star:.4f}")
print(f"naive PSR vs zero:       {psr_vs_zero:.4f}")
print(f"deflated Sharpe ratio:   {dsr:.4f}")

# %% [markdown]
# ## 4. Add 40 more pure-noise trials
#
# The same registry now holds 80 trials, so N is 80 without any number typed
# in. The benchmark SR* rises with N. The best observed trial can also change,
# and its Sharpe ratio can rise too. A higher best Sharpe ratio offsets part of
# the penalty, so the DSR can go up as well as down. Change `SEED` to see that.

# %%
for i in range(N_FIRST, N_FIRST + N_SECOND):
    reg.record(f"noise-{i:02d}", {"draw": i}, returns[:, i])

srs_after = non_annualised_srs(reg)
best_after = int(np.argmax(srs_after))
dsr_after_80 = reg.deflated_sharpe()

print(f"N trials:                {reg.n_trials}")
print(f"best trial:              {reg.records[best_after].name}")
print(f"best non-annualised SR:  {srs_after[best_after]:.4f}")
print(f"deflated Sharpe ratio:   {dsr_after_80:.4f}")

# %% [markdown]
# ## 5. JSON round trip
#
# `to_json` writes every trial with its parameters and returns, with floats at
# repr precision. `from_json` rebuilds the registry. The check compares names,
# parameters and returns exactly, the serialised text, and the deflated Sharpe
# ratio.


# %%
def same_registry(a, b):
    if a.n_trials != b.n_trials:
        return False
    for ra, rb in zip(a.records, b.records):
        if ra.name != rb.name or ra.params_dict != rb.params_dict:
            return False
        if not np.array_equal(ra.returns, rb.returns):
            return False
    return True


text = reg.to_json()
restored = TrialRegistry.from_json(text)
roundtrip_ok = bool(
    same_registry(reg, restored)
    and restored.to_json() == text
    and restored.deflated_sharpe() == dsr_after_80
)
assert roundtrip_ok, "JSON round trip changed the registry"

print(f"JSON size: {len(text):,} characters for {reg.n_trials} trials")
print("round trip equal:", roundtrip_ok)

# %% [markdown]
# ## Key values
#
# `KEY` collects the results that `tests/test_example_deflated_sharpe.py` checks.

# %%
KEY = {
    "n_trials": n_trials,
    "dsr": dsr,
    "psr_vs_zero": psr_vs_zero,
    "dsr_after_80": dsr_after_80,
    "roundtrip_ok": roundtrip_ok,
}
