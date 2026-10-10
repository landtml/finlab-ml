# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Probability of backtest overfitting (PBO)
#
# Given the return series of many candidate strategies, PBO estimates how often
# the strategy that looks best in sample ranks at or below the median out of
# sample. The estimate comes from combinatorially symmetric cross-validation
# (CSCV, AFML ch. 11): the T x N return matrix is cut into S contiguous row
# blocks, and every choice of S/2 blocks is used once as the training set.
#
# This notebook uses `finlab.pbo.probability_of_backtest_overfitting` on two
# simulated matrices with a fixed seed:
#
# 1. pure noise, where no strategy has any edge;
# 2. one real strategy with a small positive mean among noise strategies.
#
# It runs in a few seconds with the base finlab install. The plot is optional.

# %%
import numpy as np

from finlab.pbo import probability_of_backtest_overfitting

T = 320  # time observations (rows)
N = 60  # candidate strategies (columns)
S = 8  # contiguous row blocks, so C(S, S/2) = C(8, 4) = 70 splits
SIGMA = 0.01  # per-period standard deviation of every column
EDGE = 0.003  # per-period mean of the one real strategy (about 0.3 SD)
SEED_NOISE = 0
SEED_EDGE = 1

# %% [markdown]
# ## 1. Pure noise
#
# Every column is an independent draw from the same distribution, with mean zero.
# For each split the in-sample
# winner's out-of-sample rank is then uniform over the N trials, so the chance
# that its logit is at or below zero is floor((N+1)/2) / N. For even N this is
# exactly 1/2, so the expected PBO under pure noise is 1/2, not 1. The derivation
# is in `docs/proofs/pbo.md`, section 3.
#
# A single matrix is one draw. The splits share blocks, so the PBO of one matrix
# can sit well away from 1/2. The value below is the one for this seed.

# %%
rng = np.random.default_rng(SEED_NOISE)
noise = rng.normal(0.0, SIGMA, size=(T, N))
noise_result = probability_of_backtest_overfitting(noise, n_partitions=S)
print(f"pure noise: PBO = {noise_result.pbo:.4f} over {noise_result.n_combinations} splits")

# %% [markdown]
# ## 2. One real edge
#
# Column 0 gets a positive mean and the other N - 1 columns stay pure noise. In
# most splits the real strategy is the in-sample winner, and out of sample it
# also ranks near the top. Its logits are therefore positive and the PBO is near
# 0. This is one seed, not a bound: other seeds can give a noticeably larger
# value. Setting `EDGE = 0` makes the matrix pure noise again, and the PBO is then
# a draw whose expectation is 1/2.

# %%
rng = np.random.default_rng(SEED_EDGE)
edge = rng.normal(0.0, SIGMA, size=(T, N))
edge[:, 0] += EDGE  # column 0 is the real strategy
edge_result = probability_of_backtest_overfitting(edge, n_partitions=S)
print(f"one real edge: PBO = {edge_result.pbo:.4f} over {edge_result.n_combinations} splits")

# %% [markdown]
# ## 3. Logit distribution
#
# Each split gives a logit lambda = log(w / (1 - w)), where w is the relative
# out-of-sample rank of the in-sample winner. PBO is the share of logits at or
# below zero. The figure shows the logits of the pure-noise matrix, with the
# zero line and the PBO value. It uses `finlab.plot.pbo_distribution`, which
# needs plotly. Without plotly the cell prints the install hint and the rest of
# the notebook still runs.

# %%
try:
    from finlab.plot import pbo_distribution

    fig = pbo_distribution(noise_result, title="Pure noise: logit distribution")
except ImportError as exc:
    fig = None
    print(exc)
    print("Install the optional plotting extra with: pip install 'finlab[plot]'")
fig

# %% [markdown]
# ## 4. Key numbers
#
# `KEY` collects the values the test in `tests/test_example_pbo.py` checks.

# %%
KEY = {
    "noise_pbo": noise_result.pbo,
    "edge_pbo": edge_result.pbo,
    "n_combinations": noise_result.n_combinations,
}
print(KEY)
