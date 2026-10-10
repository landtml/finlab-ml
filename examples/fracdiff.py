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
# # Fractional differentiation (AFML chapter 5)
#
# This notebook applies the fixed-width-window (FFD) fractional difference from
# `finlab.fracdiff` to a seeded random walk, and checks two closed-form facts:
#
# 1. **Weights.** For fractional order `d`, the lag weights satisfy `w_0 = 1` and
#    `w_k = -w_{k-1} (d - k + 1) / k`. Section 1 computes the first five weights
#    by this recursion and compares them with `get_weights`. The reference is the
#    closed form (recursion), not a book value.
# 2. **d = 1.** The weights are then `[1, -1, 0, ...]`, so the FFD output is the
#    first difference of the series. Section 3 checks this.
#
# Run it after installing finlab (`pip install -e .`), or with `PYTHONPATH=src`
# from the repository root. The run is seeded and takes a few seconds.

# %%
import time

import numpy as np

from finlab.fracdiff import frac_diff_ffd, get_weights, get_weights_ffd

SEED = 20260101  # seed for the random walk
N_OBS = 3000  # length of the random walk
N_WEIGHTS = 5  # number of weights compared in section 1
TOL_W = 1e-12  # tolerance for the weight comparison
THRES = 1e-5  # FFD weight cut-off: a fixed tolerance, not data-dependent (the module default)

# %% [markdown]
# ## 1. Weights: closed form (recursion), not a book value
#
# The recursion is written out in plain Python. It does not share code with the
# module's numba kernel, so the comparison is a real cross-check.


# %%
def recursion_weights(d: float, size: int) -> np.ndarray:
    """First ``size`` lag-ordered weights of ``(1 - B)^d`` by the recursion.

    ``w_0 = 1`` and ``w_k = -w_{k-1} (d - k + 1) / k``.
    """
    w = np.empty(size)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


abs_errors = []
for d in (0.4, 1.0):
    ref = recursion_weights(d, N_WEIGHTS)
    mod = get_weights(d, N_WEIGHTS)
    abs_errors.append(float(np.max(np.abs(mod - ref))))
    print(f"d = {d}")
    print(f"  closed form (recursion), not a book value: {ref}")
    print(f"  get_weights:                              {mod}")

max_abs_w_err = max(abs_errors)
weights_match_recursion = bool(max_abs_w_err < TOL_W)
print(f"max |get_weights - recursion| = {max_abs_w_err:.3e}")

# %% [markdown]
# ## 2. Fixed-width window on a seeded random walk
#
# The input is a log-price-like random walk. The FFD window is fixed, so every
# output uses the same weights. The first `width` rows have no complete window
# and are NaN, so the number of finite outputs is `N_OBS - width`.

# %%
rng = np.random.default_rng(SEED)
log_price = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, N_OBS))

t0 = time.perf_counter()
fd_04 = np.asarray(frac_diff_ffd(log_price, 0.4, thres=THRES))
fd_10 = np.asarray(frac_diff_ffd(log_price, 1.0, thres=THRES))
print(f"frac_diff_ffd on {N_OBS} points: {time.perf_counter() - t0:.2f} s (includes numba compile)")

width_04 = len(get_weights_ffd(0.4, THRES)) - 1
n_out_d04 = int(np.isfinite(fd_04).sum())
assert n_out_d04 == N_OBS - width_04
print(f"d = 0.4: window width {width_04}, finite outputs {n_out_d04} of {N_OBS}")

# %% [markdown]
# ## 3. d = 1 reproduces the first difference
#
# With `d = 1` the recursion gives `w_2 = 0`, so the FFD window has width 1 and
# the weights are `[1, -1]`. The output is then `x_t - x_{t-1}` on every valid row.

# %%
diff_ref = np.diff(log_price)
d1_matches_first_difference = bool(
    np.isnan(fd_10[0]) and np.allclose(fd_10[1:], diff_ref, rtol=0.0, atol=1e-14)
)
print("d = 1 FFD weights:", get_weights_ffd(1.0, THRES))
print("first row is NaN:", bool(np.isnan(fd_10[0])))
print("max |output - first difference|:", float(np.max(np.abs(fd_10[1:] - diff_ref))))

# %% [markdown]
# ## Key values
#
# The last cell collects the values that the test in `tests/test_example_fracdiff.py`
# reads.

# %%
KEY = {
    "weights_match_recursion": weights_match_recursion,
    "d1_matches_first_difference": d1_matches_first_difference,
    "max_abs_w_err": max_abs_w_err,
    "n_out_d04": n_out_d04,
}
print(KEY)
