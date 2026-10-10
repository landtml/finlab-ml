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
# # Standard versus sequential bootstrap: average uniqueness
#
# Monte Carlo comparison from chapter 4 (Snippets 4.7 and 4.8), run with
# `finlab.monte_carlo.bootstrap_uniqueness_mc`.
#
# Each trial does three steps:
#
# 1. **Labels.** Draw `n_obs` random labels, each with a start bar and a length (Snippet 4.7).
# 2. **Standard bootstrap.** Draw a sample of the same size from those labels, uniformly and
#    with replacement.
# 3. **Sequential bootstrap.** Draw the sample one label at a time, where each candidate's
#    chance depends on how little it overlaps the labels already drawn (Snippet 4.8).
#
# The average uniqueness of a sample is computed in two stages. For each draw, take the mean over
# its bars of `1 / (number of draws covering that bar)`. Then take the mean of those values over
# the draws. A larger value means less overlap. Both samples in a trial come from the same label
# set, so the comparison is paired: the quantity of interest is `seq_u - std_u` within each trial.
#
# This notebook uses `n_obs=10`, `n_bars=100`, `max_h=5` (the settings of Snippet 4.9), 2,000
# trials and seed 0. The module default is 10,000 trials. The comparison with the book's reported
# values is in `docs/proofs/monte_carlo.md`, not here.

# %%
import numpy as np

from finlab.monte_carlo import bootstrap_uniqueness_mc

N_OBS, N_BARS, MAX_H = 10, 100, 5
N_ITER, SEED = 2_000, 0

frame = bootstrap_uniqueness_mc(n_obs=N_OBS, n_bars=N_BARS, max_h=MAX_H, n_iter=N_ITER, seed=SEED)
print(f"{len(frame):,} trials, columns: {list(frame.columns)}")

# %% [markdown]
# ## Mean and median
#
# Each average uniqueness is built from integer span lengths and integer concurrency counts, so
# its possible values are a discrete set and many trials share the same value. A median can
# therefore land on one of those common values. The last lines below count how many trials are
# within `TIE_TOL` of each median.

# %%
# Two averages of the same terms can differ by one unit in the last place, depending on the order
# of the floating-point sums. Values closer than TIE_TOL are treated as equal. For these settings
# the smallest genuine difference between two sample averages is far larger than 1e-12.
TIE_TOL = 1e-12

print(f"{'':>8} {'mean':>8} {'median':>8}")
for col in ["std_u", "seq_u"]:
    print(f"{col:>8} {frame[col].mean():>8.4f} {frame[col].median():>8.4f}")
for col in ["std_u", "seq_u"]:
    med = frame[col].median()
    n_at = int((np.abs(frame[col] - med) <= TIE_TOL).sum())
    print(f"{col}: {n_at} of {len(frame)} trials are within TIE_TOL of the median {med:.4f}")

# %% [markdown]
# ## Paired difference
#
# `d` is `seq_u - std_u` for each trial. The standard error is the sample standard deviation of `d`
# (ddof=1) divided by the square root of the number of trials. A trial counts as "higher" only if
# `seq_u` exceeds `std_u` by more than `TIE_TOL`. Trials within `TIE_TOL` are reported as ties.

# %%
d = (frame["seq_u"] - frame["std_u"]).to_numpy()
mean_gap = float(d.mean())
se_gap = float(d.std(ddof=1) / np.sqrt(len(d)))
frac_seq_higher = float((d > TIE_TOL).mean())
print(f"mean gap (seq_u - std_u):    {mean_gap:.6f}")
print(f"standard error:              {se_gap:.6f}")
print(f"mean gap / standard error:   {mean_gap / se_gap:.1f}")
print(f"fraction with seq_u > std_u: {frac_seq_higher:.4f}")
print(f"trials within TIE_TOL (ties): {int((np.abs(d) <= TIE_TOL).sum())}")

# %% [markdown]
# ## Histogram
#
# With plotly installed, the next cell draws both columns as overlaid histograms
# (`finlab.plot.monte_carlo_histogram`). Without plotly it prints the install hint and a text
# summary instead, so the notebook runs either way.

# %%
try:
    from finlab.plot import monte_carlo_histogram

    fig = monte_carlo_histogram(
        frame, title="Average uniqueness: standard and sequential bootstrap samples"
    )
except ImportError as exc:
    # plotly is missing. The message is the install hint from finlab.plot.
    print(exc)
    fig = None
    both = frame[["std_u", "seq_u"]].to_numpy()
    edges = np.linspace(float(both.min()), float(both.max()), 11)
    print("Text summary: counts per bin, with bin edges shared by both columns.")
    print("edges:", " ".join(f"{e:.3f}" for e in edges))
    for col in ["std_u", "seq_u"]:
        counts, _ = np.histogram(frame[col], bins=edges)
        print(f"{col:>6}:", " ".join(f"{c:4d}" for c in counts))

fig

# %% [markdown]
# ## Key numbers
#
# `KEY` holds the values that `tests/test_example_monte_carlo.py` checks.

# %%
KEY = {
    "mean_gap": mean_gap,
    "se_gap": se_gap,
    "frac_seq_higher": frac_seq_higher,
    "median_std": float(frame["std_u"].median()),
    "median_seq": float(frame["seq_u"].median()),
}
for name, value in KEY.items():
    print(f"{name:>16}: {value:.12f}")
