# Proofs and derivations: `finlab.labeling`

Source: AFML Chapter 3 (Labeling), Snippets 3.1 to 3.8, and trend scanning
(not in AFML Chapter 3; see Section 6).

Every claim is tagged with one of three labels:

* **Proved here**: a proof is given in this document.
* **Checked by test**: a property is verified by `tests/test_labeling.py` against a
  brute-force reference. It is not proved here.
* **Claimed from the book only**: stated in AFML and used as given. Not proved
  or tested here.

## Notation

Bars are indexed by time `t` with prices `p_t > 0`. An event starts at bar `t0`
with side `s in {-1, +1}` and unit barrier width `trgt > 0` (a return).
`pt, sl >= 0` are multipliers, and `h` is an optional vertical barrier time
`tv`. The path of the event is the set of bars `t` with `t0 < t <= e`, where
`e` is the last bar at or before the vertical barrier (or the last bar if none).

## 1. Daily volatility (Snippet 3.1)

**Definition (book, Snippet 3.1).** Let `r_t = p_t / p_{u(t)} - 1`, where
`u(t)` is the last bar strictly before `t - 1 day`. The volatility is
`sigma_t = EWMstd(r; span)`, computed by `pandas.Series.ewm(span).std()` over the
bars where `u(t)` exists.

* **Proved here.** `searchsorted(t - 1 day, side="left") - 1` returns the last
  position `j` with `index_j < t - 1 day`. Positions with no such `j` (the first
  day) are dropped before the EWM and reappear as NaN after `reindex`.
* **Checked by test.** `test_daily_vol_matches_per_bar_reference` compares the
  vectorised lookup with a per-bar loop on a random walk.
* **Claimed from the book only.** The choice of EWMA span and of the
  one-day return lookback as the default volatility estimator.

## 2. Triple-barrier first touch (Snippets 3.2, 3.3, 3.6)

Define the side-adjusted return path `R_j = s (p_j / p_{t0} - 1)`, the
horizontal times

    tau_pt = min{ j in (t0, e] : R_j > pt * trgt }        (pt > 0)
    tau_sl = min{ j in (t0, e] : R_j < -sl * trgt }       (sl > 0)

with `min(empty) = +inf`, and the vertical time `tau_v = tv`.

The label end time is `t1 = min(tau_pt, tau_sl, tau_v)`, with `t1 = NaT` when
all three are infinite.

**Lemma 2.1 (first touch of a union of barriers).** For a fixed path, the first
bar at which any barrier set is hit equals the minimum of the first bars at which
each set is hit.

*Proof.* Let `A` and `B` be the sets of bars where a barrier of each kind is
touched. The first element of `A union B` is `min(A union B) = min(min A, min B)`
because `min` over a union is the min of the mins. Apply this to the three
barriers. *(Proved here.)*

**Proposition 2.2 (scan correctness).** The numba kernel `_scan_first_touch` returns
for each event the bar `min(tau_pt, tau_sl)` (or none), and reports
`sl` when `tau_sl <= tau_pt`.

*Proof.* The kernel loops `j` from `t0 + 1` to `e` in increasing order. At each `j` it
tests the stop-loss condition first, then the profit-taking condition, and breaks
at the first `j` where either holds. So it returns the smallest `j` in the union
of the two sets, and on a tie it reports the stop loss. This is `min(tau_pt, tau_sl)`
with the tie rule. The condition `sl > 0` (and `pt > 0`) guards the disabled
barriers, so a zero multiplier never produces a touch, even when `lo = -0 = 0`.
*(Proved here.)*

Note that the tie rule (stop loss first on a shared bar) is an implementation
convention, not stated in the book. The vertical barrier takes precedence only
when no horizontal barrier is touched by its bar `e`. The `e` bound caps the scan
at the vertical bar, so a horizontal touch on bar `e` itself wins over the
vertical barrier. *(Proved here.)*

**Proposition 2.3 (monotone path).** Suppose `p` is strictly increasing with
`p_{j+1} / p_j - 1 > pt * trgt` for all `j`, and `side = +1`, `sl > 0`, `pt > 0`,
`e >= t0 + 1`. Then `t1 = t0 + 1` and the barrier is `pt`.

*Proof.* `R_{t0+1} = p_{t0+1}/p_{t0} - 1 > pt * trgt`, so `tau_pt = t0 + 1`. No
stop-loss touch is possible because `R_j > 0 > -sl * trgt` for every `j`. Then
`t1 = min(t0 + 1, ...) = t0 + 1`. *(Proved here.)* The side `-1` on the same
rising path swaps the roles: `R_j < 0`, so the stop loss is touched at `t0 + 1`.

* **Checked by test.** `test_triple_barrier_hits_upper_on_rising_path`,
  `test_triple_barrier_hits_lower_on_falling_path`,
  `test_triple_barrier_meta_side_flips_which_barrier_is_hit`.
* **Checked by test.** `test_triple_barrier_matches_pandas_reference` and the
  meta-mode version compare `t1` and the barrier code with a pandas
  path-slice reference (Snippet 3.2 logic) on random walks with vertical
  barriers, for 3 seeds plus one meta-mode case.

**Vertical barrier (Snippet 3.4).** The book's `t1 = searchsorted(t + days)`
gives the first bar at or after `t + days`. `add_vertical_barrier` returns the same bar,
and NaT when no bar exists. *(Proved here, by the definition of
`searchsorted(side="left")`.)* Checked by test:
`test_vertical_barrier_is_first_bar_at_or_after_expiry`.

## 3. Side and size labels (Snippet 3.5)

For an event with `t1 != NaT`, let `P0 = p_{t0}` and `P1 = p_{t1}`, each with
the backward fill of `reindex`. Then

    ret = P1 / P0 - 1,        bin = sign(ret) in {-1, 0, 1}.

**Proposition 3.1.** `bin = 0` if and only if `ret = 0`, and `bin` is `+1` or
`-1` when the price moves up or down. *(Proved here, by `sign`.)*

## 4. Meta-labeling (Section 3.6, Snippet 3.7)

Given a primary side `s in {-1, +1}`, the meta-return is `ret_s = s * ret` and the
meta-label is

    bin_meta = 1  if  s * ret > 0,      0  otherwise.

**Proposition 4.1.** `bin_meta = 1` if and only if the primary model's side
produced a profit (`s` agrees with the sign of the realised return).

*Proof.* `s * ret > 0` holds iff `ret` has the same sign as `s` and is nonzero.
That is the event "the primary side was right". *(Proved here.)*

* **Checked by test.** `test_meta_labels_is_one_exactly_when_primary_side_was_right`
  and `test_get_bins_and_meta_labels_agree_with_definition` compare the `ret` and
  `bin` columns with the pnl recomputed from raw returns.
* **Claimed from the book only.** That meta-labels increase F1 or improve sizing,
  and the "quantamental" use cases of Section 3.8. These are not tested here.

Design note: `get_events` always returns a `side` column (`+1` when `side` was
not supplied), so `get_bins` takes an explicit `meta=True` flag instead of
inferring meta mode from that column. Without the flag the function would
mislabel side-and-size events as 0/1. This is an API decision, not a book result.

## 5. Dropping rare labels (Snippet 3.8)

Algorithm: repeat { compute frequencies `f_c` of the classes; if the number of
classes is below 3 or `min_c f_c >= min_pct`, stop; else delete all rows of the
class `argmin_c f_c` }.

**Proposition 5.1 (termination).** The loop terminates.

*Proof.* Each iteration with `K >= 3` classes removes one class, so the number of
classes strictly decreases. There are finitely many classes, so the loop
stops after at most `K - 2` removals. *(Proved here.)*

**Proposition 5.2 (output guarantee).** On exit, either there are at most two classes,
or every remaining class has frequency at least `min_pct`.

*Proof.* The loop only exits through the break test, which requires either
fewer than three classes or `min f_c >= min_pct`. *(Proved here.)* The first
case is the book's exception ("unless only two classes are left"): with two
classes the rare one can be below `min_pct`. The test checks both cases.

* **Checked by test.** `test_drop_labels_leaves_each_class_above_min_pct`
  (5 seeds, three classes remain, all `>= min_pct`), and
  `test_drop_labels_stops_at_two_classes_as_in_book`.

## 6. Trend scanning (not AFML Chapter 3)

Trend scanning is attributed to Lopez de Prado (2019) and Hudson & Thames. The
exact citation was not checked against a primary source while writing this
module. The implementation follows the definition in the function docstring.

For bar `i` and window length `L` with `i + L <= n`, let `y_k = log p_{i+k}` and
`x_k = k` for `k = 0..L-1`. Let `xbar = (L-1)/2`, `ybar` the sample mean, and

    Sxx = sum_k (x_k - xbar)^2,     Sxy = sum_k (x_k - xbar)(y_k - ybar).

**Lemma 6.1.** `Sxx = L (L^2 - 1) / 12`.

*Proof.* `sum_k k^2 = (L-1)L(2L-1)/6` and `sum_k k = L(L-1)/2`. Then
`Sxx = sum_k k^2 - L xbar^2 = (L-1)L(2L-1)/6 - L(L-1)^2/4 = L(L-1)(L+1)/12`,
using `(2L-1)/6 - (L-1)/4 = (L+1)/12`. *(Proved here.)*

**OLS quantities.** `b = Sxy / Sxx`, `a = ybar - b xbar`,
`SSE = sum_k (y_k - a - b x_k)^2`, `s^2 = SSE / (L-2)`, and
`SE(b) = sqrt(s^2 / Sxx)`. The statistic is `t = b / SE(b)`. The formulas for
`s^2` and `SE(b)` are the standard OLS results (unbiased variance estimate with
`L-2` degrees of freedom). *(Claimed from standard regression theory. They are
not re-derived here.)*

**Definition.** The chosen window is `L* = argmax_L |t_L|` over valid `L`, with
the first one on a tie. The label is `sign(t_{L*})` when `|t_{L*}| > t_threshold`
and `0` otherwise. A perfect fit (`SSE = 0`) gives `t = +-inf`, handled explicitly.

* **Checked by test.** `test_trend_scanning_matches_linregress_reference` compares
  the signed `t`, the chosen window, and the bin with `scipy.stats.linregress`
  (slope / stderr) on random walks, for 2 seeds, with a naive per-bar loop.
  `test_trend_scanning_tail_is_undefined_and_threshold_zeroes_weak_trends` and
  `test_trend_scanning_labels_perfect_uptrend_as_plus_one` check the tail, the
  threshold, and a perfect uptrend (finite but huge `t`, because floating-point
  residuals are not exactly zero).
* **Claimed from the book only.** Nothing. This method is not in AFML Chapter 3.
* **Design choice, not from the literature.** `t_threshold = 1.96` is a default
  chosen for this module.

Scope: trend-scanning labels look forward by construction (window `i..i+L-1`).
They must not be used as features.

## Not covered

The multiprocessing engine (Chapter 20), the exercises, and the sampling of
events (Chapter 2) that generates `t_events`.
