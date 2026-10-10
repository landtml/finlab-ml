"""Numbers quoted in docs/proofs/*.md, pinned to fixed seeds.

Each test reproduces one numeric claim from a proof note with a fixed seed and a
stated tolerance. Measured timings are not tested here; they are labelled as
measurements and point to benchmarks/. Keep this file fast (under 10 s).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

from finlab.bet_sizing import calibrate_sigmoid_width
from finlab.cv import PurgedKFold
from finlab.entropy import kontoyiannis_entropy
from finlab.importance import mda, sfi
from finlab.microstructure import becker_parkinson_volatility
from finlab.monte_carlo import bootstrap_uniqueness_mc, random_t1
from finlab.pbo import probability_of_backtest_overfitting
from finlab.stats import expected_max_sharpe, min_track_record_length
from finlab.structural_breaks import chu_stinchcombe_white, csw_critical_value, cusum_test, sadf


# ---------------------------------------------------------------------------
# pbo.md
# ---------------------------------------------------------------------------


def test_pbo_split_count_s16() -> None:
    assert math.comb(16, 8) == 12870
    assert math.comb(8, 4) == 70


def test_pbo_pure_noise_simulation_seeds_0_to_59() -> None:
    def pbos(n_cols: int) -> np.ndarray:
        out = []
        for seed in range(60):
            M = np.random.default_rng(seed).normal(0.0, 0.01, size=(320, n_cols))
            out.append(probability_of_backtest_overfitting(M, n_partitions=8).pbo)
        return np.array(out)

    p60 = pbos(60)
    assert p60.mean() == pytest.approx(0.5028571429, abs=1e-6)
    assert p60.std() == pytest.approx(0.1565964526, abs=1e-6)  # population sd (ddof=0)
    assert np.percentile(p60, 5) == pytest.approx(0.255, abs=1e-9)
    assert np.percentile(p60, 95) == pytest.approx(0.745, abs=1e-9)

    p10 = pbos(10)
    assert p10.std() == pytest.approx(0.2354861811, abs=1e-6)

    p2 = pbos(2)
    assert p2.mean() == pytest.approx(0.4685714286, abs=1e-6)
    assert p2.std() == pytest.approx(0.2978471507, abs=1e-6)


# ---------------------------------------------------------------------------
# stats.md
# ---------------------------------------------------------------------------


def test_stats_min_track_record_length_exact() -> None:
    hand = 1.0 + 1.125 * (norm.ppf(0.95) / 0.5) ** 2
    assert hand == pytest.approx(13.1749455434, abs=1e-9)
    assert min_track_record_length(0.5, 0.0, 0.0, 3.0, 0.95) == pytest.approx(hand, abs=1e-9)


def test_stats_expected_max_monte_carlo_seed0() -> None:
    # One generator, one standard_normal((20000, N)) call per N, in the order 2, 10, 100, 1000.
    rng = np.random.default_rng(0)
    mc_quoted = {2: 0.570, 10: 1.542, 100: 2.508, 1000: 3.244}
    formula_quoted = {2: 0.520, 10: 1.575, 100: 2.531, 1000: 3.255}
    for n_trials in (2, 10, 100, 1000):
        mc = float(rng.standard_normal((20000, n_trials)).max(axis=1).mean())
        assert abs(mc - mc_quoted[n_trials]) < 5e-4
        formula = expected_max_sharpe(n_trials, 1.0)
        assert abs(formula - formula_quoted[n_trials]) < 5e-4
        assert abs(mc - formula) <= 0.0505  # largest error is 0.0504 at N = 2


# ---------------------------------------------------------------------------
# entropy.md
# ---------------------------------------------------------------------------


def test_entropy_gaussian_constant_in_bits() -> None:
    assert 0.5 * math.log2(2 * math.pi * math.e) == pytest.approx(2.0470955852, abs=1e-9)
    assert 0.5 * math.log(2 * math.pi * math.e) == pytest.approx(1.4189385332, abs=1e-9)


def test_entropy_book_example_values() -> None:
    # Snippet 18.4 values; the book's printed 0.96 is a book claim, not checked here.
    assert kontoyiannis_entropy("11100001") == pytest.approx(0.9682408185, abs=1e-6)
    assert kontoyiannis_entropy("01100001") == pytest.approx(0.8432408185, abs=1e-6)


def test_entropy_gao_window_arithmetic() -> None:
    assert abs(198 + math.log2(198) ** 2 - 256) < 0.5
    assert round(math.log2(198) ** 2) == 58


# ---------------------------------------------------------------------------
# hrp.md
# ---------------------------------------------------------------------------


def test_hrp_square_matrix_linkage_heights() -> None:
    rho = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, -0.2], [0.2, -0.2, 1.0]])
    D = np.sqrt(0.5 * (1.0 - rho))
    np.fill_diagonal(D, 0.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        wrong = linkage(D, method="single")  # square matrix read as observations
    assert any(w.category.__name__ == "ClusterWarning" for w in record)
    assert wrong[:, 2] == pytest.approx([0.566, 0.975], abs=5e-4)
    right = linkage(squareform(D, checks=False), method="single")
    assert right[:, 2] == pytest.approx([0.3873, 0.6325], abs=5e-4)


# ---------------------------------------------------------------------------
# importance.md (data and model copied from tests/test_importance.py)
# ---------------------------------------------------------------------------


class _NearestCentroid:
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack(
            [np.average(X[y == c], axis=0, weights=w[y == c]) for c in self.classes_]
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=-1)
        return self.classes_[np.argmin(d, axis=1)]


def _make_data(n: int = 600, p: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, p))
    X[:, 0] += 2.0 * (2 * y - 1)
    return X, y


def test_importance_mda_seeds_0_to_5() -> None:
    f0, noise_max = [], 0.0
    for seed in range(6):
        X, y = _make_data(seed=seed)
        _, imp = mda(_NearestCentroid, X, y, PurgedKFold(n_splits=5), n_repeats=3, seed=0)
        f0.append(float(imp.loc[0, "mean"]))
        noise_max = max(noise_max, float(np.abs(imp.loc[1:, "mean"]).max()))
    assert all(0.44 < v < 0.50 for v in f0)
    assert 0.47 <= float(np.mean(f0)) <= 0.49
    assert noise_max < 0.003


def test_importance_sfi_seeds_0_to_5() -> None:
    f0, noise = [], []
    for seed in range(6):
        X, y = _make_data(seed=seed)
        _, imp = sfi(_NearestCentroid, X, y, PurgedKFold(n_splits=5))
        f0.append(float(imp.loc[0, "mean"]))
        noise.extend(imp.loc[1:, "mean"].tolist())
    assert all(0.97 < v < 1.0 for v in f0)
    assert all(0.46 < v < 0.55 for v in noise)


# ---------------------------------------------------------------------------
# microstructure.md (Becker-Parkinson on GBM; the formula is rebuilt from its definition)
# ---------------------------------------------------------------------------


def test_becker_parkinson_seed0_values_and_alternative_coefficient() -> None:
    rng = np.random.default_rng(0)
    sigma, nb, steps = 0.02, 20_000, 400
    inc = rng.normal(0.0, sigma / math.sqrt(steps), size=nb * steps)
    path = np.concatenate([[0.0], np.cumsum(inc)])
    bars = np.stack([path[k * steps : (k + 1) * steps + 1] for k in range(nb)])
    hi = np.exp(bars.max(axis=1))
    lo = np.exp(bars.min(axis=1))

    snippet = np.nanmean(becker_parkinson_volatility(hi, lo, sl=1))
    assert snippet == pytest.approx(0.0189323772, abs=1e-8)
    assert snippet / sigma == pytest.approx(0.9466, abs=1e-4)

    # Rebuild the two terms from their definitions (sl = 1): h = ln(H/L),
    # beta_t = h_{t-1}^2 + h_t^2, gamma_t = ln(max H / min L)^2 over bars t-1, t.
    h = np.log(hi / lo)
    beta = h[:-1] ** 2 + h[1:] ** 2
    gamma = np.log(np.maximum(hi[:-1], hi[1:]) / np.minimum(lo[:-1], lo[1:])) ** 2
    den = 3.0 - 2.0 * math.sqrt(2.0)
    k2 = math.sqrt(8.0 / math.pi)

    def estimate(coef: float) -> float:
        s = coef * np.sqrt(beta) / (k2 * den) + np.sqrt(gamma / (k2 * k2 * den))
        return float(np.nanmean(np.clip(s, 0.0, None)))

    assert estimate(2.0**-0.5 - 1.0) == pytest.approx(snippet, rel=1e-9)
    alternative = estimate(2.0**0.5 - 1.0)
    assert alternative == pytest.approx(0.13333, abs=1e-4)
    assert alternative / snippet == pytest.approx(7.04, abs=0.01)


# ---------------------------------------------------------------------------
# structural_breaks.md
# ---------------------------------------------------------------------------


def test_structural_breaks_cusum_shift_values_seed42() -> None:
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)
    y[100:] += 3.0
    s = cusum_test(y).statistic
    assert float(np.nanmax(np.abs(s[1:100]))) == pytest.approx(3.7835, abs=1e-3)
    assert abs(float(s[-1])) == pytest.approx(117.6372, abs=1e-3)


def test_structural_breaks_csw_late_drift_values_seed7() -> None:
    rng = np.random.default_rng(7)
    y = rng.standard_normal(300).cumsum() * 0.01
    drift = np.zeros(300)
    drift[150:] = np.linspace(0.0, 1.0, 150)
    res = chu_stinchcombe_white(y + drift, reference=140)
    assert float(res.statistic[-1]) == pytest.approx(6.4823, abs=1e-3)
    assert bool(res.reject[-1])
    assert csw_critical_value(299 - 140, 4.6) == pytest.approx(3.1095, abs=1e-3)
    no_drift = chu_stinchcombe_white(y, reference=140)
    assert float(no_drift.statistic[-1]) == pytest.approx(-1.4593, abs=1e-3)
    assert not bool(no_drift.reject[-1])


def test_structural_breaks_sadf_values_seed99() -> None:
    rng = np.random.default_rng(99)
    n = 200
    walk = rng.standard_normal(n).cumsum()
    explosive = np.exp(0.02 * np.arange(n)) * (1.0 + 0.01 * rng.standard_normal(n))
    assert float(np.nanmax(sadf(walk, min_length=30, lags=1))) == pytest.approx(1.3745, abs=1e-3)
    assert float(sadf(explosive, min_length=30, lags=1)[-1]) == pytest.approx(14.5507, abs=1e-3)


# ---------------------------------------------------------------------------
# monte_carlo.md
# ---------------------------------------------------------------------------

# Expected values of the seeded statistic from the Section 4 table (seeds 0, 1, 2, 3, 42).
_GAP_TABLE = {
    0: (0.084357, 0.002807, 0.6069, 0.6912),
    1: (0.087352, 0.002860, 0.6094, 0.6968),
    2: (0.082604, 0.002995, 0.6119, 0.6945),
    3: (0.083943, 0.002911, 0.6119, 0.6958),
    42: (0.085824, 0.002858, 0.6073, 0.6932),
}


@pytest.mark.parametrize("seed", sorted(_GAP_TABLE))
def test_monte_carlo_gap_table(seed: int) -> None:
    mean_d, se_d, mean_std, mean_seq = _GAP_TABLE[seed]
    df = bootstrap_uniqueness_mc(n_iter=2000, seed=seed)
    d = (df["seq_u"] - df["std_u"]).to_numpy()
    assert d.mean() == pytest.approx(mean_d, abs=1e-6)
    assert d.std(ddof=1) / math.sqrt(len(d)) == pytest.approx(se_d, abs=1e-6)
    assert df["std_u"].mean() == pytest.approx(mean_std, abs=5e-5)
    assert df["seq_u"].mean() == pytest.approx(mean_seq, abs=5e-5)


def test_monte_carlo_medians_seeds_0_and_7() -> None:
    df0 = bootstrap_uniqueness_mc(n_iter=20_000, seed=0, num_threads=4)
    assert df0["std_u"].median() == pytest.approx(0.6, abs=1e-4)
    assert df0["seq_u"].median() == pytest.approx(0.7, abs=1e-4)
    df7 = bootstrap_uniqueness_mc(n_iter=20_000, seed=7, num_threads=2)
    assert df7["std_u"].median() == pytest.approx(0.6, abs=1e-4)
    assert df7["seq_u"].median() == pytest.approx(0.7, abs=1e-4)


def test_monte_carlo_repeated_start_fraction() -> None:
    short = sum(
        1 for s in range(2000) if len(random_t1(10, 100, 5, seed=np.random.default_rng(s))) < 10
    )
    assert short / 2000 == pytest.approx(0.369, abs=1e-9)
    assert 1 - float(np.prod([1 - i / 100 for i in range(10)])) == pytest.approx(
        0.3718435, abs=1e-6
    )


# ---------------------------------------------------------------------------
# bet_sizing.md
# ---------------------------------------------------------------------------


def test_bet_sizing_calibrated_width() -> None:
    assert calibrate_sigmoid_width(10.0, 0.95) == pytest.approx(10.8033, abs=1e-4)
