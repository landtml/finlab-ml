"""Tests for finlab.monte_carlo: run_trials and the bootstrap uniqueness experiment.

The naive references (``naive_indicator``, ``naive_sequential_draws`` and
``naive_sample_avg_uniqueness``) are copied from ``tests/test_weights.py``. The
sequential reference consumes uniforms in the same order as the numba kernel.
"""
import numpy as np
import pandas as pd
import pytest

from finlab.monte_carlo import (
    _trial_uniqueness,
    bootstrap_uniqueness_mc,
    bootstrap_uniqueness_trial,
    random_t1,
    run_trials,
)
from finlab.weights import sample_average_uniqueness


def _normal_trial(rng):
    x = rng.standard_normal(3)
    return {"mean": float(x.mean()), "first": float(x[0]), "norm": float(np.linalg.norm(x))}


def _scaled_trial(rng, scale, offset):
    return {"y": scale * float(rng.standard_normal()) + offset}


def _ragged_trial(rng):
    # Returns an extra key on about half of the trials, which run_trials must reject.
    if rng.random() < 0.5:
        return {"a": 1.0, "b": 2.0}
    return {"a": 1.0}


def _direct(seed, i, func, **kwargs):
    return func(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(i,))), **kwargs)


def test_same_seed_gives_identical_frame():
    a = run_trials(_normal_trial, 30, seed=5)
    b = run_trials(_normal_trial, 30, seed=5)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("threads", [2, 3])
def test_num_threads_does_not_change_result(threads):
    serial = run_trials(_normal_trial, 50, seed=11, num_threads=1)
    parallel = run_trials(_normal_trial, 50, seed=11, num_threads=threads)
    pd.testing.assert_frame_equal(serial, parallel)


def test_prefix_property():
    long = run_trials(_normal_trial, 50, seed=3)
    short = run_trials(_normal_trial, 20, seed=3)
    pd.testing.assert_frame_equal(long.iloc[:20], short)


def test_trial_matches_direct_call():
    seed = 9
    out = run_trials(_normal_trial, 12, seed=seed)
    for i in (0, 4, 11):
        direct = _direct(seed, i, _normal_trial)
        for key, value in direct.items():
            assert out.loc[i, key] == value


def test_output_shape_index_and_columns():
    out = run_trials(_normal_trial, 7, seed=1)
    assert list(out.columns) == ["mean", "first", "norm"]
    pd.testing.assert_index_equal(out.index, pd.RangeIndex(7))
    assert out.attrs["seed"] == 1


def test_seed_none_records_integer_and_is_reproducible():
    a = run_trials(_normal_trial, 20, seed=None)
    assert type(a.attrs["seed"]) is int
    b = run_trials(_normal_trial, 20, seed=None)
    assert not a.equals(b)
    replay = run_trials(_normal_trial, 20, seed=a.attrs["seed"])
    pd.testing.assert_frame_equal(a, replay)


def test_different_seeds_give_different_results():
    a = run_trials(_normal_trial, 20, seed=1)
    b = run_trials(_normal_trial, 20, seed=2)
    assert not a.equals(b)


def test_kwargs_are_passed_through():
    out = run_trials(_scaled_trial, 15, seed=4, scale=2.0, offset=10.0)
    for i in (0, 7, 14):
        expected = _direct(4, i, _scaled_trial, scale=2.0, offset=10.0)["y"]
        assert out.loc[i, "y"] == expected


def test_kwargs_work_with_parallel_run():
    serial = run_trials(_scaled_trial, 25, seed=4, num_threads=1, scale=2.0, offset=1.0)
    parallel = run_trials(_scaled_trial, 25, seed=4, num_threads=3, scale=2.0, offset=1.0)
    pd.testing.assert_frame_equal(serial, parallel)


def test_inconsistent_keys_raise():
    with pytest.raises(ValueError, match="returned keys"):
        run_trials(_ragged_trial, 20, seed=0)


@pytest.mark.parametrize("n_iter", [0, -1, 2.5, True])
def test_invalid_n_iter_raises(n_iter):
    with pytest.raises(ValueError, match="n_iter"):
        run_trials(_normal_trial, n_iter, seed=0)


@pytest.mark.parametrize("num_threads", [0, -2])
def test_invalid_num_threads_raises(num_threads):
    with pytest.raises(ValueError, match="num_threads"):
        run_trials(_normal_trial, 5, seed=0, num_threads=num_threads)


@pytest.mark.parametrize("seed", [-1, 1.5])
def test_invalid_seed_raises(seed):
    with pytest.raises(ValueError, match="seed"):
        run_trials(_normal_trial, 5, seed=seed)


# ---------------------------------------------------------------------------
# Bootstrap uniqueness experiment
# ---------------------------------------------------------------------------


def naive_indicator(index, t1):
    m = np.zeros((len(index), len(t1)))
    for i, (t0, t_end) in enumerate(t1.items()):
        for k, b in enumerate(index):
            if t0 <= b <= t_end:
                m[k, i] = 1.0
    return m


def naive_sequential_draws(m, uniforms):
    """Snippet 4.5 with the same inverse-CDF rule as the kernel, as explicit loops."""
    n_bars, n_ev = m.shape
    colsum = (m != 0).sum(axis=0)
    c = np.zeros(n_bars)
    draws = []
    for u in uniforms:
        avg = np.zeros(n_ev)
        for j in range(n_ev):
            if colsum[j] > 0:
                s = 0.0
                for t in range(n_bars):
                    if m[t, j] != 0:
                        s += 1.0 / (1.0 + c[t])
                avg[j] = s / colsum[j]
        total = 0.0
        for j in range(n_ev):
            total += avg[j]
        target = u * total
        acc = 0.0
        pick = -1
        last = -1
        for j in range(n_ev):
            if avg[j] > 0.0:
                last = j
                acc += avg[j]
                if acc > target:
                    pick = j
                    break
        if pick < 0:
            pick = last
        draws.append(pick)
        for t in range(n_bars):
            if m[t, pick] != 0:
                c[t] += 1.0
    return np.array(draws, dtype=np.int64)


def naive_sample_avg_uniqueness(m, draws):
    """Average uniqueness of a bootstrap sample, counting repeats, by explicit loops."""
    c = (m[:, draws] != 0).sum(axis=1).astype(float)
    vals = []
    for d in draws:
        rows = np.nonzero(m[:, d])[0]
        vals.append(np.mean(1.0 / c[rows]))
    return float(np.mean(vals))


@pytest.mark.parametrize("seed", range(4))
def test_random_t1_index_is_unique_sorted_and_values_are_ordered(seed):
    n_bars = 40
    t1 = random_t1(15, n_bars, 6, seed=seed)
    idx = t1.index.to_numpy()
    vals = t1.to_numpy()
    assert len(t1) == 15
    assert t1.index.is_unique
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() <= n_bars - 1
    assert np.all(vals >= idx)
    assert vals.max() <= n_bars - 1


@pytest.mark.parametrize("n_obs,n_bars,max_h", [(15, 40, 6), (20, 20, 8), (5, 30, 1)])
def test_random_t1_span_lengths_respect_max_h_and_clip_at_last_bar(n_obs, n_bars, max_h):
    for seed in range(5):
        t1 = random_t1(n_obs, n_bars, max_h, seed=seed)
        idx = t1.index.to_numpy()
        vals = t1.to_numpy()
        diff = vals - idx
        assert np.all(diff <= max_h)
        # A zero length can only come from clipping at the last bar.
        assert np.all((diff >= 1) | (vals == n_bars - 1))
        assert np.all(idx[diff == 0] == n_bars - 1)


def test_random_t1_same_seed_gives_same_series():
    a = random_t1(12, 50, 4, seed=7)
    b = random_t1(12, 50, 4, seed=7)
    pd.testing.assert_series_equal(a, b)


def test_random_t1_generator_and_int_seed_agree():
    a = random_t1(12, 50, 4, seed=7)
    b = random_t1(12, 50, 4, seed=np.random.default_rng(7))
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize(
    "n_obs,n_bars,max_h",
    [(0, 10, 3), (11, 10, 3), (3, 10, 0), (3, 10, -1), (2.5, 10, 3), (3, True, 3)],
)
def test_random_t1_rejects_bad_arguments(n_obs, n_bars, max_h):
    with pytest.raises(ValueError):
        random_t1(n_obs, n_bars, max_h, seed=0)


@pytest.mark.parametrize("seed", range(6))
def test_trial_uniqueness_matches_naive_reference(seed):
    n_bars, n_obs, max_h = 40, 12, 6
    t1 = random_t1(n_obs, n_bars, max_h, seed=seed)
    starts = t1.index.to_numpy(dtype=np.int64)
    ends = t1.to_numpy(dtype=np.int64)
    m = naive_indicator(np.arange(n_bars), t1)

    rng = np.random.default_rng(100 + seed)
    std = rng.integers(0, n_obs, size=n_obs)
    uniforms = rng.random(n_obs)
    seq = naive_sequential_draws(m, uniforms)

    ref_std = naive_sample_avg_uniqueness(m, std)
    ref_seq = naive_sample_avg_uniqueness(m, seq)
    assert ref_std == pytest.approx(sample_average_uniqueness(m, std), rel=1e-12)

    got_std, got_seq = _trial_uniqueness(n_bars, starts, ends, std, uniforms)
    assert got_std == pytest.approx(ref_std, rel=1e-12)
    assert got_seq == pytest.approx(ref_seq, rel=1e-12)


def test_trial_uniqueness_hand_computed_repeat_case():
    # Labels cover bars {0, 1} and {2}. Sample [0, 0, 1] gives c = [2, 2, 1].
    # Label 0 has uniqueness 1/2 on each of its bars, twice; label 1 has 1.
    # The mean over the three draws is (1/2 + 1/2 + 1) / 3 = 2/3.
    starts = np.array([0, 2], dtype=np.int64)
    ends = np.array([1, 2], dtype=np.int64)
    std = np.array([0, 0, 1], dtype=np.int64)
    uniforms = np.array([0.5, 0.5, 0.5])
    std_u, _ = _trial_uniqueness(3, starts, ends, std, uniforms)
    assert std_u == pytest.approx(2.0 / 3.0, rel=1e-12)


@pytest.mark.parametrize("n", [1, 3, 7])
def test_identical_spans_give_one_over_n(n):
    starts = np.full(n, 2, dtype=np.int64)
    ends = np.full(n, 6, dtype=np.int64)
    rng = np.random.default_rng(n)
    std = rng.integers(0, n, size=n)
    uniforms = rng.random(n)
    std_u, seq_u = _trial_uniqueness(10, starts, ends, std, uniforms)
    assert std_u == pytest.approx(1.0 / n, rel=1e-12)
    assert seq_u == pytest.approx(1.0 / n, rel=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_trial_follows_documented_draw_order(seed):
    n_obs, n_bars, max_h = 9, 30, 4
    got = bootstrap_uniqueness_trial(np.random.default_rng(seed), n_obs, n_bars, max_h)

    rng = np.random.default_rng(seed)
    t1 = random_t1(n_obs, n_bars, max_h, rng)
    m = naive_indicator(np.arange(n_bars), t1)
    std = rng.integers(0, n_obs, size=n_obs)
    uniforms = rng.random(n_obs)
    seq = naive_sequential_draws(m, uniforms)
    assert got["std_u"] == pytest.approx(naive_sample_avg_uniqueness(m, std), rel=1e-12)
    assert got["seq_u"] == pytest.approx(naive_sample_avg_uniqueness(m, seq), rel=1e-12)


def test_trial_is_deterministic_for_fixed_rng_seed():
    a = bootstrap_uniqueness_trial(np.random.default_rng(5), 10, 50, 4)
    b = bootstrap_uniqueness_trial(np.random.default_rng(5), 10, 50, 4)
    assert a == b
    assert set(a) == {"std_u", "seq_u"}


@pytest.mark.parametrize("seed", range(3))
def test_single_label_sample_has_uniqueness_one(seed):
    out = bootstrap_uniqueness_trial(np.random.default_rng(seed), 1, 20, 4)
    assert out["std_u"] == 1.0
    assert out["seq_u"] == 1.0


def test_trial_values_lie_in_unit_interval():
    df = bootstrap_uniqueness_mc(n_obs=8, n_bars=30, max_h=5, n_iter=200, seed=2)
    assert (df > 0.0).all().all()
    assert (df <= 1.0).all().all()


def test_bootstrap_uniqueness_mc_shape_and_columns():
    df = bootstrap_uniqueness_mc(n_iter=25, seed=1)
    assert list(df.columns) == ["std_u", "seq_u"]
    assert len(df) == 25
    pd.testing.assert_index_equal(df.index, pd.RangeIndex(25))


def test_bootstrap_uniqueness_mc_same_seed_gives_same_frame():
    a = bootstrap_uniqueness_mc(n_iter=30, seed=4)
    b = bootstrap_uniqueness_mc(n_iter=30, seed=4)
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_uniqueness_mc_num_threads_does_not_change_result():
    serial = bootstrap_uniqueness_mc(n_iter=40, seed=3, num_threads=1)
    parallel = bootstrap_uniqueness_mc(n_iter=40, seed=3, num_threads=2)
    pd.testing.assert_frame_equal(serial, parallel)


# ---------------------------------------------------------------------------
# Statistical and slow checks
# ---------------------------------------------------------------------------


def test_sequential_bootstrap_uniqueness_exceeds_standard_by_four_se():
    # The margin was measured once when this test was written and is far above
    # the bound. The bound is a loose check that the paired gap is not noise,
    # not a pinned value.
    df = bootstrap_uniqueness_mc(n_iter=2000, seed=0)
    assert ((df > 0.0) & (df <= 1.0)).all().all()
    d = (df["seq_u"] - df["std_u"]).to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert d.mean() > 4 * se


@pytest.mark.slow
def test_large_run_sequential_median_exceeds_standard_median():
    df = bootstrap_uniqueness_mc(n_iter=20_000, seed=7, num_threads=2)
    assert df["seq_u"].median() > df["std_u"].median()
    means = df.mean()
    assert ((means > 0.0) & (means <= 1.0)).all()
