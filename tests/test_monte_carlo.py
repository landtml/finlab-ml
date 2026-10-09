"""Tests for finlab.monte_carlo.run_trials."""
import numpy as np
import pandas as pd
import pytest

from finlab.monte_carlo import run_trials


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
