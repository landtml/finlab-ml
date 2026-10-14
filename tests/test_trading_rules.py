"""Tests for finlab.trading_rules (AFML ch. 13: OU helpers, fit_ou, otr_grid)."""

from __future__ import annotations

import math

import numba
import numpy as np
import pandas as pd
import pytest

from finlab.trading_rules import (
    OUFit,
    _exits_one_pair,
    _otr_exits_python,
    _pair_seeds,
    _standard_normals,
    fit_ou,
    otr_grid,
    ou_half_life,
    ou_phi,
)

BIG = 1e9  # a barrier that no simulated path reaches


# ---------------------------------------------------------------------------
# O-U helpers
# ---------------------------------------------------------------------------


def test_ou_phi_of_five_steps_half_life() -> None:
    assert ou_phi(5.0) == 2 ** (-0.2)


def test_ou_half_life_round_trip() -> None:
    for h in (0.5, 1.0, 5.0, 20.0, 1000.0):
        assert ou_half_life(ou_phi(h)) == pytest.approx(h, rel=1e-12)


def test_deviation_halves_after_half_life() -> None:
    # Checked by test: phi ** tau is 1/2 at tau = ou_half_life(phi).
    for phi in (0.1, 0.5, 0.9, 0.99):
        assert phi ** ou_half_life(phi) == pytest.approx(0.5, rel=1e-12)


def test_half_life_increases_with_phi() -> None:
    taus = [ou_half_life(p) for p in np.linspace(0.05, 0.95, 19)]
    assert all(b > a for a, b in zip(taus, taus[1:]))


@pytest.mark.parametrize("phi", [0.0, 1.0, 1.2, -0.5, np.nan, np.inf, True])
def test_ou_half_life_refuses_phi_outside_unit_interval(phi) -> None:
    with pytest.raises(ValueError, match="phi"):
        ou_half_life(phi)


@pytest.mark.parametrize("half_life", [0.0, -3.0, np.inf, np.nan, True])
def test_ou_phi_refuses_bad_half_life(half_life) -> None:
    with pytest.raises(ValueError, match="half_life"):
        ou_phi(half_life)


# ---------------------------------------------------------------------------
# fit_ou
# ---------------------------------------------------------------------------


def _simulate_ou(n: int, phi: float, sigma: float, forecast: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = forecast
    for t in range(1, n):
        x[t] = (1.0 - phi) * forecast + phi * x[t - 1] + sigma * eps[t]
    return x


def test_fit_ou_closed_form_on_three_prices() -> None:
    # Hand computed: lag = [1, 2], cur = [2, 0], so phi = (2 + 0) / (1 + 4) = 0.4.
    # Residuals cur - phi * lag = [1.6, -0.8], mean 0.4, ddof=1 variance 2.88.
    fit = fit_ou([1.0, 2.0, 0.0], 0.0)
    assert fit.phi == pytest.approx(0.4, rel=1e-14)
    assert fit.sigma == pytest.approx(math.sqrt(2.88), rel=1e-14)
    assert fit.half_life == pytest.approx(ou_half_life(0.4), rel=1e-14)
    assert fit.n_obs == 2


def test_fit_ou_recovers_parameters_on_seeded_path() -> None:
    # Checked by test. Standard errors at n = 20000: about 0.0031 for phi and 0.0075 for
    # sigma. The tolerances are about 6 standard errors, and the seed is fixed.
    x = _simulate_ou(20_000, phi=0.9, sigma=1.5, forecast=2.0, seed=20261013)
    fit = fit_ou(x, forecast=2.0)
    assert abs(fit.phi - 0.9) < 0.02
    assert abs(fit.sigma - 1.5) < 0.05
    assert fit.n_obs == 19_999
    assert fit.half_life == pytest.approx(ou_half_life(fit.phi), rel=1e-12)


def test_fit_ou_half_life_is_inf_when_phi_at_or_above_one() -> None:
    fit = fit_ou([1.0, 2.0, 3.0, 4.0], 0.0)  # phi = 20 / 14 > 1
    assert fit.phi > 1.0
    assert fit.half_life == math.inf


def test_fit_ou_half_life_is_nan_when_phi_not_positive() -> None:
    fit = fit_ou([1.0, -1.0, 1.0, -1.0], 0.0)  # phi = -1
    assert fit.phi == pytest.approx(-1.0)
    assert math.isnan(fit.half_life)
    assert isinstance(fit, OUFit)


def test_fit_ou_accepts_series_and_single_column_frame() -> None:
    expected = fit_ou([1.0, 2.0, 0.0], 0.0)
    assert fit_ou(pd.Series([1.0, 2.0, 0.0]), 0.0) == expected
    assert fit_ou(pd.DataFrame({"p": [1.0, 2.0, 0.0]}), 0.0) == expected


@pytest.mark.parametrize(
    "prices, forecast, match",
    [
        ([1.0, 2.0], 0.0, "at least 3"),
        ([1.0, np.nan, 2.0], 0.0, "finite"),
        ([1.0, None, 2.0], 0.0, "prices"),
        ([1.0, 2.0, 3.0], np.nan, "forecast"),
        ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 0.0, "one-dimensional"),
        ([0.0, 0.0, 0.0], 0.0, "undefined"),
    ],
)
def test_fit_ou_refusals(prices, forecast, match) -> None:
    with pytest.raises(ValueError, match=match):
        fit_ou(prices, forecast)


# ---------------------------------------------------------------------------
# Kernel against the pure-Python transcription
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "half_life, forecast, sigma, start, pt, sl, max_hp, n_iter, index",
    [
        (3.0, 0.5, 1.3, 0.2, 1.0, 2.0, 10, 7, 2),
        (5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5, 10, 0),  # zero barriers: every path closes at update 1
        (20.0, -1.0, 0.5, -0.3, 3.0, 0.7, 30, 25, 1),
        (1.0, 2.0, 2.0, 0.0, 0.5, 0.5, 0, 8, 0),  # max_hp = 0: one update per path
    ],
)
def test_kernel_matches_python_transcription(
    half_life, forecast, sigma, start, pt, sl, max_hp, n_iter, index
) -> None:
    # Exact equality: the kernel and the transcription run the same operations on the same draws.
    phi = ou_phi(half_life)
    seed = _pair_seeds(11, 4)[index]
    kernel_exits = _exits_one_pair(phi, forecast, sigma, start, pt, sl, max_hp, n_iter, seed)
    normals = _standard_normals(seed, n_iter * (max_hp + 1))
    py_exits, used = _otr_exits_python(phi, forecast, sigma, start, pt, sl, max_hp, n_iter, normals)
    assert kernel_exits.dtype == np.float64
    assert used <= normals.size
    assert np.array_equal(kernel_exits, py_exits)


def test_grid_rows_match_python_transcription_with_their_pair_seed() -> None:
    pts, sls = [0.5, 2.0], [1.0, 3.0]
    grid = otr_grid(0.25, 4.0, sigma=0.8, profit_taking=pts, stop_loss=sls, n_iter=200, max_hp=12, seed=5)
    phi = ou_phi(4.0)
    seeds = _pair_seeds(5, 4)
    for k, (pt, sl) in enumerate([(p, q) for p in pts for q in sls]):
        normals = _standard_normals(seeds[k], 200 * 13)
        exits, _ = _otr_exits_python(phi, 0.25, 0.8, 0.0, pt * 0.8, sl * 0.8, 12, 200, normals)
        row = grid.iloc[k]
        assert row["mean"] == pytest.approx(np.mean(exits), rel=1e-12, abs=1e-15)
        assert row["std"] == pytest.approx(np.std(exits), rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# Exit rule: strict barriers, holding cap, closed forms
# ---------------------------------------------------------------------------


def test_exit_test_is_strict_and_holding_cap_by_hand() -> None:
    # phi = 0.5, forecast 0, sigma 1, start 0.
    # Barrier equal to the value does not close; the path closes on the next update.
    exits, used = _otr_exits_python(0.5, 0.0, 1.0, 0.0, 1.5, BIG, 5, 1, np.array([1.5, 0.0, 2.0]))
    # update 1: p = 1.5, not > 1.5. update 2: p = 0.75. update 3: p = 0.375 + 2 = 2.375 > 1.5.
    assert exits.tolist() == [2.375]
    assert used == 3


def test_max_hp_one_holds_two_updates_by_hand() -> None:
    # hp > max_hp first holds at hp = 2 when max_hp = 1.
    # update 1: p = 1.0. update 2: p = 0.5 * 1.0 + 1.0 = 1.5, and 2 > 1 closes the path.
    exits, used = _otr_exits_python(0.5, 0.0, 1.0, 0.0, BIG, BIG, 1, 1, np.array([1.0, 1.0]))
    assert exits.tolist() == [1.5]
    assert used == 2


def test_max_hp_zero_closes_after_one_update_by_hand() -> None:
    exits, used = _otr_exits_python(0.5, 0.0, 1.0, 0.0, BIG, BIG, 0, 1, np.array([1.0]))
    assert exits.tolist() == [1.0]
    assert used == 1


def test_max_hp_one_kernel_uses_two_updates_per_path() -> None:
    n_iter, seed = 50, np.uint32(9)
    phi = ou_phi(2.0)
    kernel_exits = _exits_one_pair(phi, 0.0, 1.0, 0.0, BIG, BIG, 1, n_iter, seed)
    normals = _standard_normals(seed, 2 * n_iter)
    py_exits, used = _otr_exits_python(phi, 0.0, 1.0, 0.0, BIG, BIG, 1, n_iter, normals)
    assert used == 2 * n_iter
    assert np.array_equal(kernel_exits, py_exits)


def test_pt_zero_exits_on_first_positive_move() -> None:
    # With no stop and no cap, a path can only close on the first update above zero.
    exits = _exits_one_pair(ou_phi(5.0), 0.0, 1.0, 0.0, 0.0, BIG, 10**6, 4000, np.uint32(3))
    assert np.all(exits > 0.0)


def test_one_step_holding_matches_closed_form() -> None:
    # max_hp = 0 and huge barriers: each path closes on update 1, where
    # p1 = (1 - phi) * forecast + phi * start + sigma * g, and the exit is p1 - start.
    half_life, forecast, sigma, start, n, seed = 5.0, 2.0, 0.5, 0.25, 3000, 7
    grid = otr_grid(
        forecast, half_life, sigma=sigma, profit_taking=[BIG], stop_loss=[BIG],
        n_iter=n, max_hp=0, seed=seed, start=start,
    )
    phi = ou_phi(half_life)
    g = _standard_normals(_pair_seeds(seed, 1)[0], n)
    exits = (1.0 - phi) * forecast + phi * start + sigma * g - start
    assert grid.loc[0, "mean"] == pytest.approx(np.mean(exits), rel=1e-12, abs=1e-12)
    assert grid.loc[0, "std"] == pytest.approx(np.std(exits), rel=1e-12, abs=1e-12)


def test_sigma_scales_exit_values() -> None:
    kwargs = dict(forecast=0.0, half_life=4.0, profit_taking=[0.5, 2.0], stop_loss=[1.0], n_iter=400, max_hp=20, seed=2)
    unit = otr_grid(sigma=1.0, **kwargs)
    doubled = otr_grid(sigma=2.0, **kwargs)
    assert np.array_equal(unit["pt"], doubled["pt"])  # grid stays in units of sigma
    np.testing.assert_allclose(doubled["mean"], 2.0 * unit["mean"], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(doubled["std"], 2.0 * unit["std"], rtol=1e-12, atol=0.0)


def test_sharpe_is_nan_when_exits_are_constant() -> None:
    # half_life = 1e-3 gives phi = 2 ** -1000, so the price stays at forecast = start = 1e20
    # (the update is below one ulp). Every path then closes at cP = 0 with std = 0.
    grid = otr_grid(
        1e20, 1e-3, sigma=1e-3, profit_taking=[1.0], stop_loss=[1.0],
        n_iter=10, max_hp=3, seed=0, start=1e20,
    )
    assert grid.loc[0, "mean"] == 0.0
    assert grid.loc[0, "std"] == 0.0
    assert math.isnan(grid.loc[0, "sharpe"])


# ---------------------------------------------------------------------------
# otr_grid: output, determinism, threads
# ---------------------------------------------------------------------------


def test_grid_order_columns_and_sharpe() -> None:
    grid = otr_grid(0.0, 5.0, profit_taking=[0.0, 1.0, 2.0], stop_loss=[1.0, 3.0], n_iter=500, max_hp=20, seed=1)
    assert list(grid.columns) == ["pt", "sl", "mean", "std", "sharpe"]
    assert grid["pt"].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    assert grid["sl"].tolist() == [1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
    np.testing.assert_allclose(grid["sharpe"], grid["mean"] / grid["std"], rtol=1e-12)


def test_default_grid_is_21_by_21() -> None:
    grid = otr_grid(0.0, 5.0, n_iter=2, max_hp=1, seed=0)
    assert len(grid) == 441
    levels = np.linspace(0.0, 10.0, 21)
    assert np.allclose(np.unique(grid["pt"]), levels)
    assert np.allclose(np.unique(grid["sl"]), levels)


def test_same_seed_gives_identical_frames() -> None:
    kwargs = dict(profit_taking=np.linspace(0, 4, 5), stop_loss=np.linspace(0, 4, 5), n_iter=300, max_hp=40)
    first = otr_grid(0.0, 5.0, seed=12, **kwargs)
    second = otr_grid(0.0, 5.0, seed=12, **kwargs)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    other = otr_grid(0.0, 5.0, seed=13, **kwargs)
    assert not np.array_equal(first["mean"].to_numpy(), other["mean"].to_numpy())


def test_parallel_result_equals_serial_and_other_thread_counts() -> None:
    kwargs = dict(profit_taking=np.linspace(0, 4, 5), stop_loss=np.linspace(0, 4, 5), n_iter=300, max_hp=40, seed=21)
    default = otr_grid(0.0, 5.0, **kwargs)
    previous = numba.get_num_threads()
    try:
        for threads in sorted({1, min(2, numba.config.NUMBA_NUM_THREADS), numba.config.NUMBA_NUM_THREADS}):
            numba.set_num_threads(threads)
            other = otr_grid(0.0, 5.0, **kwargs)
            pd.testing.assert_frame_equal(default, other, check_exact=True)
    finally:
        numba.set_num_threads(previous)


def test_extending_the_grid_keeps_existing_rows() -> None:
    # Pair k is seeded from its index, so pair 0 of a larger grid is the same stream as a 1 x 1 grid.
    big = otr_grid(0.0, 5.0, profit_taking=[0.5, 2.0], stop_loss=[1.0, 3.0], n_iter=200, max_hp=10, seed=4)
    small = otr_grid(0.0, 5.0, profit_taking=[0.5], stop_loss=[1.0], n_iter=200, max_hp=10, seed=4)
    assert small.loc[0, "mean"] == big.loc[0, "mean"]
    assert small.loc[0, "std"] == big.loc[0, "std"]


def test_pair_seeds_are_distinct() -> None:
    seeds = _pair_seeds(0, 441)
    assert seeds.dtype == np.uint32
    assert len(set(seeds.tolist())) == 441


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"half_life": 0.0}, "half_life"),
        ({"half_life": -2.0}, "half_life"),
        ({"half_life": np.nan}, "half_life"),
        ({"half_life": np.inf}, "half_life"),
        ({"half_life": True}, "half_life"),
        ({"n_iter": 1}, "n_iter"),
        ({"n_iter": 0}, "n_iter"),
        ({"n_iter": True}, "n_iter"),
        ({"n_iter": 2.5}, "n_iter"),
        ({"max_hp": -1}, "max_hp"),
        ({"max_hp": True}, "max_hp"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"sigma": 0.0}, "sigma"),
        ({"sigma": -1.0}, "sigma"),
        ({"sigma": np.nan}, "sigma"),
        ({"start": np.nan}, "start"),
        ({"forecast": np.inf}, "forecast"),
        ({"profit_taking": [-1.0]}, "profit_taking"),
        ({"profit_taking": [np.nan]}, "profit_taking"),
        ({"profit_taking": []}, "profit_taking"),
        ({"profit_taking": [[1.0], [2.0]]}, "profit_taking"),
        ({"stop_loss": [-0.5]}, "stop_loss"),
    ],
)
def test_otr_grid_refusals(kwargs, match) -> None:
    args = dict(forecast=0.0, half_life=5.0, n_iter=10, max_hp=5, seed=0)
    args.update(kwargs)
    forecast = args.pop("forecast")
    half_life = args.pop("half_life")
    with pytest.raises(ValueError, match=match):
        otr_grid(forecast, half_life, **args)


@pytest.mark.parametrize("scale", [1e170, 1e-170])
def test_fit_ou_is_equivariant_in_price_scale_away_from_float_limits(scale: float) -> None:
    # Squares of prices near 1e170 overflow, and squares near 1e-170 underflow to zero.
    base = _simulate_ou(300, phi=0.8, sigma=1.0, forecast=0.0, seed=20261014)
    ref = fit_ou(base, forecast=0.0)
    fit = fit_ou(base * scale, forecast=0.0)
    assert fit.phi == pytest.approx(ref.phi, rel=1e-12)
    assert fit.sigma == pytest.approx(ref.sigma * scale, rel=1e-12)
    assert fit.half_life == pytest.approx(ref.half_life, rel=1e-12)
    assert fit.n_obs == ref.n_obs


@pytest.mark.parametrize("sigma", [1e-300, 1e300])
def test_otr_grid_is_scale_free_at_extreme_sigma(sigma: float) -> None:
    # Exits scale with sigma, so the Sharpe ratio must not depend on it. Squared deviations of exits
    # near 1e-300 underflow to zero, and near 1e300 they overflow to inf.
    kwargs = dict(
        forecast=0.0, half_life=5.0, profit_taking=[1.0, 2.0], stop_loss=[1.0],
        n_iter=400, max_hp=20, seed=3,
    )
    unit = otr_grid(sigma=1.0, **kwargs)
    grid = otr_grid(sigma=sigma, **kwargs)
    np.testing.assert_allclose(grid["std"], unit["std"] * sigma, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(grid["sharpe"], unit["sharpe"], rtol=1e-9, atol=0.0)


def test_otr_grid_refuses_barriers_that_overflow_float64() -> None:
    # sigma * 10 is inf, so the profit barrier would be silently absent.
    with pytest.raises(ValueError, match="sigma"):
        otr_grid(0.0, 5.0, sigma=1e308, profit_taking=[10.0], stop_loss=[1.0], n_iter=50, max_hp=100, seed=0)


def test_otr_grid_refuses_exits_that_overflow_float64() -> None:
    # Barriers are finite here, but sigma * eps overflows on some draws, so an exit is inf.
    with pytest.raises(ValueError, match="sigma"):
        otr_grid(0.0, 5.0, sigma=1e308, profit_taking=[0.5], stop_loss=[0.5], n_iter=50, max_hp=100, seed=0)


def test_scalar_arguments_accept_zero_dimensional_arrays() -> None:
    # finlab.strategy_risk takes a 0-d array as a scalar, so this module does too.
    assert ou_phi(np.array(5.0)) == ou_phi(5.0)
    assert ou_half_life(np.array(0.5)) == ou_half_life(0.5)
    assert fit_ou([1.0, 2.0, 0.0], np.array(0.0)).phi == fit_ou([1.0, 2.0, 0.0], 0.0).phi
    kwargs = dict(n_iter=50, profit_taking=[1.0], stop_loss=[1.0], seed=1)
    pd.testing.assert_frame_equal(
        otr_grid(np.array(0.0), np.array(5.0), sigma=np.array(1.0), start=np.array(0.0), **kwargs),
        otr_grid(0.0, 5.0, sigma=1.0, start=0.0, **kwargs),
    )


def test_ou_phi_rounds_to_endpoints_for_extreme_half_lives() -> None:
    # 2 ** (-1 / half_life) rounds to 0 below a half-life of about 9.3e-4 and to 1 above about
    # 1.25e16. ou_phi documents the closed interval, and otr_grid still simulates both cases.
    assert ou_phi(1e-6) == 0.0
    assert ou_phi(1e-3) > 0.0
    assert ou_phi(1e16) < 1.0
    assert ou_phi(1e17) == 1.0
    for half_life in (1e-6, 1e17):
        grid = otr_grid(0.0, half_life, n_iter=20, profit_taking=[1.0], stop_loss=[1.0], max_hp=3)
        assert np.isfinite(grid.loc[0, "mean"])


def test_scalars_beyond_float64_range_are_refused_by_name() -> None:
    with pytest.raises(ValueError, match="half_life"):
        ou_phi(10**400)
    with pytest.raises(ValueError, match="forecast"):
        otr_grid(10**400, 5.0, n_iter=10, profit_taking=[1.0], stop_loss=[1.0])


@pytest.mark.parametrize("name", ["n_iter", "max_hp"])
def test_otr_grid_refuses_counts_beyond_int64_by_name(name: str) -> None:
    # numba receives these as int64, so 2**63 raised OverflowError from the kernel call.
    kwargs = dict(n_iter=10, max_hp=3, profit_taking=[1.0], stop_loss=[1.0])
    kwargs[name] = 2**63
    with pytest.raises(ValueError, match=name):
        otr_grid(0.0, 5.0, **kwargs)
