"""Optimal trading rules on an Ornstein-Uhlenbeck process (AFML chapter 13).

The discrete Ornstein-Uhlenbeck (OU) model of AFML section 13.5 is::

    P_t = (1 - phi) * forecast + phi * P_{t-1} + sigma * eps_t,   eps_t ~ N(0, 1)

* :func:`ou_half_life` and :func:`ou_phi` convert between the AR(1) coefficient
  phi and the half-life tau = -log(2) / log(phi).
* :func:`fit_ou` estimates phi, sigma and the half-life from a price path by
  least squares without intercept, around the forecast (Snippet 13.1).
* :func:`otr_grid` simulates the exit value of a holding rule for every pair of
  profit-taking and stop-loss levels, and summarises each pair by the mean, the
  standard deviation and the Sharpe ratio of the exits (Snippet 13.2).

Notes
-----
The snippet and section numbers are cited from the book. They were not checked
against the book's text.

The simulation runs in a numba kernel, parallel over the grid pairs. Each pair
seeds its own random stream from the user seed and its index in the grid, so the
result does not depend on the number of threads. The draws come from numba's
random generator, not from ``numpy.random.default_rng``.

Scope
-----
A single holding rule on a stationary OU price with a fixed forecast. No
transaction costs, no estimation error in phi and sigma, and no search beyond
the grid that the caller supplies.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import prange

from quantowl._checks import _require_real_values
from quantowl._jit import jit, pjit

__all__ = ["OUFit", "fit_ou", "otr_grid", "ou_half_life", "ou_phi"]

_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class OUFit:
    """Result of :func:`fit_ou`.

    Attributes
    ----------
    phi : float
        Estimated AR(1) coefficient of the deviations from the forecast.
    sigma : float
        Sample standard deviation of the regression residuals (ddof=1). This is
        the innovation scale, not the stationary standard deviation
        sigma / sqrt(1 - phi^2).
    half_life : float
        ``ou_half_life(phi)`` when 0 < phi < 1. ``inf`` when phi >= 1, where the
        process does not revert. ``nan`` when phi <= 0, where the decay is not
        monotone and no half-life is defined.
    n_obs : int
        Number of (previous, current) price pairs in the regression, len(prices) - 1.
    """

    phi: float
    sigma: float
    half_life: float
    n_obs: int


def ou_half_life(phi: float) -> float:
    """Half-life, in steps, of an OU process with AR(1) coefficient ``phi``.

    Parameters
    ----------
    phi : float
        Decay factor per step. Must lie strictly between 0 and 1.

    Returns
    -------
    float
        tau = -log(2) / log(phi). Each step multiplies the expected deviation from
        the forecast by ``phi``, so the deviation halves after tau steps.

    Raises
    ------
    ValueError
        If ``phi`` is not a finite real number, or lies outside (0, 1).

    Notes
    -----
    Derivation in ``docs/proofs/trading_rules.md``, section 1.
    """
    x = _finite("phi", phi)
    if not 0.0 < x < 1.0:
        raise ValueError(f"phi must lie strictly between 0 and 1, got {x}")
    return -math.log(2.0) / math.log(x)


def ou_phi(half_life: float) -> float:
    """AR(1) coefficient of an OU process with the given half-life.

    Parameters
    ----------
    half_life : float
        Half-life in steps. Must be finite and positive.

    Returns
    -------
    float
        phi = 2 ** (-1 / half_life), in [0, 1]. The value rounds to 0 for a half-life below
        about 9.3e-4 and to 1 for one above about 1.25e16. :func:`ou_half_life` accepts only
        the open interval (0, 1).

    Raises
    ------
    ValueError
        If ``half_life`` is not finite, or is not positive.

    Examples
    --------
    >>> ou_phi(5.0) == 2 ** (-0.2)
    True
    """
    tau = _finite("half_life", half_life)
    if tau <= 0.0:
        raise ValueError(f"half_life must be positive, got {tau}")
    return 2.0 ** (-1.0 / tau)


def fit_ou(prices, forecast: float) -> OUFit:
    """Estimate the OU parameters of a price path around a fixed forecast.

    Parameters
    ----------
    prices : array-like or Series, shape (n,)
        Price path in time order. Must be finite, with n >= 3.
    forecast : float
        Mean the process reverts to, the same value used in :func:`otr_grid`.

    Returns
    -------
    OUFit
        ``phi``, ``sigma``, ``half_life`` and ``n_obs``.

    Raises
    ------
    ValueError
        If ``prices`` is not one-dimensional (one column), has fewer than 3 values,
        or contains non-finite values; if ``forecast`` is not finite; or if every
        lagged price equals the forecast, which leaves phi undefined.

    Notes
    -----
    With x_t = P_{t-1} - forecast and y_t = P_t - forecast, phi is the least-squares
    slope without intercept, sum(x * y) / sum(x * x). sigma is the sample standard
    deviation of y - phi * x with ddof=1. The half-life follows the rule in
    :class:`OUFit`.
    """
    f = _finite("forecast", forecast)
    x = _as_values("prices", prices)
    if x.size < 3:
        raise ValueError(f"prices must hold at least 3 values, got {x.size}")
    lag = x[:-1] - f
    cur = x[1:] - f
    if not np.any(lag):
        raise ValueError("every lagged price equals forecast, so phi is undefined")
    # Dividing by the largest deviation keeps the squares finite. phi does not change with the
    # scale, and sigma scales with it.
    mag = max(float(np.max(np.abs(lag))), float(np.max(np.abs(cur))))
    u = lag / mag
    v = cur / mag
    phi = float(np.sum(u * v)) / float(np.sum(u * u))
    sigma = mag * float(np.std(v - phi * u, ddof=1))
    return OUFit(phi=phi, sigma=sigma, half_life=_fitted_half_life(phi), n_obs=int(x.size - 1))


def otr_grid(
    forecast: float,
    half_life: float,
    sigma: float = 1.0,
    profit_taking=None,
    stop_loss=None,
    n_iter: int = 100_000,
    max_hp: int = 100,
    seed: int = 0,
    start: float = 0.0,
) -> pd.DataFrame:
    """Simulate the exit value of a holding rule for each profit-taking and stop-loss pair.

    Each path starts at ``start`` and follows the OU update
    ``p = (1 - phi) * forecast + phi * p + sigma * eps``. Let ``cP = p - start``
    and ``hp`` the number of updates so far. The path closes at the first update
    where ``cP > pt * sigma``, or ``cP < -sl * sigma``, or ``hp > max_hp``, and
    its exit value is the ``cP`` at that update.

    Parameters
    ----------
    forecast : float
        Mean that the process reverts to.
    half_life : float
        Half-life of the process in steps, finite and positive. phi = ou_phi(half_life).
    sigma : float, default 1.0
        Innovation scale. Must be finite and positive. Barriers are in units of sigma.
    profit_taking : array-like, optional
        Profit-taking levels in units of sigma, finite and non-negative. Defaults to
        ``np.linspace(0, 10, 21)``.
    stop_loss : array-like, optional
        Stop-loss levels in units of sigma, finite and non-negative. Defaults to
        ``np.linspace(0, 10, 21)``.
    n_iter : int, default 100_000
        Number of simulated paths per pair. Must be at least 2.
    max_hp : int, default 100
        Holding cap. A path closes at the update where the count exceeds ``max_hp``,
        so it lasts at most ``max_hp + 1`` updates. Must be a non-negative integer.
    seed : int, default 0
        Non-negative integer. Pair k draws from a stream seeded by ``seed`` and k.
    start : float, default 0.0
        Starting price. Exit values are measured from it.

    Returns
    -------
    pandas.DataFrame
        One row per pair, pair order is profit-taking outer, stop-loss inner. Columns:

        * ``pt``, ``sl``: the grid values, in units of sigma (not multiplied).
        * ``mean``: mean of the exit values, in price units.
        * ``std``: standard deviation of the exit values, ddof=0.
        * ``sharpe``: ``mean / std``. ``nan`` when ``std`` is zero.

    Raises
    ------
    ValueError
        If any argument is not a finite real number where one is required, if
        ``half_life`` or ``sigma`` is not positive, if ``n_iter`` is below 2, if
        ``max_hp`` or ``seed`` is negative or not an integer, or if a grid is empty,
        not one-dimensional, non-finite or negative. Also if ``sigma`` times the grid, or a
        simulated exit, overflows float64.

    Notes
    -----
    Snippet 13.2 of AFML (section 13.5) is the source of the procedure. The cited
    section and snippet numbers are not checked against the book's text.

    The result depends only on the arguments. Pair k is simulated by one kernel call
    that seeds numba's generator with the k-th child of ``numpy.random.SeedSequence``
    spawned from ``seed``. The number of threads does not change the output. The
    kernel does not use fastmath, so the floating-point operations run in the order
    written.

    Holding time is counted in price updates. The exit test is strict on both barriers.
    """
    f = _finite("forecast", forecast)
    tau = _finite("half_life", half_life)
    if tau <= 0.0:
        raise ValueError(f"half_life must be positive, got {tau}")
    s = _finite("sigma", sigma)
    if s <= 0.0:
        raise ValueError(f"sigma must be positive, got {s}")
    st = _finite("start", start)
    n = _nonneg_int("n_iter", n_iter, 2)
    holding = _nonneg_int("max_hp", max_hp, 0)
    for name, count in (("n_iter", n), ("max_hp", holding)):
        # The kernel receives both as int64, so a larger value would fail inside numba.
        if count > _INT64_MAX:
            raise ValueError(f"{name} must be at most {_INT64_MAX}, got {count}")
    root = _nonneg_int("seed", seed, 0)
    pts = _grid("profit_taking", profit_taking)
    sls = _grid("stop_loss", stop_loss)

    phi = ou_phi(tau)
    pt_col = np.repeat(pts, sls.size)
    sl_col = np.tile(sls, pts.size)
    n_pairs = pt_col.size
    with np.errstate(over="ignore"):  # overflow is caught by the finiteness check below
        pt_barrier = pt_col * s
        sl_barrier = sl_col * s
    # An infinite barrier would never be hit, so the result would be silently wrong.
    if not (np.all(np.isfinite(pt_barrier)) and np.all(np.isfinite(sl_barrier))):
        raise ValueError("sigma times the profit_taking or stop_loss grid overflows float64")
    means = np.empty(n_pairs, dtype=np.float64)
    stds = np.empty(n_pairs, dtype=np.float64)
    _otr_grid_kernel(
        phi, f, s, st, pt_barrier, sl_barrier, holding, n, _pair_seeds(root, n_pairs), means, stds
    )
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(stds))):
        raise ValueError("simulated exits overflow float64 for this sigma; use a smaller sigma")

    sharpe = np.full(n_pairs, np.nan)
    ok = stds > 0.0
    sharpe[ok] = means[ok] / stds[ok]
    return pd.DataFrame({"pt": pt_col, "sl": sl_col, "mean": means, "std": stds, "sharpe": sharpe})


def _finite(name: str, value: object) -> float:
    """Return ``value`` as a finite float; refuse bools, non-real and non-finite values."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        x = float(value)
    except OverflowError:
        raise ValueError(f"{name} is outside the float64 range") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return x


def _nonneg_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum``; refuse bools and non-integers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    n = int(value)
    if n < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {n}")
    return n


def _as_values(name: str, x: object) -> np.ndarray:
    """Coerce a one-column price container to a 1-D finite float64 array."""
    raw = np.asarray(x)
    _require_real_values(name, raw)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional (one column), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _grid(name: str, values: object) -> np.ndarray:
    """Return a grid of barrier levels in units of sigma, or the default grid for None."""
    if values is None:
        return np.linspace(0.0, 10.0, 21)
    raw = np.asarray(values)
    _require_real_values(name, raw)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative, got {arr.min()}")
    return arr


def _fitted_half_life(phi: float) -> float:
    """Half-life of a fitted phi, with inf for phi >= 1 and nan for phi <= 0."""
    if phi >= 1.0:
        return math.inf
    if phi <= 0.0:
        return math.nan
    return ou_half_life(phi)


def _pair_seeds(root: int, n_pairs: int) -> np.ndarray:
    """One uint32 seed per grid pair, derived from ``root`` and the pair index.

    Each pair gets its own ``spawn_key``, so its seed depends on the root and on
    its index only, not on the other pairs in the grid.
    """
    seeds = [
        np.random.SeedSequence(root, spawn_key=(k,)).generate_state(1, dtype=np.uint32)[0]
        for k in range(n_pairs)
    ]
    return np.array(seeds, dtype=np.uint32)


@jit
def _standard_normals(seed, n):
    """``n`` standard normal draws from numba's generator, seeded with ``seed``."""
    np.random.seed(seed)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.random.standard_normal()
    return out


@jit
def _exits_one_pair(phi, forecast, sigma, start, pt, sl, max_hp, n_iter, seed):
    """Exit values of ``n_iter`` paths for one barrier pair, in path order.

    The stream is seeded once per pair. Each path draws one normal per update and
    keeps drawing until it closes, so a path's draws do not depend on other pairs.
    """
    np.random.seed(seed)
    out = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        p = start
        hp = 0
        while True:
            p = (1.0 - phi) * forecast + phi * p + sigma * np.random.standard_normal()
            cp = p - start
            hp += 1
            if cp > pt or cp < -sl or hp > max_hp:
                out[i] = cp
                break
    return out


@jit
def _mean_std(x):
    """Mean and population standard deviation (ddof=0), two passes in index order.

    The exits are divided by their largest magnitude first, so the squared deviations
    neither overflow nor underflow. A non-finite exit returns nan for both statistics.
    Written as loops rather than ``np.mean`` and ``np.std``: the NumPy versions add
    about 0.7 s to the first compile of the kernel.
    """
    n = x.shape[0]
    scale = 0.0
    for i in range(n):
        if not math.isfinite(x[i]):
            return math.nan, math.nan
        scale = max(scale, abs(x[i]))
    if scale == 0.0:
        return 0.0, 0.0
    total = 0.0
    for i in range(n):
        total += x[i] / scale
    mu = total / n
    ss = 0.0
    for i in range(n):
        d = x[i] / scale - mu
        ss += d * d
    return mu * scale, math.sqrt(ss / n) * scale


@pjit
def _otr_grid_kernel(phi, forecast, sigma, start, pts, sls, max_hp, n_iter, seeds, means, stds):
    """Mean and std of the exit values for every pair, parallel over pairs.

    Each pair writes only its own slot of ``means`` and ``stds``, so the threads
    share no state and the output is the same for any thread count.
    """
    for k in prange(pts.shape[0]):
        exits = _exits_one_pair(phi, forecast, sigma, start, pts[k], sls[k], max_hp, n_iter, seeds[k])
        mu, sd = _mean_std(exits)
        means[k] = mu
        stds[k] = sd


def _otr_exits_python(phi, forecast, sigma, start, pt, sl, max_hp, n_iter, normals):
    """Pure-Python transcription of the path loop, fed with pre-drawn standard normals.

    Consumes ``normals`` in order, one per update, and returns the exit values and
    the number of normals used. Its arithmetic follows ``_exits_one_pair`` operation
    for operation, so the two agree bit for bit on the same draws.
    """
    exits = np.empty(n_iter, dtype=np.float64)
    k = 0
    for i in range(n_iter):
        p = start
        hp = 0
        while True:
            p = (1.0 - phi) * forecast + phi * p + sigma * float(normals[k])
            k += 1
            cp = p - start
            hp += 1
            if cp > pt or cp < -sl or hp > max_hp:
                exits[i] = cp
                break
    return exits, k
