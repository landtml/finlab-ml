"""Market microstructure features from price, high-low and volume data.

Covers AFML chapter 19 ("Microstructural Features"):

* :func:`tick_rule` -- trade-sign classification, section 19.3.1.
* :func:`roll_measure` -- Roll's (1984) effective spread from the serial
  covariance of price changes, section 19.3.2.
* :func:`becker_parkinson_volatility` -- high-low volatility estimator,
  section 19.3.3, and Snippet 19.2.
* :func:`corwin_schultz_spread` -- high-low bid-ask spread estimator,
  section 19.3.4, and Snippet 19.1.
* :func:`kyle_lambda` / :func:`kyle_lambda_tstat` -- price impact from the
  regression of price changes on signed volume, section 19.4.1.
* :func:`amihud_illiquidity` -- Amihud's lambda, section 19.4.2.
* :func:`bulk_volume_classification` and :func:`vpin` -- volume-clock order
  imbalance and VPIN, section 19.5.2. The book does not give a bulk-volume
  classifier; BVC is taken from Easley, Lopez de Prado and O'Hara (2012),
  "Flow Toxicity and Liquidity in a High-Frequency World".

Scope: each function takes one instrument's series. Series inputs return
Series (index preserved); numpy inputs return numpy arrays. Loops that are
not expressible as a single vectorized expression are compiled with numba.

Not covered: the Lee-Ready classifier, Hasbrouck's regression and its
time-bar sampling, the Kalman smoothing of spread estimates, the first-order
PIN mixture-of-Poissons MLE (section 19.5.1), and the book's Snippet 19.1
rolling implementation (its ``pd.stats.moments`` calls are obsolete; the
rolling windows are re-derived here and checked against a naive formula).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ._jit import jit

__all__ = [
    "tick_rule",
    "roll_measure",
    "becker_parkinson_volatility",
    "corwin_schultz_spread",
    "kyle_lambda",
    "kyle_lambda_tstat",
    "amihud_illiquidity",
    "bulk_volume_classification",
    "vpin",
]

ArrayLike = Union[np.ndarray, pd.Series]

_SQRT2 = math.sqrt(2.0)
_DEN = 3.0 - 2.0 * _SQRT2  # 3 - 2*sqrt(2), Corwin-Schultz denominator
_K2 = math.sqrt(8.0 / math.pi)  # Parkinson first-moment constant


def _to_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _rewrap(out: np.ndarray, like: ArrayLike) -> ArrayLike:
    if isinstance(like, pd.Series):
        return pd.Series(out, index=like.index, name=like.name)
    return out


# ---------------------------------------------------------------------------
# 19.3.1 Tick rule
# ---------------------------------------------------------------------------
@jit
def _tick_rule_kernel(prices: np.ndarray) -> np.ndarray:
    n = prices.shape[0]
    b = np.ones(n, dtype=np.int64)  # b_0 = 1 by convention
    for t in range(1, n):
        dp = prices[t] - prices[t - 1]
        if dp > 0.0:
            b[t] = 1
        elif dp < 0.0:
            b[t] = -1
        else:
            b[t] = b[t - 1]  # zero change carries the previous sign
    return b


def tick_rule(prices: ArrayLike) -> ArrayLike:
    """Classify each trade as buyer- or seller-initiated with the tick rule.

    Implements AFML eq. section 19.3.1::

        b_t = +1 if dp_t > 0,  -1 if dp_t < 0,  b_{t-1} if dp_t = 0,

    with ``b_0 = 1``.

    Parameters
    ----------
    prices : array-like or pandas.Series, shape (T,)
        Trade prices in time order.

    Returns
    -------
    numpy.ndarray or pandas.Series of int64, shape (T,)
        Aggressor flags in {-1, +1}. The first element is always +1.

    Notes
    -----
    Ties in price inherit the previous sign, so long runs at one price keep
    the sign of the last move. Proved in ``docs/proofs/microstructure.md``.

    Scope: one instrument, trade prices only. No quote data is used.
    """
    p = _to_float(prices, "prices")
    out = _tick_rule_kernel(p) if p.shape[0] else np.empty(0, dtype=np.int64)
    return _rewrap(out, prices)


# ---------------------------------------------------------------------------
# 19.3.2 Roll model
# ---------------------------------------------------------------------------
def roll_measure(prices: ArrayLike) -> float:
    """Roll (1984) effective bid-ask spread estimate.

    Under Roll's model, the serial covariance of price changes equals
    ``-c**2`` where ``c`` is half the spread (AFML section 19.3.2). The
    estimator is::

        S = 2 * sqrt(max(0, -Cov(dp_t, dp_{t-1})))

    and this function returns the full spread ``S = 2c``.

    Parameters
    ----------
    prices : array-like or pandas.Series, shape (T,)
        Trade or mid prices in time order. At least three observations are
        needed to form two price changes with a lag.

    Returns
    -------
    float
        Estimated full bid-ask spread in price units. Zero when the serial
        covariance is non-negative.

    Notes
    -----
    The covariance is the sample covariance over the lagged pairs
    ``(dp_t, dp_{t-1})``, ``t = 2..T-1``, with divisor ``n = T-2``. The book
    gives the estimator but no sampling distribution.

    Scope: a single price series. Price impact and non-zero drift violate
    the model's assumptions, so the estimate is biased in those cases.
    """
    p = _to_float(prices, "prices")
    if p.shape[0] < 3:
        raise ValueError("roll_measure needs at least 3 prices")
    dp = np.diff(p)
    x = dp[1:]
    y = dp[:-1]
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return 2.0 * math.sqrt(max(0.0, -cov))


# ---------------------------------------------------------------------------
# 19.3.3 / 19.3.4 High-low estimators (Becker-Parkinson, Corwin-Schultz)
# ---------------------------------------------------------------------------
@jit
def _hl_window_kernel(
    high: np.ndarray, low: np.ndarray, sl: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``beta`` and ``gamma`` (NaN where undefined).

    ``beta[t]`` is the mean over ``k = 0..sl-1`` of the two-bar sums
    ``h_{t-k}^2 + h_{t-k-1}^2`` with ``h_s = ln(H_s / L_s)``. Defined for
    ``t >= sl``. ``gamma[t]`` is ``ln(max(H_{t-1}, H_t) / min(L_{t-1}, L_t))^2``,
    defined for ``t >= 1``.
    """
    n = high.shape[0]
    h2 = np.empty(n, dtype=np.float64)
    for s in range(n):
        h = math.log(high[s] / low[s])
        h2[s] = h * h
    beta = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    for t in range(sl, n):
        acc = 0.0
        for k in range(sl):
            acc += h2[t - k] + h2[t - k - 1]
        beta[t] = acc / sl
    for t in range(1, n):
        hh = max(high[t - 1], high[t])
        ll = min(low[t - 1], low[t])
        g = math.log(hh / ll)
        gamma[t] = g * g
    return beta, gamma


def _check_hl(high: ArrayLike, low: ArrayLike, sl: int) -> tuple[np.ndarray, np.ndarray]:
    h = _to_float(high, "high")
    lo = _to_float(low, "low")
    if h.shape != lo.shape:
        raise ValueError("high and low must have the same length")
    if sl < 1:
        raise ValueError("sl must be >= 1")
    if np.any(~(h > 0)) or np.any(~(lo > 0)):
        raise ValueError("high and low must be strictly positive")
    if np.any(h < lo):
        raise ValueError("high must be >= low in every bar")
    return h, lo


def becker_parkinson_volatility(high: ArrayLike, low: ArrayLike, sl: int = 1) -> ArrayLike:
    """Becker-Parkinson volatility from high and low prices.

    AFML Snippet 19.2, derived from the Corwin-Schultz system::

        sigma_t = (2^{-1/2} - 1) * sqrt(beta_t) / (k2 * (3 - 2*sqrt(2)))
                  + sqrt(gamma_t / (k2^2 * (3 - 2*sqrt(2)))),   clipped at 0,

    with ``k2 = sqrt(8/pi)``. ``beta_t`` and ``gamma_t`` are as in
    :func:`corwin_schultz_spread`.

    Parameters
    ----------
    high, low : array-like or pandas.Series, shape (T,)
        Bar high and low prices.
    sl : int, default 1
        Number of two-bar sums averaged into ``beta``.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (T,)
        Volatility per bar, NaN where ``beta`` or ``gamma`` is undefined
        (``t < sl`` for beta, ``t = 0`` for gamma).

    Notes
    -----
    The formula was checked by simulation on a GBM path (20,000 bars, 400
    steps per bar): the average estimate is within 10% of the true sigma at
    the test seed, and about 5% low (0.01893 against 0.02 at seed 0). The
    book gives no proof, so the formula is "claimed from the book" and only
    the simulation check supports it. See ``docs/proofs/microstructure.md``.

    Scope: continuous-time volatility proxy from bar extremes. Discrete
    monitoring of the path biases the estimate downward.
    """
    h, lo = _check_hl(high, low, sl)
    beta, gamma = _hl_window_kernel(h, lo, sl)
    k2 = _K2
    with np.errstate(invalid="ignore"):
        sigma = (2.0**-0.5 - 1.0) * np.sqrt(beta) / (k2 * _DEN) + np.sqrt(
            gamma / (k2 * k2 * _DEN)
        )
    sigma = np.where(sigma < 0.0, 0.0, sigma)
    return _rewrap(sigma, high)


def corwin_schultz_spread(high: ArrayLike, low: ArrayLike, sl: int = 1) -> ArrayLike:
    """Corwin-Schultz (2012) high-low bid-ask spread estimator.

    AFML section 19.3.4 and Snippet 19.1::

        S_t = 2 (exp(alpha_t) - 1) / (1 + exp(alpha_t)),
        alpha_t = (sqrt(2) - 1) sqrt(beta_t) / (3 - 2 sqrt(2))
                  - sqrt(gamma_t / (3 - 2 sqrt(2))),

    with ``alpha_t`` set to 0 whenever it is negative (the book's rule).

    Parameters
    ----------
    high, low : array-like or pandas.Series, shape (T,)
        Bar high and low prices, strictly positive with ``high >= low``.
    sl : int, default 1
        Number of two-bar sums averaged into ``beta``.

    Returns
    -------
    numpy.ndarray or pandas.Series, shape (T,)
        Estimated spread as a fraction of price, NaN for ``t < sl``.

    Notes
    -----
    Because alpha is clipped at 0, the output is always non-negative. The
    output is a fraction of price, not a price-unit spread.

    Scope: one instrument, bar high and low only. It does not use closes or
    quotes. The book's rolling windows follow Snippet 19.1, re-derived here
    for numpy input.
    """
    h, lo = _check_hl(high, low, sl)
    beta, gamma = _hl_window_kernel(h, lo, sl)
    with np.errstate(invalid="ignore"):
        alpha = (_SQRT2 - 1.0) * np.sqrt(beta) / _DEN - np.sqrt(gamma / _DEN)
    alpha = np.where(alpha < 0.0, 0.0, alpha)
    spread = 2.0 * np.expm1(alpha) / (1.0 + np.exp(alpha))
    return _rewrap(spread, high)


# ---------------------------------------------------------------------------
# 19.4 Strategic trade models
# ---------------------------------------------------------------------------
@jit
def _ols_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and t-statistic of ``y = lambda * x + e`` with no intercept."""
    n = x.shape[0]
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        sxx += x[i] * x[i]
        sxy += x[i] * y[i]
    lam = sxy / sxx
    ssr = 0.0
    for i in range(n):
        e = y[i] - lam * x[i]
        ssr += e * e
    dof = n - 1
    if ssr == 0.0:
        # Exact fit: the standard error is zero, so t is +/- infinity.
        if lam > 0.0:
            return lam, np.inf
        if lam < 0.0:
            return lam, -np.inf
        return lam, np.nan
    se = math.sqrt(ssr / dof / sxx)
    return lam, lam / se


def _kyle_inputs(prices: ArrayLike, signed_volume: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` with regressor ``x = signed volume`` and response ``y = dp``."""
    p = _to_float(prices, "prices")
    v = _to_float(signed_volume, "signed_volume")
    if p.shape != v.shape:
        raise ValueError("prices and signed_volume must have the same length")
    if p.shape[0] < 3:
        raise ValueError("kyle_lambda needs at least 3 observations")
    return v[1:], np.diff(p)


def kyle_lambda(prices: ArrayLike, signed_volume: ArrayLike) -> float:
    """Kyle's lambda: price impact per unit of signed volume.

    Estimates ``dp_t = lambda * (b_t V_t) + e_t`` (AFML section 19.4.1) by
    OLS without intercept, where ``b_t V_t`` is ``signed_volume``.

    Parameters
    ----------
    prices : array-like, shape (T,)
        Prices in time order.
    signed_volume : array-like, shape (T,)
        Signed volume ``b_t * V_t`` for each price. Element ``t`` is paired
        with the change ``p_t - p_{t-1}``, so element 0 is unused.

    Returns
    -------
    float
        The slope ``lambda = sum(x*y) / sum(x*x)`` with ``x`` the signed
        volume and ``y`` the price change.

    Notes
    -----
    Without an intercept, the OLS slope is exact for the model in the book.
    The book notes that the t-value is a more informative feature than the
    mean estimate; use :func:`kyle_lambda_tstat` for that.

    Scope: one instrument over one estimation window. A zero-volume window
    gives division by zero, which is raised as ``ValueError``.
    """
    x, y = _kyle_inputs(prices, signed_volume)
    sxx = float(np.dot(x, x))
    if sxx == 0.0:
        raise ValueError("signed_volume is identically zero")
    return float(np.dot(x, y) / sxx)


def kyle_lambda_tstat(prices: ArrayLike, signed_volume: ArrayLike) -> tuple[float, float]:
    """Kyle's lambda and its t-statistic.

    Parameters
    ----------
    prices, signed_volume : array-like, shape (T,)
        As in :func:`kyle_lambda`.

    Returns
    -------
    tuple of (float, float)
        ``(lambda, t)`` where ``t = lambda / se(lambda)`` and the standard
        error uses ``T - 2`` degrees of freedom (one slope, no intercept).

    Notes
    -----
    Standard OLS through the origin. The residual variance divides by
    ``n - 1`` where ``n`` is the number of regression rows. This is the
    estimator used for the t-value feature the book recommends.
    """
    x, y = _kyle_inputs(prices, signed_volume)
    if float(np.dot(x, x)) == 0.0:
        raise ValueError("signed_volume is identically zero")
    lam, t = _ols_origin(x, y)
    return float(lam), float(t)


def amihud_illiquidity(
    returns: ArrayLike, dollar_volume: ArrayLike, method: str = "regression"
) -> float:
    """Amihud's illiquidity measure from bar returns and dollar volume.

    Parameters
    ----------
    returns : array-like, shape (B,)
        Bar log returns ``log(p_tau / p_{tau-1})``.
    dollar_volume : array-like, shape (B,)
        Dollar volume ``sum_t p_t V_t`` traded in each bar, positive.
    method : {"regression", "ratio"}, default "regression"
        ``"regression"``: AFML section 19.4.2, the OLS slope of
        ``|r_tau|`` on ``DV_tau`` without intercept,
        ``lambda = sum(|r| DV) / sum(DV^2)``.
        ``"ratio"``: the classic Amihud ratio, mean of ``|r_tau| / DV_tau``.

    Returns
    -------
    float
        Illiquidity estimate. Larger means more price impact per dollar.

    Notes
    -----
    The two methods are different statistics and are not expected to match.
    The regression form is the book's definition. The ratio form is the
    original Amihud (2002) and is included for comparison.

    Scope: one instrument. Bars with zero dollar volume are rejected in
    ``"ratio"`` mode and are allowed but uninformative in ``"regression"``
    mode.
    """
    r = np.abs(_to_float(returns, "returns"))
    dv = _to_float(dollar_volume, "dollar_volume")
    if r.shape != dv.shape:
        raise ValueError("returns and dollar_volume must have the same length")
    if r.shape[0] == 0:
        raise ValueError("empty input")
    if method == "regression":
        den = float(np.dot(dv, dv))
        if den == 0.0:
            raise ValueError("dollar_volume is identically zero")
        return float(np.dot(r, dv) / den)
    if method == "ratio":
        if np.any(dv <= 0.0):
            raise ValueError("ratio method needs strictly positive dollar_volume")
        return float(np.mean(r / dv))
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# 19.5.2 VPIN: bulk volume classification and volume buckets
# ---------------------------------------------------------------------------
def bulk_volume_classification(
    prices: ArrayLike, volume: ArrayLike, sigma: float | None = None
) -> np.ndarray:
    """Buy-volume fraction from bulk volume classification (BVC).

    For each observation, the buy fraction is::

        z_t = (p_t - p_{t-1}) / sigma,       buy_frac_t = Phi(z_t),

    where ``Phi`` is the standard normal CDF and ``sigma`` is the standard
    deviation of price changes. The buy volume is ``V_t * buy_frac_t`` and
    the sell volume is ``V_t * (1 - buy_frac_t)``.

    Parameters
    ----------
    prices : array-like, shape (T,)
        Prices in time order.
    volume : array-like, shape (T,)
        Non-negative volume for each price.
    sigma : float, optional
        Scale of price changes. If a float, a fixed scale is used for every bar
        (this is the only way to pass a scale estimated outside the sample). If
        omitted, an expanding-window estimate is used: for bar ``t`` the sample
        standard deviation of the price changes up to bar ``t-1``. Using only past
        changes keeps the classification causal, with no look-ahead. Bars without
        enough history fall back to 0.5.

    Returns
    -------
    numpy.ndarray, shape (T,)
        Buy fraction in ``[0, 1]``. Element 0 is 0.5 (no prior price).

    Notes
    -----
    This is the classifier of Easley, Lopez de Prado and O'Hara (2012), not
    part of AFML's text. The book names the tick rule and Lee-Ready as
    alternatives. Under this classifier, E[buy_frac] = 1/2 when price
    changes are symmetric about zero.

    Scope: one instrument.
    """
    p = _to_float(prices, "prices")
    v = _to_float(volume, "volume")
    if p.shape != v.shape:
        raise ValueError("prices and volume must have the same length")
    if np.any(v < 0.0):
        raise ValueError("volume must be non-negative")
    buy = np.full(p.shape[0], 0.5)
    if p.shape[0] < 2:
        return buy
    dp = np.diff(p)
    if sigma is not None:
        if not sigma > 0.0:
            raise ValueError("sigma must be positive (price changes have no variation)")
        buy[1:] = ndtr(dp / sigma)
        return buy
    # Causal expanding-window scale: sd of dp[0..t-1] for the change dp[t].
    # Bar t uses only changes before it, so no future prices enter the estimate.
    n = dp.shape[0]
    csum = np.concatenate(([0.0], np.cumsum(dp)))
    csq = np.concatenate(([0.0], np.cumsum(dp * dp)))
    k = np.arange(n, dtype=np.float64)  # past changes dp[0..j-1] for dp[j]
    k_safe = np.maximum(k, 1.0)
    mean = csum[:-1] / k_safe
    var = (csq[:-1] - k * mean * mean) / np.maximum(k - 1.0, 1.0)
    scale = np.sqrt(np.maximum(var, 0.0))
    ok = (k >= 2) & (scale > 0.0)
    z = np.zeros(n)
    z[ok] = dp[ok] / scale[ok]
    out = ndtr(z)
    out[~ok] = 0.5
    buy[1:] = out
    return buy


@jit
def _fill_buckets(
    volume: np.ndarray, buy_frac: np.ndarray, bucket_volume: float, max_buckets: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack observations into equal-volume buckets.

    An observation that straddles a bucket boundary is split pro rata, so
    each bucket holds exactly ``bucket_volume`` units of volume.
    """
    out_b = np.zeros(max_buckets, dtype=np.float64)
    out_s = np.zeros(max_buckets, dtype=np.float64)
    nb = 0
    fill = 0.0
    cur_b = 0.0
    cur_s = 0.0
    n = volume.shape[0]
    tol = 1e-12 * bucket_volume
    for i in range(n):
        v = volume[i]
        if v <= 0.0:
            continue
        bi = v * buy_frac[i]
        si = v - bi
        rem = v
        while rem > 0.0:
            space = bucket_volume - fill
            take = space if space < rem else rem
            f = take / v
            cur_b += bi * f
            cur_s += si * f
            fill += take
            rem -= take
            if bucket_volume - fill <= tol:
                if nb < max_buckets:
                    out_b[nb] = cur_b
                    out_s[nb] = cur_s
                nb += 1
                cur_b = 0.0
                cur_s = 0.0
                fill = 0.0
    return out_b, out_s, nb


def vpin(
    prices: ArrayLike,
    volume: ArrayLike,
    bucket_volume: float,
    n_window: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Volume-synchronized probability of informed trading (VPIN).

    Implements AFML section 19.5.2::

        VPIN_tau = sum_{k=tau-n+1}^{tau} |V^B_k - V^S_k| / (n * V)

    where ``V`` is the bucket volume and ``V^B``, ``V^S`` are the buy and
    sell volumes in each bucket. Buy/sell volumes come from
    :func:`bulk_volume_classification`.

    Parameters
    ----------
    prices : array-like, shape (T,)
        Trade or bar prices in time order.
    volume : array-like, shape (T,)
        Non-negative trade or bar volume aligned with ``prices``.
    bucket_volume : float
        Volume per bucket ``V`` (strictly positive).
    n_window : int
        Number of buckets ``n`` in the rolling window (at least 1).
    sigma : float, optional
        Passed to :func:`bulk_volume_classification`.

    Returns
    -------
    numpy.ndarray, shape (K,)
        VPIN for each completed bucket, where ``K`` is the number of full
        buckets. Entries before the first full window are NaN. A trailing
        partial bucket is dropped.

    Notes
    -----
    VPIN lies in ``[0, 1]`` by construction because
    ``|V^B - V^S| <= V^B + V^S = V`` in each bucket. Proved in
    ``docs/proofs/microstructure.md``.

    Scope: one instrument. The per-bucket volume split uses the BVC buy
    fraction of the observation that straddles each boundary, which is an
    approximation.
    """
    p = _to_float(prices, "prices")
    v = _to_float(volume, "volume")
    if bucket_volume <= 0.0:
        raise ValueError("bucket_volume must be positive")
    if n_window < 1:
        raise ValueError("n_window must be >= 1")
    buy_frac = bulk_volume_classification(p, v, sigma=sigma)
    total = float(v.sum())
    max_buckets = int(math.floor(total / bucket_volume)) + 2
    vb, vs, nb = _fill_buckets(v, buy_frac, float(bucket_volume), max_buckets)
    nb = min(nb, max_buckets)
    # Complete buckets only. The kernel writes a bucket for every boundary.
    full = int(math.floor(total / bucket_volume + 1e-12))
    nb = min(nb, full)
    out = np.full(nb, np.nan)
    if nb < n_window:
        return out
    imb = np.abs(vb[:nb] - vs[:nb])
    csum = np.concatenate(([0.0], np.cumsum(imb)))
    # window ending at bucket tau (0-based) covers imb[tau-n+1 .. tau]
    idx = np.arange(n_window - 1, nb)
    window = csum[idx + 1] - csum[idx + 1 - n_window]
    out[idx] = window / (n_window * bucket_volume)
    return out
