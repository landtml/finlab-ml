"""Backtest statistics: Sharpe ratio, PSR, expected maximum SR, DSR, minTRL, and the bet and drawdown statistics of AFML ch. 14.

Implements the efficiency statistics of AFML chapter 14 (section 14.7):

* :func:`sharpe_ratio` -- annualised Sharpe ratio, AFML 14.7.1 / 14.7.4.
* :func:`probabilistic_sharpe_ratio` -- PSR, AFML 14.7.2 (eq. displayed on
  p. 203; Bailey and Lopez de Prado 2012).
* :func:`expected_max_sharpe` -- the Euler-Mascheroni approximation of the
  expected maximum of N IID Normal Sharpe estimates, AFML 14.7.3.
* :func:`deflated_sharpe_ratio` -- DSR, AFML 14.7.3: PSR evaluated against the
  expected-maximum benchmark instead of a user-chosen SR*.
* :func:`deflated_sharpe` -- the same DSR, returned with its inputs and SR* as a
  :class:`DeflatedSharpeResult`.
* :func:`min_track_record_length` -- minTRL, the smallest T at which PSR reaches
  a given confidence (Bailey and Lopez de Prado 2012; not in the AFML text).

Implements the bet and drawdown statistics of AFML ch. 14 (snippets 14.1 to 14.4):

* :func:`bet_timing` -- timestamps where a bet ends (flattenings and flips), Snippet 14.1.
* :func:`holding_period` -- average holding period weighted by the size closed, Snippet 14.2.
* :func:`hhi_concentration` -- HHI of a return series, Snippet 14.3.
* :func:`return_concentration` -- HHI of the positive returns, the negative returns and the
  returns per calendar period, Snippet 14.3.
* :func:`drawdown_time_under_water` -- drawdowns and time under water, Snippet 14.4.

Notes
-----
All ``sr_*`` arguments of the PSR/DSR/minTRL functions are *non-annualised*
Sharpe ratios, i.e. computed on the sampling frequency of the observations.
The annualised value returned by :func:`sharpe_ratio` must be de-annualised
(divided by sqrt(periods_per_year)) before being fed into these functions.
``skew`` is the sample skewness and ``kurt`` the (non-excess) kurtosis, so
``kurt == 3`` for Gaussian returns.

Scope
-----
Mostly scalar statistics. The bet and drawdown functions return a DatetimeIndex
or a pair of Series. No multiple-testing corrections beyond DSR, and no
estimation of skewness/kurtosis (callers compute them, e.g. with
``scipy.stats.skew`` and ``scipy.stats.kurtosis(fisher=False)``).

Not covered: AFML 14.7.4 information ratio, implied precision (14.4-14.6), and the
variance estimation of the trial Sharpe ratios, which is left to the caller
(``var_sr`` is an input).
"""

from __future__ import annotations

import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from finlab._checks import _require_real_values, _require_type, as_datetime_index
from finlab._display import html_table
from finlab._jit import jit

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "deflated_sharpe",
    "DeflatedSharpeResult",
    "min_track_record_length",
    "bet_timing",
    "holding_period",
    "hhi_concentration",
    "return_concentration",
    "ReturnConcentration",
    "drawdown_time_under_water",
]

EULER_MASCHERONI: float = 0.5772156649015329

ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray, list]


def sharpe_ratio(
    returns: ArrayLike,
    periods_per_year: float = 252.0,
    *,
    annualize: bool = True,
) -> float:
    """Sharpe ratio of a return series, optionally annualised.

    Parameters
    ----------
    returns : array-like
        Period returns (excess of the risk-free rate). NaNs are dropped.
        A one-column DataFrame is accepted; several columns are refused.
    periods_per_year : float, default 252.0
        Number of return observations per year (252 daily, 52 weekly, 12
        monthly). Used as the annualisation factor ``sqrt(periods_per_year)``.
    annualize : bool, default True
        If False, return the per-period Sharpe ratio (mean / std).

    Returns
    -------
    float
        ``mean / std(ddof=1)``, multiplied by ``sqrt(periods_per_year)`` when
        ``annualize`` is True. NaN if fewer than two observations remain. An
        all-zero series gives 0.0; any other constant series gives NaN with a
        warning.

    Notes
    -----
    Assumes IID returns (AFML 14.7.4). Serial correlation or non-stationarity
    makes the sqrt-time scaling invalid.
    """
    if annualize and not (math.isfinite(periods_per_year) and periods_per_year > 0):
        # NaN compares false with "<= 0", so it would pass and return NaN; inf would return inf.
        # Checked only when the value is used: annualize=False ignores it.
        raise ValueError("periods_per_year must be positive and finite")
    x = _as_1d(returns)
    if np.isinf(x).any():
        # Only NaN is a missing value. An infinite return is bad data, and dropping it would hide that.
        raise ValueError("returns must not contain +/-inf; NaN is dropped as missing data")
    x = x[np.isfinite(x)]
    if x.size < 2:
        # Warn so a caller does not mistake the NaN for a missing-data marker.
        warnings.warn(
            "sharpe_ratio: fewer than two finite observations, returning NaN",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    if np.all(x == 0.0):
        # An identically zero series has Sharpe ratio 0 by definition, which is a defined value.
        return 0.0
    mu = float(np.mean(x))
    # Constant input still gives a tiny nonzero np.std from float noise, so test the
    # range against a relative tolerance instead of testing the std for exact zero.
    # Relative to the magnitude of the data, so the test is scale-invariant: an absolute floor made
    # returns of scale 1e-13 look constant.
    if float(np.ptp(x)) <= 1e-12 * float(np.max(np.abs(x))):
        warnings.warn(
            "sharpe_ratio: returns are constant, so the Sharpe ratio is undefined; returning NaN",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    sd = float(np.std(x, ddof=1))
    sr = mu / sd
    return sr * math.sqrt(periods_per_year) if annualize else sr


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe ratio, AFML 14.7.2.

    .. math::

        \\widehat{PSR}[SR^*] = Z\\left[\\frac{(\\widehat{SR} - SR^*)\\sqrt{T-1}}
        {\\sqrt{1 - \\hat\\gamma_3 \\widehat{SR} + \\frac{\\hat\\gamma_4 - 1}{4}
        \\widehat{SR}^2}}\\right]

    Parameters
    ----------
    sr_hat : float
        Observed, non-annualised Sharpe ratio.
        Per period: pass ``sharpe_ratio(r, annualize=False)``. The default annualised
        value is about sqrt(252) times too large and saturates the result.
    sr_benchmark : float
        Benchmark SR* (non-annualised). 0 tests against no skill.
    n_obs : int
        Number of return observations T (must be >= 2).
    skew : float, default 0.0
        Sample skewness of the returns, gamma_3.
    kurtosis : float, default 3.0
        Sample (non-excess) kurtosis of the returns, gamma_4.

    Returns
    -------
    float
        Probability in [0, 1] that the true SR exceeds ``sr_benchmark``.

    Raises
    ------
    ValueError
        If ``n_obs < 2`` or the variance term under the square root is not
        positive (returns with extreme skew/kurtosis for the given SR).

    Notes
    -----
    With ``sr_hat == sr_benchmark`` the argument of Z is zero, so PSR = 0.5
    exactly for any skew, kurtosis and T.
    """
    if math.isnan(sr_benchmark):
        # A NaN benchmark would give a NaN result. An infinite one is a defined limit and stays.
        raise ValueError(f"sr_benchmark must not be NaN, got {sr_benchmark!r}")
    # An infinite skew or kurtosis gave exactly 0.5 ("no skill") with no warning; reject it as
    # deflated_sharpe does.
    skew = _as_finite("skew", skew)
    kurtosis = _as_finite("kurtosis", kurtosis)
    denom = _psr_denominator(sr_hat, skew, kurtosis)
    if not math.isfinite(n_obs) or n_obs != int(n_obs):
        # NaN slips past the comparison below, and an infinite sample gives a certain answer.
        raise ValueError(f"n_obs must be an integer, got {n_obs!r}")
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    z = (sr_hat - sr_benchmark) * math.sqrt(n_obs - 1) / denom
    if not (math.isfinite(sr_hat) and math.isfinite(sr_benchmark)):
        # Infinite inputs make z NaN or infinite, which would otherwise pass silently.
        warnings.warn(
            "probabilistic_sharpe_ratio: non-finite input (sr_hat or sr_benchmark), result may be NaN",
            UserWarning,
            stacklevel=2,
        )
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe ratio across N independent trials under H0.

    Implements the benchmark SR* of AFML 14.7.3:

    .. math::

        SR^* = \\sqrt{V[\\{\\hat{SR}_n\\}]}\\left[(1-\\gamma)Z^{-1}\\left[1-\\tfrac{1}{N}\\right]
        + \\gamma Z^{-1}\\left[1-\\tfrac{1}{Ne}\\right]\\right]

    Parameters
    ----------
    n_trials : int
        Number of independent trials N (integer >= 1).
    var_sr : float
        Cross-trial variance V[{SR_n}] of the (non-annualised) Sharpe ratios.
        Finite and >= 0.

    Returns
    -------
    float
        Expected maximum SR under the null of zero skill. Returns 0.0 for
        ``n_trials == 1`` (a single trial involves no selection); the book's
        formula is undefined there.

    Raises
    ------
    ValueError
        If ``n_trials`` is not an integer >= 1 (bool and non-integral values are
        rejected), or ``var_sr`` is not finite and non-negative.

    Notes
    -----
    Approximation of E[max of N IID N(0,1)] from the book (Bailey et al. 2014
    proof). It is not exact for small N. Against exact numerical integration
    (``tests/test_stats.py``, numerical reference) it is 7.9% below the exact
    maximum at N = 2, 2.5% above at N = 5, and 0.4% above at N = 1000.
    """
    n_trials = _as_int("n_trials", n_trials, minimum=1)
    var_sr = _check_var_sr(var_sr)
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    a = norm.ppf(1.0 - 1.0 / n)
    b = norm.ppf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(var_sr) * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated Sharpe ratio together with the benchmark and inputs it used.

    Returned by :func:`deflated_sharpe`. All fields are scalars.

    Attributes
    ----------
    dsr : float
        Deflated Sharpe ratio: PSR evaluated at ``sr_star``, in [0, 1].
    sr_hat : float
        Observed non-annualised Sharpe ratio of the selected trial.
    sr_star : float
        Expected maximum Sharpe ratio of ``n_trials`` zero-skill trials.
    n_trials : int
        Number of trials N.
    n_obs : int
        Number of return observations T of the selected trial.
    skew : float
        Skewness of the selected trial's returns.
    kurt : float
        (Non-excess) kurtosis of the selected trial's returns.
    """

    dsr: float
    sr_hat: float
    sr_star: float
    n_trials: int
    n_obs: int
    skew: float
    kurt: float

    def _repr_html_(self) -> str:
        """Return a notebook HTML table of the DSR, its benchmark and its inputs.

        The values are read from the object; the DSR is not recomputed.
        """
        return html_table(
            "DeflatedSharpeResult",
            [
                ("dsr", self.dsr),
                ("sr_hat", self.sr_hat),
                ("sr_star", self.sr_star),
                ("n_trials", self.n_trials),
                ("n_obs", self.n_obs),
            ],
        )


def deflated_sharpe(
    sr_hat: float,
    n_trials: int,
    var_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio with its benchmark and inputs, AFML 14.7.3.

    Computes the same value as :func:`deflated_sharpe_ratio` and also returns
    the expected-maximum benchmark ``sr_star``.

    Parameters
    ----------
    sr_hat : float
        Best observed non-annualised Sharpe ratio among the trials. Finite.
        Per period: pass ``sharpe_ratio(r, annualize=False)``. The default annualised
        value is about sqrt(252) times too large and saturates the result.
    n_trials : int
        Number of trials N that were run to select ``sr_hat``. Integer >= 1.
    var_sr : float
        Cross-trial variance of the non-annualised Sharpe ratios. Finite, >= 0.
    n_obs : int
        Number of return observations T of the selected strategy. Integer >= 2.
    skew, kurt : float
        Skewness and (non-excess) kurtosis of the selected strategy's returns.
        Finite.

    Returns
    -------
    DeflatedSharpeResult
        ``result.dsr`` is the deflated Sharpe ratio.

    Raises
    ------
    ValueError
        If an argument is outside the ranges above, or the PSR variance term is
        not positive. Bool values are rejected for the counts and the floats.

    Notes
    -----
    Integral-valued floats such as ``40.0`` are accepted for ``n_trials`` and
    ``n_obs``.
    """
    n_trials_i = _as_int("n_trials", n_trials, minimum=1)
    n_obs_i = _as_int("n_obs", n_obs, minimum=2)
    sr = _as_finite("sr_hat", sr_hat)
    v = _check_var_sr(var_sr)
    g3 = _as_finite("skew", skew)
    g4 = _as_finite("kurt", kurt)
    sr_star = float(expected_max_sharpe(n_trials_i, v))
    dsr = probabilistic_sharpe_ratio(sr, sr_star, n_obs_i, g3, g4)
    return DeflatedSharpeResult(
        dsr=dsr,
        sr_hat=sr,
        sr_star=sr_star,
        n_trials=n_trials_i,
        n_obs=n_obs_i,
        skew=g3,
        kurt=g4,
    )


def deflated_sharpe_ratio(
    sr_hat: float,
    n_trials: int,
    var_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe ratio, AFML 14.7.3.

    DSR = PSR[SR*] with SR* = :func:`expected_max_sharpe` (n_trials, var_sr).

    Parameters
    ----------
    sr_hat : float
        Best observed non-annualised Sharpe ratio among the trials.
        Per period: pass ``sharpe_ratio(r, annualize=False)``. The default annualised
        value is about sqrt(252) times too large and saturates the result.
    n_trials : int
        Number of trials N that were run to select ``sr_hat``.
    var_sr : float
        Cross-trial variance of the non-annualised Sharpe ratios.
    n_obs : int
        Number of return observations T of the selected strategy.
    skew, kurt : float
        Skewness and (non-excess) kurtosis of the selected strategy's returns.

    Returns
    -------
    float
        Probability that the selected strategy's true SR exceeds the
        expected maximum of N zero-skill trials. Values above 0.95 are a
        common significance convention (not from the book).

    Raises
    ------
    ValueError
        On the same inputs as :func:`deflated_sharpe`.
    """
    return deflated_sharpe(sr_hat, n_trials, var_sr, n_obs, skew, kurt).dsr


def min_track_record_length(
    sr_hat: float,
    sr_benchmark: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    prob: float = 0.95,
) -> float:
    """Minimum track record length (minTRL) for a PSR confidence level.

    Solves ``PSR[SR*](T) = prob`` for T:

    .. math::

        minTRL = 1 + \\left[1 - \\hat\\gamma_3 \\widehat{SR} + \\frac{\\hat\\gamma_4 - 1}{4}
        \\widehat{SR}^2\\right]\\left(\\frac{Z^{-1}[prob]}{\\widehat{SR} - SR^*}\\right)^2

    Parameters
    ----------
    sr_hat : float
        Observed non-annualised Sharpe ratio.
    sr_benchmark : float
        Benchmark SR* (non-annualised).
    skew, kurt : float
        Skewness and (non-excess) kurtosis of the returns.
    prob : float, default 0.95
        Target PSR confidence, in (0, 1). The default 0.95 is a common
        convention, not a book value.

    Returns
    -------
    float
        Real-valued minimum number of observations; round up to get an
        integer sample size.

    Raises
    ------
    ValueError
        If ``sr_hat <= sr_benchmark``, if ``prob`` < 0.5 (PSR stays above 0.5
        for every track length when ``sr_hat > sr_benchmark``, so no length solves
        it), or if ``prob`` is outside (0, 1).
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must lie in (0, 1)")
    if not math.isfinite(sr_benchmark):
        raise ValueError(f"sr_benchmark must be finite, got {sr_benchmark!r}")
    # Check skew and kurt here, so a NaN is reported as such and not as an
    # incompatible skew/kurtosis combination. This matches probabilistic_sharpe_ratio.
    skew = _as_finite("skew", skew)
    kurt = _as_finite("kurt", kurt)
    if sr_hat <= sr_benchmark:
        raise ValueError("minTRL requires sr_hat > sr_benchmark")
    if prob < 0.5:
        # With sr_hat > sr_benchmark, PSR(T) > 0.5 for every T, so no track record solves PSR = prob
        # below 0.5. At prob = 0.5 the root is T = 1 (z = 0), which is returned.
        raise ValueError("minTRL requires prob >= 0.5 when sr_hat > sr_benchmark; no track record reaches it")
    denom_sq = _psr_denominator(sr_hat, skew, kurt) ** 2
    z = norm.ppf(prob)
    return 1.0 + denom_sq * (z / (sr_hat - sr_benchmark)) ** 2


def bet_timing(positions: pd.Series) -> pd.DatetimeIndex:
    """Timestamps where a bet ends, from a series of target positions (AFML Snippet 14.1).

    A bet ends at a flattening (the position goes from non-zero to zero) and at a
    flip (the sign changes between two rows).

    Parameters
    ----------
    positions : pd.Series
        Target positions indexed by a strictly increasing DatetimeIndex. Positive is
        long, negative short, 0 flat. Finite values only.

    Returns
    -------
    pd.DatetimeIndex
        Sorted bet-end timestamps, without duplicates. The last timestamp is always in
        the result, so a position still open at the end is reported there.

    Raises
    ------
    TypeError
        If ``positions`` is not a Series, or its index is not a DatetimeIndex.
    ValueError
        If ``positions`` is empty, holds NaN or non-numeric values, or its index is not
        strictly increasing.

    Notes
    -----
    The first row has no predecessor, so it counts as non-flat: the NaN that ``shift(1)``
    leaves in front of it compares unequal to zero. A flat first row is therefore a bet end.

    The last timestamp is appended whenever it is not already a bet end, so a flat final
    row that did not flatten a bet is also reported.
    """
    values, index = _timed_values("positions", positions)
    flat_now = index[values == 0]
    # The first row is treated as non-flat, the same as the NaN the book code compares.
    was_open = index[np.r_[True, values[:-1] != 0]]
    bets = flat_now.intersection(was_open)
    # Signs, not the product: the product of two tiny opposite positions underflows to -0.0.
    signs = np.sign(values)
    flips = index[1:][signs[1:] * signs[:-1] < 0]
    bets = bets.union(flips).sort_values()
    if index[-1] not in bets:
        bets = bets.append(index[-1:])
    return bets


def holding_period(positions: pd.Series) -> float:
    """Average holding period in days, weighted by the size closed (AFML Snippet 14.2).

    Units are pooled, so the entry time of the open position is the size-weighted mean
    of the times its units were entered. A reduction or a flip closes size at the
    current time, and that size contributes (close time - entry time) to the average.

    Parameters
    ----------
    positions : pd.Series
        Target positions, as in :func:`bet_timing`.

    Returns
    -------
    float
        Size-weighted average holding period in days, measured from the first timestamp.
        NaN if no size is closed.

    Raises
    ------
    TypeError
        If ``positions`` is not a Series, or its index is not a DatetimeIndex.
    ValueError
        If ``positions`` is empty, holds NaN or non-numeric values, or its index is not
        strictly increasing.

    Notes
    -----
    Only closed size counts. A position still open at the end adds nothing to the
    average. Time is in float days, so a 36-hour holding period is 1.5.

    The loop runs in a numba kernel (``finlab._jit.jit``), with fastmath off.
    """
    values, index = _timed_values("positions", positions)
    tdiff = np.asarray((index - index[0]) / pd.Timedelta(days=1), dtype=float)
    weighted, closed = _holding_kernel(np.ascontiguousarray(values), np.ascontiguousarray(tdiff))
    if closed > 0.0:
        return float(weighted / closed)
    return float("nan")


def hhi_concentration(returns: ArrayLike) -> float:
    """Herfindahl-Hirschman concentration of a return series (AFML Snippet 14.3).

    With weights w = r / sum(r), the HHI is ``(sum(w**2) - 1/n) / (1 - 1/n)``. Equal
    returns give 0, and a single non-zero return among n gives 1.

    Parameters
    ----------
    returns : array-like
        Period returns, as a pd.Series, list or 1-D array. Finite values only.

    Returns
    -------
    float
        The HHI. NaN if there are two or fewer returns.

    Raises
    ------
    ValueError
        If ``returns`` holds NaN or infinite values, or its values sum to zero (the
        weights are then undefined).

    Notes
    -----
    The result lies in [0, 1] when all returns share a sign. Mixed signs can push the
    weights outside [0, 1], so the value can exceed 1: for returns [2, -1, 0] it is 7.
    """
    x = _finite_returns(returns)
    return _hhi(x)


@dataclass(frozen=True)
class ReturnConcentration:
    """HHI of the positive returns, the negative returns and the returns per period.

    Returned by :func:`return_concentration`.

    Attributes
    ----------
    positive : float
        HHI of the returns that are >= 0.
    negative : float
        HHI of the returns that are < 0.
    time : float
        HHI of the number of returns in each calendar period.
    """

    positive: float
    negative: float
    time: float


def return_concentration(
    returns: pd.Series, freq: str | pd.DateOffset | None = None
) -> ReturnConcentration:
    """Concentration of returns by sign and over time (AFML Snippet 14.3).

    Parameters
    ----------
    returns : pd.Series
        Period returns indexed by a DatetimeIndex. Finite values only.
    freq : str, DateOffset or None, default None
        Calendar period for the time component, passed to ``pd.Grouper``. None means
        month end (``pd.offsets.MonthEnd()``), the book's "M". The offset object is used
        because pandas 2.0 and 2.1 spell month end "M", and pandas 3 accepts only "ME".

    Returns
    -------
    ReturnConcentration
        ``positive`` is the HHI of ``returns[returns >= 0]``, ``negative`` the HHI of
        ``returns[returns < 0]``, and ``time`` the HHI of the number of returns in each
        period. Periods with no returns count as 0.

    Raises
    ------
    TypeError
        If ``returns`` is not a Series, or its index is not a DatetimeIndex.
    ValueError
        If ``returns`` holds NaN or infinite values.

    Notes
    -----
    A component is NaN when its sub-series has two or fewer values, or sums to zero. A
    positive sub-series sums to zero only when all its values are zero. Unlike
    :func:`hhi_concentration`, this function does not refuse that case, because a
    series with no gains is a normal input here.
    """
    _require_type(returns, "returns", pd.Series)
    index = as_datetime_index(returns.index)
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"returns must be indexed by a DatetimeIndex, got {type(returns.index).__name__}."
        )
    values = _finite_returns(returns)
    period = pd.offsets.MonthEnd() if freq is None else freq
    counts = pd.Series(values, index=index).groupby(pd.Grouper(freq=period)).count()
    return ReturnConcentration(
        positive=_concentration_or_nan(values[values >= 0.0]),
        negative=_concentration_or_nan(values[values < 0.0]),
        time=_concentration_or_nan(counts.to_numpy(dtype=float)),
    )


def drawdown_time_under_water(
    pnl: pd.Series, dollars: bool = False
) -> tuple[pd.Series, pd.Series]:
    """Drawdowns and time under water between high-water marks (AFML Snippet 14.4).

    A high-water mark (HWM) is a new running maximum of ``pnl``. The drawdown that
    starts at an HWM is the deepest point before ``pnl`` rises above that HWM again.
    Only HWMs followed by a drawdown are reported.

    Parameters
    ----------
    pnl : pd.Series
        Cumulative profit and loss, or an equity curve, indexed by a strictly increasing
        DatetimeIndex. Finite values only. Strictly positive when ``dollars`` is False.
    dollars : bool, default False
        If False, the drawdown is ``1 - min / hwm``, a fraction of the HWM, and ``pnl``
        must be strictly positive. If True, it is ``hwm - min`` in pnl units.

    Returns
    -------
    dd : pd.Series
        One drawdown per reported HWM, indexed by the time that HWM was set. The last
        drawdown is included even if it is still open.
    tuw : pd.Series
        Time under water in years, indexed by the time of the HWM that starts each
        interval. It has one entry fewer than ``dd``, so the last drawdown has no entry.

    Raises
    ------
    TypeError
        If ``pnl`` is not a Series, its index is not a DatetimeIndex, or ``dollars`` is not
        a bool.
    ValueError
        If ``pnl`` is empty, holds NaN or infinite values, its index is not strictly
        increasing, or it is not strictly positive when ``dollars`` is False.

    Notes
    -----
    A year is 365.25 days (a convention, not from the book). The book divides by a
    ``'Y'`` timedelta, which numpy 2.5 rejects with ``ValueError: Unit Y is not supported``.

    Time under water runs from an HWM to the next HWM that has a drawdown, not to the
    next HWM. If an HWM with no later drawdown falls in between, the interval runs past
    the true recovery. With pnl ``[0, 1, 0.5, 2, 3, 2, 4]`` the HWM at row 1 gets 3 days,
    but pnl is back above 1 at row 3, so the true time is 2 days.
    """
    if not isinstance(dollars, bool):
        raise TypeError(f"dollars must be a bool, got {type(dollars).__name__}.")
    values, index = _timed_values("pnl", pnl)
    if not dollars and not bool((values > 0.0).all()):
        raise ValueError(
            "pnl must be strictly positive when dollars is False, because the drawdown is "
            "divided by the high-water mark."
        )
    hwm = np.maximum.accumulate(values)
    # The running maximum never falls, so each HWM is one contiguous run of rows.
    starts = np.flatnonzero(np.r_[True, hwm[1:] != hwm[:-1]])
    run_min = np.minimum.reduceat(values, starts)
    run_hwm = hwm[starts]
    keep = run_hwm > run_min
    run_hwm, run_min = run_hwm[keep], run_min[keep]
    run_time = index[starts][keep]
    dd_values = run_hwm - run_min if dollars else 1.0 - run_min / run_hwm
    dd = pd.Series(dd_values, index=run_time, name="dd", dtype=float)
    gaps = (run_time[1:] - run_time[:-1]) / pd.Timedelta(days=365.25)
    tuw = pd.Series(np.asarray(gaps, dtype=float), index=run_time[:-1], name="tuw")
    return dd, tuw


@jit
def _holding_kernel(pos: np.ndarray, tdiff: np.ndarray) -> tuple[float, float]:
    """Return (sum of days held x size closed, sum of size closed) for Snippet 14.2.

    ``tdiff`` holds the timestamps in float days from the first row.
    """
    weighted = 0.0
    closed = 0.0
    t_entry = 0.0
    for i in range(1, pos.shape[0]):
        p_diff = pos[i] - pos[i - 1]
        # Same test as p_diff * pos[i-1] >= 0, written on signs: that product underflows to -0.0
        # for tiny positions, which then read as an increase.
        if pos[i - 1] == 0.0 or p_diff == 0.0 or (p_diff > 0.0) == (pos[i - 1] > 0.0):
            # Increased or unchanged: the entry time moves to the size-weighted mean.
            if pos[i] != 0.0:
                t_entry = (t_entry * pos[i - 1] + tdiff[i] * p_diff) / pos[i]
        elif pos[i] != 0.0 and (pos[i] > 0.0) != (pos[i - 1] > 0.0):
            # Flip: the whole old position closes, and the new one opens at this row.
            size = abs(pos[i - 1])
            weighted += (tdiff[i] - t_entry) * size
            closed += size
            t_entry = tdiff[i]
        else:
            size = abs(p_diff)
            weighted += (tdiff[i] - t_entry) * size
            closed += size
    return weighted, closed


def _timed_values(name: str, series: object) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return the float values and the DatetimeIndex of a time-indexed series.

    The rows must be in time order, because the bet and drawdown rules read the order of
    the rows. NaN, infinite and non-numeric values are refused.
    """
    _require_type(series, name, pd.Series)
    index = as_datetime_index(series.index)
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"{name} must be indexed by a DatetimeIndex, got {type(series.index).__name__}."
        )
    if index.hasnans:
        raise ValueError(f"{name} index must not contain NaT.")
    if len(series) == 0:
        raise ValueError(f"{name} must hold at least one observation.")
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not (
        pd.api.types.is_numeric_dtype(dtype) or dtype == object
    ):
        raise ValueError(f"{name} must hold real numbers, got dtype {dtype}.")
    _require_real_values(name, series.to_numpy())
    if bool(series.isna().any()):
        raise ValueError(f"{name} must not contain NaN; a missing value is not a position or a pnl.")
    values = np.asarray(series.to_numpy(dtype=float), dtype=float)
    if not np.isfinite(values).all():
        bad = values[~np.isfinite(values)][0]
        raise ValueError(f"{name} must be finite, got {bad!r}.")
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError(f"{name} index must be strictly increasing.")
    return values, index


def _finite_returns(returns: ArrayLike) -> np.ndarray:
    """Return ``returns`` as a 1-D float array, refusing NaN and infinite values."""
    if np.ndim(returns) == 0:
        # A scalar would reshape to one value and come back as NaN, which reads as "no answer".
        raise ValueError("returns must be a sequence of numbers, got a scalar.")
    x = _as_1d(returns)
    if np.isnan(x).any():
        raise ValueError("returns must not contain NaN; drop missing values before calling.")
    if np.isinf(x).any():
        raise ValueError("returns must be finite, got an infinite value.")
    return x


def _sums_to_zero(x: np.ndarray) -> bool:
    """True when the values sum to zero, within float precision relative to their size."""
    return bool(abs(float(np.sum(x))) <= 1e-12 * float(np.sum(np.abs(x))))


def _unit_scale(x: np.ndarray) -> np.ndarray:
    """Return ``x`` divided by its largest magnitude, so that sums of it cannot overflow."""
    # The HHI is scale-free. Without this, three returns of 1e308 sum to inf and look like a zero sum.
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x / m if m > 0.0 else x


def _hhi(x: np.ndarray) -> float:
    """HHI of a finite 1-D array; NaN for two or fewer values, refused when the sum is zero."""
    n = x.size
    if n <= 2:
        return float("nan")
    z = _unit_scale(x)
    if _sums_to_zero(z):
        raise ValueError("returns sum to zero, so the HHI weights r / sum(r) are undefined.")
    w = z / float(np.sum(z))
    h = float(np.sum(w * w))
    return (h - 1.0 / n) / (1.0 - 1.0 / n)


def _concentration_or_nan(x: np.ndarray) -> float:
    """HHI of a sub-series; NaN when it is too short or sums to zero, where HHI is undefined."""
    if x.size <= 2 or _sums_to_zero(_unit_scale(x)):
        return float("nan")
    return _hhi(x)


def _psr_denominator(sr_hat: float, skew: float, kurt: float) -> float:
    """Return sqrt(1 - skew*SR + (kurt-1)/4 * SR^2), raising if non-positive."""
    # Checked first: an infinite SR would otherwise fail below with a message about skew and kurtosis.
    if not math.isfinite(sr_hat):
        raise ValueError(f"sr_hat must be finite, got {sr_hat!r}")
    inside = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat * sr_hat
    if not inside > 0.0:
        raise ValueError(
            "non-positive PSR variance term; skew/kurtosis incompatible with sr_hat"
        )
    return math.sqrt(inside)


def _as_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum``.

    Accepts Python and NumPy integers and integral-valued floats such as 40.0.
    Rejects bool, non-integral values, NaN and infinities.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if isinstance(value, numbers.Integral):
        n = int(value)
    elif math.isfinite(value) and float(value).is_integer():
        n = int(float(value))
    else:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if n < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {n}")
    return n


def _as_finite(name: str, value: object) -> float:
    """Return ``value`` as a finite float; reject bool, non-real and non-finite values."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # A 0-d array holds one number; it is unwrapped so the real-number test sees it.
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return x


def _check_var_sr(var_sr: object) -> float:
    """Return the cross-trial variance as a finite float >= 0."""
    v = _as_finite("var_sr", var_sr)
    if v < 0.0:
        raise ValueError(f"var_sr must be non-negative, got {v}")
    return v


def _as_1d(x: ArrayLike) -> np.ndarray:
    """Coerce a return container to a 1-D float64 array."""
    # A DataFrame goes through the same column rule as an ndarray: several columns are refused.
    raw = np.asarray(x)
    _require_real_values("returns", raw)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim >= 2 and int(np.prod(arr.shape[1:])) > 1:
        # Raveling a multi-column array would pool every column into one series; refuse instead.
        # A single row with several columns counts too, so only the column count is tested.
        raise ValueError(f"returns must be one-dimensional or one column, got shape {arr.shape}.")
    return arr.ravel()
