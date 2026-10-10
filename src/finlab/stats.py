"""Backtest statistics: Sharpe ratio, PSR, expected maximum SR, DSR, minTRL.

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
Scalar statistics only; no multiple-testing corrections beyond DSR, and no
estimation of skewness/kurtosis (callers compute them, e.g. with
``scipy.stats.skew`` and ``scipy.stats.kurtosis(fisher=False)``).

Not covered: AFML 14.7.4 information ratio, implied precision and bet timing
(14.4-14.6), and the variance estimation of the trial Sharpe ratios, which is
left to the caller (``var_sr`` is an input).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "deflated_sharpe",
    "DeflatedSharpeResult",
    "min_track_record_length",
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
        A DataFrame is reduced to its first column.
    periods_per_year : float, default 252.0
        Number of return observations per year (252 daily, 52 weekly, 12
        monthly). Used as the annualisation factor ``sqrt(periods_per_year)``.
    annualize : bool, default True
        If False, return the per-period Sharpe ratio (mean / std).

    Returns
    -------
    float
        ``mean / std(ddof=1)``, multiplied by ``sqrt(periods_per_year)`` when
        ``annualize`` is True. NaN if fewer than two observations remain.

    Notes
    -----
    Assumes IID returns (AFML 14.7.4). Serial correlation or non-stationarity
    makes the sqrt-time scaling invalid.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    x = _as_1d(returns)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    mu = float(np.mean(x))
    if sd == 0.0:
        sr = 0.0 if mu == 0.0 else math.copysign(math.inf, mu)
    else:
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
    denom = _psr_denominator(sr_hat, skew, kurtosis)
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    z = (sr_hat - sr_benchmark) * math.sqrt(n_obs - 1) / denom
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
        If ``sr_hat <= sr_benchmark`` (no finite track record reaches
        ``prob`` > 0.5), or ``prob`` is outside (0, 1).
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must lie in (0, 1)")
    if sr_hat <= sr_benchmark:
        raise ValueError("minTRL requires sr_hat > sr_benchmark")
    denom_sq = _psr_denominator(sr_hat, skew, kurt) ** 2
    z = norm.ppf(prob)
    return 1.0 + denom_sq * (z / (sr_hat - sr_benchmark)) ** 2


def _psr_denominator(sr_hat: float, skew: float, kurt: float) -> float:
    """Return sqrt(1 - skew*SR + (kurt-1)/4 * SR^2), raising if non-positive."""
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
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    arr = np.asarray(x, dtype=np.float64)
    return arr.ravel()
