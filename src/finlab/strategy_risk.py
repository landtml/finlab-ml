"""Strategy risk: binomial Sharpe ratio, implied precision and frequency, failure probability.

Implements AFML chapter 15 (Snippets 15.1 to 15.5). A strategy makes ``freq`` bets a year.
Each bet wins ``pt`` (> 0) with probability ``p`` and loses ``sl`` (< 0) otherwise. The bets
are IID.

* :func:`binomial_sharpe` -- annualised Sharpe ratio of the bet sequence, eq. 15.1 and 15.3.
* :func:`implied_precision` -- hit rate that delivers a target Sharpe, Snippet 15.3 (binHR).
* :func:`implied_frequency` -- bets per year that deliver a target Sharpe, Snippet 15.4 (binFreq).
* :func:`probability_of_failure` -- probability that a strategy falls short of a target Sharpe,
  Snippet 15.5 (probFailure).
* :func:`mixture_returns` -- sample from a mixture of two Gaussians, Snippet 15.5 (mixGaussians).

Scope
-----
The bet model has two outcomes and IID bets. :func:`probability_of_failure` takes the hit rate
and the two payoff means from a return sample; no other estimation is done here.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.stats import norm

from finlab._checks import _require_real_values

__all__ = [
    "binomial_sharpe",
    "implied_precision",
    "implied_frequency",
    "probability_of_failure",
    "mixture_returns",
]

# Smallest positive normal float64. Results below it have lost precision to subnormal rounding.
_TINY = float(np.finfo(np.float64).tiny)


def _as_finite(name: str, value: object) -> float:
    """Return ``value`` as a finite float; reject bool, non-real and non-finite values."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return x


def _as_count(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an integer >= ``minimum``; reject bool, floats and other types."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    n = int(value)
    if n < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {n}")
    return n


def _check_prob(p: object) -> float:
    value = _as_finite("p", p)
    if not 0.0 < value < 1.0:
        raise ValueError(f"p must be strictly between 0 and 1, got {value}")
    return value


def _check_freq(freq: object) -> float:
    value = _as_finite("freq", freq)
    if value <= 0.0:
        raise ValueError(f"freq must be positive, got {value}")
    return value


def _check_payoffs(pt: object, sl: object) -> tuple[float, float]:
    profit = _as_finite("pt", pt)
    stop = _as_finite("sl", sl)
    if profit <= 0.0:
        raise ValueError(f"pt must be positive (a winning bet's profit), got {profit}")
    if stop >= 0.0:
        raise ValueError(f"sl must be negative (a losing bet's return), got {stop}")
    return profit, stop


def _unit_payoffs(pt: float, sl: float) -> tuple[float, float]:
    """Divide both payoffs by the larger magnitude, so the outcome width lies in (1, 2].

    The Sharpe ratio, the hit rate and the frequency depend on the payoffs only through their
    ratio. Scaling first keeps the arithmetic away from float64 overflow and underflow.
    """
    scale = max(pt, -sl)
    return pt / scale, sl / scale


def _as_returns(returns: object) -> np.ndarray:
    """Return a 1-D float64 array of finite returns with both signs present."""
    raw = np.asarray(returns)
    _require_real_values("returns", raw)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim > 1 and int(np.prod(arr.shape[1:])) > 1:
        # Raveling several columns would pool them into one series.
        raise ValueError(f"returns must be one-dimensional or one column, got shape {arr.shape}")
    arr = arr.ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must be finite; NaN and inf are not allowed")
    if not np.any(arr > 0.0):
        raise ValueError("returns must contain at least one positive value")
    if not np.any(arr <= 0.0):
        raise ValueError("returns must contain at least one value <= 0")
    return arr


def binomial_sharpe(p: float, freq: float, pt: float = 1.0, sl: float = -1.0) -> float:
    """Annualised Sharpe ratio of IID two-outcome bets.

    Parameters
    ----------
    p : float
        Probability that a bet wins, strictly between 0 and 1.
    freq : float
        Bets per year, strictly positive. Need not be an integer.
    pt : float, default 1.0
        Profit of a winning bet, positive.
    sl : float, default -1.0
        Return of a losing bet, negative.

    Returns
    -------
    float
        ``theta = ((pt - sl) * p + sl) / ((pt - sl) * sqrt(p * (1 - p))) * sqrt(freq)``.
        The sign follows the mean bet return.

    Notes
    -----
    A bet returns ``pt`` with probability ``p`` and ``sl`` otherwise. Its mean is
    ``(pt - sl) * p + sl`` and its standard deviation is ``(pt - sl) * sqrt(p * (1 - p))``.
    ``freq`` IID bets a year give an annual Sharpe ratio of ``sqrt(freq)`` times the per-bet
    ratio (AFML eq. 15.3). With ``pt = 1`` and ``sl = -1`` this reduces to eq. 15.1,
    ``(2p - 1) * sqrt(freq) / (2 * sqrt(p * (1 - p)))``.
    """
    p = _check_prob(p)
    freq = _check_freq(freq)
    pt, sl = _unit_payoffs(*_check_payoffs(pt, sl))
    width = pt - sl
    mean = width * p + sl
    std = width * math.sqrt(p * (1.0 - p))
    return float(mean / std * math.sqrt(freq))


def implied_precision(sl: float, pt: float, freq: float, target_sharpe: float) -> float:
    """Hit rate at which bets deliver a target annualised Sharpe ratio.

    Parameters
    ----------
    sl : float
        Return of a losing bet, negative.
    pt : float
        Profit of a winning bet, positive.
    freq : float
        Bets per year, strictly positive.
    target_sharpe : float
        Target annualised Sharpe ratio. Any finite real value.

    Returns
    -------
    float
        The unique ``p`` in (0, 1) with ``binomial_sharpe(p, freq, pt, sl) == target_sharpe``.

    Raises
    ------
    ValueError
        If the square of ``target_sharpe`` overflows float64, or if the hit rate is not a normal
        float strictly between 0 and 1. The second happens when the hit rate rounds to 0 or 1, or
        falls below the normal float range.

    Notes
    -----
    Squaring ``binomial_sharpe(p, freq, pt, sl) = target_sharpe`` gives the quadratic
    ``a p^2 + b p + c = 0`` with ``d = pt - sl``, ``a = (freq + target^2) d^2``,
    ``b = (2 freq sl - target^2 d) d`` and ``c = freq sl^2`` (AFML Snippet 15.3).
    Its discriminant is ``target^2 d^2 (target^2 d^2 - 4 freq sl pt)``, which is at least zero
    for ``sl < 0 < pt``, so a real root always exists. The mean bet return ``d p + sl`` is
    strictly increasing in ``p``, and ``binomial_sharpe`` is strictly increasing on (0, 1), with
    limits -inf and +inf. Exactly one ``p`` in (0, 1) therefore solves the equation.

    The quadratic also has the root of ``-target_sharpe``. For ``target_sharpe >= 0`` the larger
    root is returned, which is the formula of Snippet 15.3. For ``target_sharpe < 0`` the smaller
    root is returned, computed as ``2 c / (-b + sqrt(disc))`` to avoid cancellation.

    See ``docs/proofs/strategy_risk.md``, section 2.
    """
    pt, sl = _unit_payoffs(*_check_payoffs(pt, sl))
    freq = _check_freq(freq)
    target = _as_finite("target_sharpe", target_sharpe)
    width = pt - sl
    a = (freq + target * target) * width * width
    c = freq * sl * sl
    # The discriminant factors as t2 * (t2 + 4 freq |sl| pt) with t2 = (target * width)**2, so the
    # root is written as a sum of positive terms. Taking b^2 - 4ac directly would cancel when
    # target is 0, where the two roots coincide at -sl / width.
    t2 = (target * width) * (target * width)
    minus_b = t2 + 2.0 * freq * (-sl) * width
    root = abs(target) * width * math.sqrt(t2 + 4.0 * freq * (-sl) * pt)
    if not all(math.isfinite(v) for v in (a, t2, minus_b, root)):
        raise ValueError("target_sharpe is too large in magnitude for float64 arithmetic")
    if target >= 0.0:
        p = float((minus_b + root) / (2.0 * a))
    else:
        # A zero denominator needs sl to underflow to 0 and target to underflow too, which puts
        # the true hit rate at 0. The range check below then refuses it.
        denom = minus_b + root
        p = float(2.0 * c / denom) if denom > 0.0 else 0.0
    if not (_TINY <= p < 1.0):  # also refuses nan
        raise ValueError(
            "target_sharpe gives a hit rate that is not a normal float strictly between 0 and 1"
        )
    return p


def implied_frequency(sl: float, pt: float, p: float, target_sharpe: float) -> float:
    """Bets per year at which a hit rate delivers a target annualised Sharpe ratio.

    Parameters
    ----------
    sl : float
        Return of a losing bet, negative.
    pt : float
        Profit of a winning bet, positive.
    p : float
        Hit rate, strictly between 0 and 1.
    target_sharpe : float
        Target annualised Sharpe ratio. Any finite real value.

    Returns
    -------
    float
        ``freq = (target_sharpe * (pt - sl))**2 * p * (1 - p) / ((pt - sl) * p + sl)**2``
        (AFML Snippet 15.4), or nan when no positive frequency gives the target.

    Raises
    ------
    ValueError
        If the frequency overflows float64, or falls below the normal float range. Either
        happens when ``target_sharpe`` is extreme in magnitude.

    Notes
    -----
    Squaring ``binomial_sharpe = target_sharpe`` gives this ``freq``. The squared equation also
    holds when the mean bet return has the opposite sign to the target. The result is kept only
    when ``binomial_sharpe(p, freq, pt, sl)`` equals ``target_sharpe`` within a tight tolerance.
    Otherwise the function returns nan. nan is also returned when the target is 0 (a zero Sharpe
    needs no bets) or the mean bet return is exactly 0.

    See ``docs/proofs/strategy_risk.md``, section 3.
    """
    pt, sl = _unit_payoffs(*_check_payoffs(pt, sl))
    p = _check_prob(p)
    target = _as_finite("target_sharpe", target_sharpe)
    width = pt - sl
    mean = width * p + sl
    if target == 0.0 or mean == 0.0:
        return float("nan")
    # Squared as ratio * ratio: mean * mean would underflow for a mean near zero.
    ratio = abs(target * width) * math.sqrt(p * (1.0 - p)) / abs(mean)
    freq = ratio * ratio
    if not math.isfinite(freq):
        raise ValueError("target_sharpe is too large in magnitude for float64 arithmetic")
    if not freq >= _TINY:  # also refuses 0 from underflow
        raise ValueError(
            "target_sharpe gives a frequency below the normal float range; no float64 frequency "
            "reaches it"
        )
    theta = binomial_sharpe(p, freq, pt, sl)
    if not math.isclose(theta, target, rel_tol=1e-9, abs_tol=1e-12):
        return float("nan")
    return float(freq)


def probability_of_failure(returns: object, freq: float, target_sharpe: float) -> float:
    """Probability that a strategy's return sample falls short of a target Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Period returns, one-dimensional or one column. Finite, with at least one value above
        zero and at least one value at or below zero.
    freq : float
        Bets per year, strictly positive.
    target_sharpe : float
        Target annualised Sharpe ratio. Any finite real value.

    Returns
    -------
    float
        ``norm.cdf(thresP, loc=p, scale=p * (1 - p))``, where ``p`` is the share of returns above
        zero, ``rPos`` the mean of those, ``rNeg`` the mean of the returns at or below zero, and
        ``thresP = implied_precision(rNeg, rPos, freq, target_sharpe)`` (AFML Snippet 15.5).

    Raises
    ------
    ValueError
        If the non-positive returns are all zero, so ``rNeg`` is 0 and no bet loss exists.

    Notes
    -----
    The scale argument is ``p * (1 - p)``, the Bernoulli variance, passed where
    ``norm.cdf`` expects a standard deviation. This follows Snippet 15.5 and is not changed here.
    The standard error of the sample share ``p`` is ``sqrt(p * (1 - p) / n)`` for ``n`` returns,
    which is smaller. See ``docs/proofs/strategy_risk.md``, section 4.
    """
    arr = _as_returns(returns)
    freq = _check_freq(freq)
    target = _as_finite("target_sharpe", target_sharpe)
    positive = arr > 0.0
    # Averages of the scaled returns: summing values near 1e308 overflows float64.
    scale = float(np.max(np.abs(arr)))
    unit = arr / scale
    r_pos = float(unit[positive].mean()) * scale
    r_neg = float(unit[~positive].mean()) * scale
    p = float(positive.mean())
    if r_neg == 0.0:
        raise ValueError("returns: every non-positive return is zero, so the mean loss is 0")
    thresh = implied_precision(r_neg, r_pos, freq, target)
    return float(norm.cdf(thresh, loc=p, scale=p * (1.0 - p)))


def mixture_returns(
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    prob1: float,
    n_obs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample from a mixture of two Gaussians (AFML Snippet 15.5, mixGaussians).

    Parameters
    ----------
    mu1, mu2 : float
        Means of the two components. Finite.
    sigma1, sigma2 : float
        Standard deviations of the two components. Finite and non-negative.
    prob1 : float
        Share of observations drawn from component 1, between 0 and 1 inclusive.
    n_obs : int
        Number of observations, at least 1.
    seed : int or None, default None
        Passed to ``numpy.random.default_rng``.

    Returns
    -------
    numpy.ndarray
        Length ``n_obs`` array of float64. Component 1 contributes ``round(n_obs * prob1)``
        values, component 2 the rest. The values are shuffled.

    Notes
    -----
    The count is rounded, not truncated. ``2600 * 0.7`` is slightly below 1820 in floating
    point, and truncation would give 1819. See ``docs/proofs/strategy_risk.md``, section 5.
    """
    mu1 = _as_finite("mu1", mu1)
    mu2 = _as_finite("mu2", mu2)
    sigma1 = _as_finite("sigma1", sigma1)
    sigma2 = _as_finite("sigma2", sigma2)
    if sigma1 < 0.0 or sigma2 < 0.0:
        raise ValueError(f"sigma1 and sigma2 must be non-negative, got {sigma1} and {sigma2}")
    prob1 = _as_finite("prob1", prob1)
    if not 0.0 <= prob1 <= 1.0:
        raise ValueError(f"prob1 must be between 0 and 1, got {prob1}")
    n_obs = _as_count("n_obs", n_obs, minimum=1)

    if seed is not None:
        seed = _as_count("seed", seed, minimum=0)
    rng = np.random.default_rng(seed)
    n1 = int(round(n_obs * prob1))
    first = rng.normal(mu1, sigma1, size=n1)
    second = rng.normal(mu2, sigma2, size=n_obs - n1)
    out = np.concatenate([first, second])
    if not np.all(np.isfinite(out)):
        raise ValueError("sigma1, sigma2, mu1 or mu2 is too large: some draws overflow float64")
    rng.shuffle(out)
    return out
