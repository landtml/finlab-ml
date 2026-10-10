"""Entropy estimators and encoding schemes (AFML ch. 18).

Implements, following the book's notation:

* :func:`plug_in_entropy` -- the plug-in (maximum-likelihood) entropy rate
  ``H_{n,w} = -(1/w) sum_y p_w(y) log2 p_w(y)`` over overlapping words of
  length ``w`` (AFML 18.3, Snippet 18.1).
* :func:`lempel_ziv_parse` -- the Lempel-Ziv non-redundant dictionary
  (AFML 18.4, Snippet 18.2).
* :func:`lempel_ziv_entropy` -- an entropy estimate from the LZ dictionary size
  (see the Notes: the book gives no formula, so this is the standard LZ78
  normalisation, not an AFML formula).
* :func:`kontoyiannis_entropy` -- the Kontoyiannis / Gao et al. (2008) LZ
  estimator with the expanding or sliding centred window (AFML 18.4,
  Snippet 18.4).
* :func:`encode_binary`, :func:`encode_quantile`, :func:`encode_sigma` --
  discretisations of returns into symbols (AFML 18.5.1-18.5.3).

Conventions
-----------
* Messages may be ``str``, or any 1-D sequence / array / Series of hashable
  values. Symbols are first mapped to dense codes ``0..K-1`` (``np.unique``),
  so only equality between symbols matters.
* Entropies are in bits (``log2``) and are per symbol.
* Entropy kernels are numba-compiled; the Python wrappers only prepare input.

Literal readings and deviations
-------------------------------
* Plug-in (Snippet 18.1): the snippet's ``pmf1`` iterates ``i in range(w, n)``
  and so omits the final word ``msg[n-w:n]``. We use all ``n - w + 1`` windows,
  normalised by ``n - w + 1``.
* Kontoyiannis (Snippet 18.4): the snippet's ``log2(n + 1) / L`` terms are kept
  as written, with ``n = i`` (expanding) or ``n = window`` (sliding). The
  snippet's ``redundancy`` normalises by ``log2(len(msg))``, which is not the
  alphabet-size normalisation, so we return only the entropy estimate ``h``.
  For an expanding window the book asks for an even length. An odd-length
  message simply leaves the last symbol unparsed.
* Match length (Snippet 18.3): the snippet tests every ``l``. Matching is
  monotone in ``l`` (a match of length ``l+1`` implies one of length ``l``), so
  our kernel stops at the first failure. The result is the same, and a test
  checks it against the literal loop.
* Lempel-Ziv: the book describes the estimate only qualitatively ("the number
  of items in a Lempel-Ziv dictionary relative to the length of the message").
  :func:`lempel_ziv_entropy` uses ``c log2(c) / n`` with ``c`` the number of
  phrases. This is the standard LZ78 normalisation from the literature, not a
  formula given in AFML, and it is labelled as such in the proofs document.
* Encoding: the sigma-encoding counts ``ceil((max - min) / sigma)`` codes, as
  the book says. When ``max`` lands exactly on a bin edge the top bin is
  clipped to the last code so that the alphabet size is preserved.
* No "message round-trip" helpers are provided, because the book does not
  describe any.

Scope
-----
Not covered: the mutual information and the normalised variation of
information (18.2), the entropy of a Gaussian process (18.6), the
generalised-mean interpretation (18.7), the Gao et al. (2008) bias/variance
window rule as code (it is described in the text), and the Gaussian entropy-
implied volatility. Fractionally differentiated encodings are the user's job.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ._jit import jit

__all__ = [
    "plug_in_entropy",
    "lempel_ziv_parse",
    "lempel_ziv_entropy",
    "kontoyiannis_entropy",
    "encode_binary",
    "encode_quantile",
    "encode_sigma",
]

_MAX_WORD_CODE = 2**62


def _as_codes(message: Any) -> tuple[np.ndarray, int]:
    """Map a message to dense int64 codes in ``0..K-1``. Returns ``(codes, K)``."""
    if isinstance(message, str):
        arr = np.array(list(message))
    elif isinstance(message, (pd.Series, pd.Index)):
        arr = np.asarray(message.to_numpy())
    else:
        arr = np.asarray(message)
    if arr.ndim != 1:
        raise ValueError(f"message must be 1-D, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("message is empty.")
    _, inv = np.unique(arr, return_inverse=True)
    inv = np.asarray(inv, dtype=np.int64).reshape(-1)
    return np.ascontiguousarray(inv), int(inv.max()) + 1


# ---------------------------------------------------------------------------
# Plug-in estimator (AFML 18.3)
# ---------------------------------------------------------------------------


@jit
def _plug_in_kernel(codes: np.ndarray, k: int, w: int) -> float:
    n = codes.shape[0]
    m = n - w + 1
    words = np.empty(m, dtype=np.int64)
    for i in range(m):
        code = 0
        for j in range(w):
            code = code * k + codes[i + j]
        words[i] = code
    words.sort()
    h = 0.0
    i = 0
    while i < m:
        j = i
        while j < m and words[j] == words[i]:
            j += 1
        p = (j - i) / m
        h -= p * math.log2(p)
        i = j
    return h / w


def plug_in_entropy(message: Any, word_length: int = 1) -> float:
    """Plug-in (maximum-likelihood) entropy rate in bits per symbol (AFML 18.3).

    Parameters
    ----------
    message : str or sequence
        The symbol stream ``x_1^n``.
    word_length : int, default 1
        Word length ``w``. Frequencies are taken over all ``n - w + 1``
        overlapping words.

    Returns
    -------
    float
        ``-(1/w) sum p(y) log2 p(y)``. For ``w = 1`` this is Shannon's entropy of
        the empirical symbol distribution. A constant message gives 0. A uniform
        binary i.i.d. message gives about 1.

    Raises
    ------
    ValueError
        If ``word_length < 1``, the message is shorter than ``word_length``, or
        the word code would overflow 64 bits (``K ** w >= 2**62``).
    """
    codes, k = _as_codes(message)
    if word_length < 1:
        raise ValueError("word_length must be >= 1.")
    if codes.shape[0] < word_length:
        raise ValueError("message is shorter than word_length.")
    if k**word_length >= _MAX_WORD_CODE:
        raise ValueError(
            f"{k} symbols with word_length={word_length} overflow the word code; "
            "use a smaller word length or a smaller alphabet."
        )
    return float(_plug_in_kernel(codes, k, int(word_length)))


# ---------------------------------------------------------------------------
# Lempel-Ziv dictionary (AFML 18.4, Snippet 18.2)
# ---------------------------------------------------------------------------


@jit
def _lz_parse_kernel(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = codes.shape[0]
    starts = np.empty(n + 1, dtype=np.int64)
    lens = np.empty(n + 1, dtype=np.int64)
    count = 1
    starts[0] = 0
    lens[0] = 1
    i = 1
    while i < n:
        j = i
        while j < n:
            length = j - i + 1
            in_lib = False
            for d in range(count):
                if lens[d] == length:
                    same = True
                    for t in range(length):
                        if codes[starts[d] + t] != codes[i + t]:
                            same = False
                            break
                    if same:
                        in_lib = True
                        break
            if not in_lib:
                starts[count] = i
                lens[count] = length
                count += 1
                break
            j += 1
        i = j + 1
    return starts[:count].copy(), lens[:count].copy()


def lempel_ziv_parse(message: Any) -> list[tuple[Any, ...]]:
    """Non-redundant LZ dictionary of a message (AFML Snippet 18.2).

    The message is parsed left to right. Each new phrase is the shortest
    substring starting at the current position that is not already in the
    dictionary. The first symbol is always entered as the initial phrase, and a
    trailing run that is already in the dictionary is not added.

    Parameters
    ----------
    message : str or sequence

    Returns
    -------
    list of tuple
        Phrases in order of appearance, each a tuple of the original symbols.
    """
    codes, _ = _as_codes(message)
    starts, lens = _lz_parse_kernel(codes)
    values = list(message) if isinstance(message, str) else list(np.asarray(message).tolist())
    return [tuple(values[int(s) : int(s) + int(length)]) for s, length in zip(starts, lens)]


def lempel_ziv_entropy(message: Any) -> float:
    """Entropy estimate from the LZ dictionary size, in bits per symbol.

    Computes ``c * log2(c) / n`` with ``c`` the number of phrases returned by
    :func:`lempel_ziv_parse` and ``n`` the message length.

    Notes
    -----
    AFML 18.4 describes the LZ rate qualitatively (dictionary size relative to
    message length) and gives no formula. This normalisation is the standard
    LZ78 one (Ziv and Lempel, 1978), and it is labelled as external to the book
    in ``docs/proofs/entropy.md``. The estimate is biased for short messages.
    """
    codes, _ = _as_codes(message)
    starts, _ = _lz_parse_kernel(codes)
    c = starts.shape[0]
    n = codes.shape[0]
    return float(c * math.log2(c) / n) if c > 1 else 0.0


# ---------------------------------------------------------------------------
# Kontoyiannis / Gao et al. (2008) estimator (AFML 18.4, Snippet 18.4)
# ---------------------------------------------------------------------------


@jit
def _match_length(codes: np.ndarray, i: int, n: int) -> int:
    """Length of the longest match (length of msg[i:i+m] inside msg[i-n:i]) plus one.

    Snippet 18.3 returns ``len(subS) + 1`` with ``subS`` the longest matched
    substring. Matching is monotone in the length, so the loop stops at the
    first failure.
    """
    total = codes.shape[0]
    best = 0
    for length in range(1, n + 1):
        if i + length > total:
            break
        found = False
        for j in range(i - n, i):
            same = True
            for t in range(length):
                if codes[j + t] != codes[i + t]:
                    same = False
                    break
            if same:
                found = True
                break
        if not found:
            break
        best = length
    return best + 1


@jit
def _konto_sum(codes: np.ndarray, first: int, last: int, window: int) -> tuple[float, int]:
    total = 0.0
    num = 0
    for i in range(first, last + 1):
        n = window if window > 0 else i
        length = _match_length(codes, i, n)
        total += math.log2(n + 1) / length
        num += 1
    return total, num


def kontoyiannis_entropy(message: Any, window: int | None = None) -> float:
    """Kontoyiannis LZ entropy estimate, bits per symbol (AFML 18.4, Snippet 18.4).

    Parameters
    ----------
    message : str or sequence
        The symbol stream.
    window : int, optional
        Sliding centred window ``n`` (capped at ``len(message) // 2``). If None,
        the expanding window is used (``n = i`` at each centre ``i``).

    Returns
    -------
    float
        ``h = mean_i [ log2(n + 1) / L_i ]``, where ``L_i`` is the match length
        (plus one) of :func:`_match_length`. Larger values mean less redundancy.

    Notes
    -----
    The snippet's ``(n + 1)`` in the numerator is kept as written. The
    centre positions are ``1..len//2`` for the expanding window, and
    ``window..len-window`` for the sliding window. The estimator is asymptotic;
    see the book for the bias-variance window rule ``N ~ n + (log2 n)^2``.
    """
    codes, _ = _as_codes(message)
    n_total = codes.shape[0]
    if window is None:
        first, last, win = 1, n_total // 2, 0
    else:
        if window < 1:
            raise ValueError("window must be >= 1.")
        win = min(int(window), n_total // 2)
        if win < 1:
            raise ValueError("message is too short for the requested window.")
        first, last = win, n_total - win
    if last < first:
        raise ValueError("message is too short for the Kontoyiannis estimator.")
    total, num = _konto_sum(codes, first, last, win)
    return float(total / num)


# ---------------------------------------------------------------------------
# Encoding schemes (AFML 18.5)
# ---------------------------------------------------------------------------


def _returns_array(returns: Any) -> tuple[np.ndarray, pd.Index | None]:
    index = returns.index if isinstance(returns, (pd.Series, pd.DataFrame)) else None
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contain NaN or infinite values.")
    return arr, index


def _maybe_series(values: np.ndarray, index: pd.Index | None) -> Any:
    return values if index is None else pd.Series(values, index=index)


def encode_binary(returns: Any) -> Any:
    """Sign encoding of returns (AFML 18.5.1).

    Returns 1 where ``r_t > 0`` and 0 where ``r_t < 0``. Observations with
    ``r_t == 0`` are removed, as the book prescribes.

    Parameters
    ----------
    returns : array-like or pd.Series

    Returns
    -------
    np.ndarray of int64 or pd.Series
        Codes for the retained observations (a Series keeps their index).
    """
    arr, index = _returns_array(returns)
    keep = arr != 0.0
    codes = (arr[keep] > 0.0).astype(np.int64)
    kept_index = None if index is None else index[keep]
    return _maybe_series(codes, kept_index)


def encode_quantile(returns: Any, n_bins: int, reference: Any = None) -> Any:
    """Quantile encoding of returns (AFML 18.5.2).

    Each return is assigned the index of the quantile bin it falls into. The
    bin edges are the ``1/n_bins, ..., (n_bins-1)/n_bins`` quantiles of
    ``reference``.

    Parameters
    ----------
    returns : array-like or pd.Series
        Series to encode.
    n_bins : int
        Alphabet size ``>= 2``. Codes are ``0..n_bins-1``.
    reference : array-like, optional
        In-sample series that defines the bin edges (the book's training set).
        Defaults to ``returns``.

    Returns
    -------
    np.ndarray of int64 or pd.Series
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    arr, index = _returns_array(returns)
    ref = arr if reference is None else _returns_array(reference)[0]
    if ref.shape[0] == 0:
        raise ValueError("reference is empty.")
    edges = np.quantile(ref, np.arange(1, n_bins) / n_bins)
    codes = np.searchsorted(edges, arr, side="right").astype(np.int64)
    return _maybe_series(codes, index)


def encode_sigma(returns: Any, sigma: float) -> Any:
    """Sigma encoding of returns (AFML 18.5.3).

    Code ``k`` covers ``[min + k*sigma, min + (k+1)*sigma)``, with
    ``ceil((max - min) / sigma)`` codes in total.

    Parameters
    ----------
    returns : array-like or pd.Series
    sigma : float
        Discretisation step ``> 0`` (the book's ``sigma``). It is a step size, not
        a number of bins.

    Returns
    -------
    np.ndarray of int64 or pd.Series
        Codes ``0..K-1`` with ``K = max(1, ceil((max - min) / sigma))``.
    """
    if not sigma > 0:
        raise ValueError("sigma must be > 0.")
    arr, index = _returns_array(returns)
    lo = float(arr.min())
    k_codes = max(1, math.ceil((float(arr.max()) - lo) / sigma))
    codes = np.floor((arr - lo) / sigma).astype(np.int64)
    codes = np.clip(codes, 0, k_codes - 1)
    return _maybe_series(codes, index)
