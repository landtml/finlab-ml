"""Tests for finlab.entropy (AFML ch. 18).

Fast kernels are cross-checked against literal transcriptions of Snippets 18.1-18.4
on seeded random inputs. The book's worked numbers from 18.4 are checked as well.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from finlab.entropy import (
    _match_length,
    encode_binary,
    encode_quantile,
    encode_sigma,
    kontoyiannis_entropy,
    lempel_ziv_entropy,
    lempel_ziv_parse,
    plug_in_entropy,
)

# ---------------------------------------------------------------------------
# Literal transcriptions of the book's snippets (reference implementations)
# ---------------------------------------------------------------------------


def naive_plug_in(msg, w):
    """Snippet 18.1 (pmf1 + plugIn), with all n-w+1 windows."""
    msg = "".join(map(str, msg))
    lib = Counter(msg[i : i + w] for i in range(len(msg) - w + 1))
    total = len(msg) - w + 1
    return -sum((c / total) * math.log2(c / total) for c in lib.values()) / w


def naive_lz_lib(msg):
    """Snippet 18.2, literal (msg is a sequence of hashable symbols)."""
    i, lib = 1, [tuple(msg[:1])]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = tuple(msg[i : j + 1])
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return lib


def naive_match_length(msg, i, n):
    """Snippet 18.3, literal (no early exit)."""
    sub = ""
    for l in range(n):
        msg1 = msg[i : i + l + 1]
        for j in range(i - n, i):
            msg0 = msg[j : j + l + 1]
            if msg1 == msg0:
                sub = msg1
                break
    return len(sub) + 1


def naive_konto(msg, window=None):
    """Snippet 18.4 (konto), literal."""
    total, num = 0.0, 0
    msg = "".join(map(str, msg))
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            length = naive_match_length(msg, i, i)
            total += math.log2(i + 1) / length
        else:
            length = naive_match_length(msg, i, window)
            total += math.log2(window + 1) / length
        num += 1
    return total / num


# ---------------------------------------------------------------------------
# Plug-in entropy (AFML 18.3)
# ---------------------------------------------------------------------------


def test_constant_string_has_zero_entropy():
    assert plug_in_entropy("aaaaaaaaaa") == 0.0
    assert plug_in_entropy([7] * 50, word_length=3) == 0.0


def test_uniform_binary_sequence_is_one_bit():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=200_000)
    assert plug_in_entropy(bits) == pytest.approx(1.0, abs=0.01)
    assert plug_in_entropy("01" * 500) == pytest.approx(1.0 / 1, abs=1e-3)


def test_uniform_four_symbols_is_two_bits():
    seq = np.tile(np.arange(4), 2500)
    assert plug_in_entropy(seq) == pytest.approx(2.0)


@pytest.mark.parametrize("w", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_plug_in_matches_naive(w, seed):
    rng = np.random.default_rng(seed)
    msg = rng.integers(0, 4, size=300)
    assert plug_in_entropy(msg, word_length=w) == pytest.approx(naive_plug_in(msg, w), rel=1e-12)


def test_plug_in_string_and_pandas_agree():
    s = "abracadabra" * 7
    ser = pd.Series(list(s))
    assert plug_in_entropy(s, 2) == pytest.approx(plug_in_entropy(ser, 2))
    assert plug_in_entropy(s, 2) == pytest.approx(naive_plug_in(s, 2), rel=1e-12)


def test_plug_in_errors():
    with pytest.raises(ValueError):
        plug_in_entropy("abc", word_length=0)
    with pytest.raises(ValueError):
        plug_in_entropy("ab", word_length=5)
    with pytest.raises(ValueError):
        plug_in_entropy(np.arange(200) % 100, word_length=20)


# ---------------------------------------------------------------------------
# Lempel-Ziv dictionary (AFML 18.4, Snippet 18.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lz_parse_matches_snippet(seed):
    rng = np.random.default_rng(seed)
    msg = rng.integers(0, 3, size=200)
    got = lempel_ziv_parse(msg)
    ref = naive_lz_lib(list(msg))
    assert [tuple(p) for p in got] == [tuple(int(v) for v in p) for p in ref]


def test_lz_parse_known_phrases():
    # Hand trace of Snippet 18.2: a | b | r | ac | ad | ab | ra
    phrases = lempel_ziv_parse("abracadabra")
    assert phrases == [("a",), ("b",), ("r",), ("a", "c"), ("a", "d"), ("a", "b"), ("r", "a")]


def test_lz_entropy_uses_dictionary_size():
    rng = np.random.default_rng(4)
    msg = rng.integers(0, 2, size=1000)
    c = len(lempel_ziv_parse(msg))
    assert lempel_ziv_entropy(msg) == pytest.approx(c * math.log2(c) / 1000)


# ---------------------------------------------------------------------------
# Kontoyiannis / Gao et al. (2008) estimator (AFML 18.4, Snippet 18.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_match_length_matches_literal_snippet(seed):
    rng = np.random.default_rng(seed)
    msg = rng.integers(0, 2, size=80)
    text = "".join(map(str, msg))
    for i in range(10, 60, 7):
        n = i
        assert _match_length(msg.astype(np.int64), i, n) == naive_match_length(text, i, n)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kontoyiannis_expanding_matches_naive(seed):
    rng = np.random.default_rng(seed)
    msg = rng.integers(0, 2, size=120)
    assert kontoyiannis_entropy(msg) == pytest.approx(naive_konto(msg), rel=1e-12)


@pytest.mark.parametrize("window", [5, 20, 40])
@pytest.mark.parametrize("seed", [3, 4])
def test_kontoyiannis_sliding_matches_naive(window, seed):
    rng = np.random.default_rng(seed)
    msg = rng.integers(0, 3, size=101)  # odd length on purpose
    assert kontoyiannis_entropy(msg, window=window) == pytest.approx(
        naive_konto(msg, window), rel=1e-12
    )


def test_kontoyiannis_book_worked_examples():
    # AFML 18.4: the last bit of 10000111 is irrelevant (same rate as 10000110). The book
    # states the rates of 11100001 and 01100001 (forward order) as 0.96 and 0.84.
    assert kontoyiannis_entropy("10000111") == pytest.approx(
        kontoyiannis_entropy("10000110"), abs=1e-12
    )
    # Snippet 18.4 gives 0.9682 for 11100001; the book prints 0.96 (not reproduced
    # exactly, see docs/proofs/entropy.md). 0.8432 for 01100001 matches its 0.84.
    assert kontoyiannis_entropy("11100001") == pytest.approx(0.968, abs=0.001)
    assert kontoyiannis_entropy("11100001") == pytest.approx(0.96, abs=0.01)
    assert kontoyiannis_entropy("01100001") == pytest.approx(0.84, abs=0.005)


def test_kontoyiannis_random_is_more_entropic_than_periodic():
    rng = np.random.default_rng(8)
    rand = rng.integers(0, 2, size=600)
    periodic = np.tile([0, 1], 300)
    assert kontoyiannis_entropy(rand) > kontoyiannis_entropy(periodic) + 0.3


def test_kontoyiannis_errors():
    with pytest.raises(ValueError):
        kontoyiannis_entropy("a")
    with pytest.raises(ValueError):
        kontoyiannis_entropy("abcdef", window=0)


# ---------------------------------------------------------------------------
# Encoders (AFML 18.5)
# ---------------------------------------------------------------------------


def test_encode_binary_drops_zero_returns():
    r = pd.Series([0.1, -0.2, 0.0, 0.3, 0.0, -0.1], index=list("abcdef"))
    codes = encode_binary(r)
    assert list(codes.index) == ["a", "b", "d", "f"]
    assert list(codes.values) == [1, 0, 1, 0]


def test_encode_quantile_balances_in_sample_bins():
    rng = np.random.default_rng(5)
    r = rng.standard_normal(1000)
    codes = encode_quantile(r, 10)
    counts = np.bincount(codes, minlength=10)
    assert codes.min() == 0 and codes.max() == 9
    assert counts.min() >= 99 and counts.max() <= 101  # ~100 per letter in sample
    # Out-of-sample codes use in-sample edges.
    oos = encode_quantile(np.array([-10.0, 10.0]), 10, reference=r)
    assert list(oos) == [0, 9]


def test_encode_sigma_bins_have_fixed_width():
    r = np.array([0.0, 0.1, 0.49, 0.5, 1.0])
    codes = encode_sigma(r, sigma=0.25)
    # min 0, width 0.25, ceil(1.0/0.25) = 4 codes; r=1.0 clipped to code 3.
    assert list(codes) == [0, 0, 1, 2, 3]


def test_encode_feeds_entropy_estimators():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(400)
    codes = encode_quantile(r, 4)
    # 400 draws in 4 quantile bins: exactly 100 per code, so the plug-in entropy is exactly 2 bits.
    np.testing.assert_array_equal(np.bincount(codes, minlength=4), [100, 100, 100, 100])
    assert plug_in_entropy(codes) == pytest.approx(2.0, abs=1e-12)
    assert kontoyiannis_entropy(codes) > 0.0
