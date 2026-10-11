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
    # min 0, width 0.25. r=1.0 is the upper edge of code 3, so it gets its own code 4. Clipping it
    # into code 3 would merge two distinct bins.
    assert list(codes) == [0, 0, 1, 2, 4]


def test_encode_feeds_entropy_estimators():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(400)
    codes = encode_quantile(r, 4)
    # 400 draws in 4 quantile bins: exactly 100 per code, so the plug-in entropy is exactly 2 bits.
    np.testing.assert_array_equal(np.bincount(codes, minlength=4), [100, 100, 100, 100])
    assert plug_in_entropy(codes) == pytest.approx(2.0, abs=1e-12)
    assert kontoyiannis_entropy(codes) > 0.0


@pytest.mark.parametrize("n_bins", [2.5, float("nan"), float("inf")])
def test_encode_quantile_rejects_non_integer_n_bins(n_bins):
    x = np.random.default_rng(0).normal(size=200)
    with pytest.raises(ValueError, match="n_bins"):
        encode_quantile(x, n_bins)


def test_encode_quantile_accepts_integer_valued_n_bins():
    x = np.random.default_rng(0).normal(size=1000)
    ref = encode_quantile(x, 3)
    np.testing.assert_array_equal(encode_quantile(x, 3.0), ref)
    np.testing.assert_array_equal(encode_quantile(x, np.int64(3)), ref)
    assert np.bincount(ref).tolist() == [333, 333, 334]


def test_encode_sigma_rejects_alphabet_beyond_exact_codes() -> None:
    # Codes past the int64 range must not wrap and merge onto symbol 0.
    with pytest.raises(ValueError, match="too large"):
        encode_sigma(np.array([0.0, 1.0, 2.0]), 1e-19)
    with pytest.raises(ValueError, match="too large"):
        encode_sigma(np.array([0.0, 1.0]), 1e-300)


def test_encode_sigma_large_alphabet_within_exact_range_unchanged() -> None:
    codes = encode_sigma(np.array([0.0, 1.0, 2.0]), 1e-15)
    assert codes.tolist() == [0, 999999999999999, 1999999999999999]


def test_kontoyiannis_infinite_window_is_value_error() -> None:
    # An infinite window must give a ValueError, not an OverflowError from int(inf).
    from finlab.entropy import kontoyiannis_entropy

    with pytest.raises(ValueError, match="window"):
        kontoyiannis_entropy("abcabcabab", window=float("inf"))


@pytest.mark.parametrize("bad_word_length", [2.5, float("nan"), float("inf")])
def test_plug_in_entropy_rejects_non_integer_word_length(bad_word_length: float) -> None:
    # A fractional word_length must be refused, not truncated to 2.
    from finlab.entropy import plug_in_entropy

    with pytest.raises(ValueError, match="word_length"):
        plug_in_entropy("aabbabab", word_length=bad_word_length)


def test_kontoyiannis_rejects_fractional_window() -> None:
    # A fractional window must be refused, not truncated to 5.
    from finlab.entropy import kontoyiannis_entropy

    msg = "abcabcabcabcabcabcab"
    with pytest.raises(ValueError, match="window"):
        kontoyiannis_entropy(msg, window=5.5)



def test_encode_sigma_empty_input_raises_clear_error() -> None:
    # An empty input must raise a clear error, not numpy's "zero-size array" error.
    from finlab.entropy import encode_sigma

    with pytest.raises(ValueError, match="empty"):
        encode_sigma([], 0.1)


def test_encode_sigma_maximum_on_a_bin_edge_gets_its_own_code() -> None:
    # With (max - min) / sigma an integer s, the maximum sits on the edge of code s, but only s codes
    # were allocated, so np.clip folded it into code s - 1 and distinct bins merged.
    from finlab.entropy import encode_sigma

    assert encode_sigma(np.array([0.0, 1.0, 2.0]), 1.0).tolist() == [0, 1, 2]
    assert encode_sigma(np.array([0.0, 2.0]), 1.0).tolist() == [0, 2]
    # Non-integer spans are unchanged: ceil and floor + 1 agree there.
    assert encode_sigma(np.array([0.0, 2.5]), 1.0).tolist() == [0, 2]


def test_entropy_rejects_nan_in_message() -> None:
    # np.unique counted NaN as a third symbol, so a message with a missing value gave a biased
    # estimate with no error. The encode_* functions already refuse non-finite input.
    from finlab.entropy import lempel_ziv_entropy, plug_in_entropy

    with pytest.raises(ValueError, match="NaN or infinite"):
        plug_in_entropy([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        lempel_ziv_entropy([1.0, np.nan])
    assert plug_in_entropy([1.0, 2.0, 2.0]) == pytest.approx(0.9182958340544896)


def test_encode_sigma_rejects_infinite_sigma() -> None:
    # sigma = inf passed the "sigma > 0" check, the span became 0, and every return got code 0
    # silently. A non-finite sigma is refused.
    from finlab.entropy import encode_sigma

    with pytest.raises(ValueError, match="sigma must be"):
        encode_sigma(np.array([0.0, 1.0, 2.0]), float("inf"))
    with pytest.raises(ValueError, match="sigma must be"):
        encode_sigma(np.array([0.0, 1.0, 2.0]), float("nan"))


def test_plug_in_entropy_refuses_bool_word_length() -> None:
    # A bool word_length must be refused, not read as 1.
    from finlab.entropy import plug_in_entropy

    with pytest.raises(ValueError, match="word_length"):
        plug_in_entropy([0, 1, 1, 0, 1, 0], word_length=True)


def test_encode_quantile_bounds_n_bins() -> None:
    # n_bins had no upper bound, so 1e9 bins allocated about 7 GiB of edges.
    # The cap is 1e6 (a provisional choice); 1e6 bins still work.
    from finlab.entropy import encode_quantile

    with pytest.raises(ValueError, match="n_bins must be <="):
        encode_quantile([1.0, 2.0, 3.0], 10**9)
    codes = np.asarray(encode_quantile([1.0, 2.0, 3.0], 10**6))
    assert codes[0] < codes[1] < codes[2] < 10**6  # quantile bins, in range; 1e6 bins still work


def test_entropy_refuses_object_nan_as_float_nan_does() -> None:
    # The NaN refusal ran only for float dtypes, so an object array of the same values counted each
    # NaN as its own symbol (2.0), while the float form was refused.
    from finlab.entropy import lempel_ziv_entropy, plug_in_entropy

    n = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        plug_in_entropy(np.array([1.0, n, 2.0, n]), 1)
    with pytest.raises(ValueError, match="NaN or infinite"):
        plug_in_entropy(np.array([1.0, n, 2.0, n], dtype=object), 1)
    with pytest.raises(ValueError, match="NaN or infinite"):
        plug_in_entropy([1.0, None, 2.0], 1)
    with pytest.raises(ValueError, match="NaN or infinite"):
        lempel_ziv_entropy(np.array([1.0, n, 2.0, n], dtype=object))


def test_entropy_object_string_messages_are_unchanged() -> None:
    # A non-numeric object message of strings is still a valid message.
    from finlab.entropy import plug_in_entropy

    msg = np.array(["a", "b", "a", "b"], dtype=object)
    assert plug_in_entropy(msg, 1) == pytest.approx(1.0)


def test_kontoyiannis_refuses_bool_window() -> None:
    # window=True was read as 1. Bool is refused by name, as plug_in_entropy refuses bool
    # word_length.
    from finlab.entropy import kontoyiannis_entropy

    msg = [1, 2, 1, 3, 2, 1, 1, 2, 3, 1, 2, 2, 1, 3] * 2
    with pytest.raises(ValueError, match="got True"):
        kontoyiannis_entropy(msg, True)
    assert math.isfinite(kontoyiannis_entropy(msg, 1))


def test_encode_sigma_decimal_grid_gives_distinct_codes() -> None:
    # 0.3 / 0.1 = 2.9999999999999996 in float, so floor() put the value 0.3 one step too low and
    # merged it with 0.2. Values one sigma apart need distinct codes.
    from finlab.entropy import encode_sigma

    assert np.asarray(encode_sigma(np.array([0.2, 0.3]), 0.1)).tolist() == [0, 1]
    grid = np.round(np.arange(0, 1.01, 0.1), 10)
    assert np.asarray(encode_sigma(grid, 0.1)).tolist() == list(range(11))
    rng = np.random.default_rng(3)
    vals = np.round(rng.uniform(0, 5, 2000), 2)
    codes = np.asarray(encode_sigma(vals, 0.01))
    assert len(np.unique(codes)) == len(np.unique(vals))


def test_encode_sigma_integer_span_keeps_maximum_code() -> None:
    # The maximum of an integer span keeps its own code.
    from finlab.entropy import encode_sigma

    assert np.asarray(encode_sigma(np.array([0.0, 1.0, 2.0]), 1.0)).tolist() == [0, 1, 2]
    assert np.asarray(encode_sigma(np.array([0.0, 0.5, 1.0, 1.5]), 0.5)).tolist() == [0, 1, 2, 3]


def test_tied_values_straddling_edges_warn():
    # 60 zeros and 40 normals: the zero tie covers ranks that span several of the 5 bins.
    rng = np.random.default_rng(0)
    r = np.concatenate([np.zeros(60), rng.standard_normal(40)])
    with pytest.warns(UserWarning, match="tie|equal") as record:
        codes = encode_quantile(r, 5)
    assert len(record) == 1
    assert "spans bins 1 to 4 of 5" in str(record[0].message)
    # The numbers are unchanged: the tie still goes to one code.
    np.testing.assert_array_equal(np.bincount(codes, minlength=5), [20, 1, 0, 0, 79])


def test_untied_input_does_not_warn():
    import warnings

    rng = np.random.default_rng(1)
    r = rng.standard_normal(500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encode_quantile(r, 5)
        encode_quantile(r, 10, reference=r[:300])


def test_plug_in_entropy_refuses_nan_list_symbol() -> None:
    # A list that mixes strings with NaN or inf was coerced to the strings "nan" and "inf", which were
    # then counted as symbols. Such a message is now refused by name; clean string messages are unchanged.
    from finlab.entropy import lempel_ziv_entropy, plug_in_entropy

    with pytest.raises(ValueError, match="NaN or infinite"):
        plug_in_entropy(["a", np.nan, "b", "a"], 1)
    with pytest.raises(ValueError, match="NaN or infinite"):
        plug_in_entropy(["a", np.inf, "b", "a"], 1)
    with pytest.raises(ValueError, match="NaN or infinite"):
        lempel_ziv_entropy(["a", np.nan, "b"])
    assert plug_in_entropy(["a", "b", "a"], 1) == pytest.approx(0.9182958340544896)
    assert plug_in_entropy(["nan", "a", "nan"], 1) == pytest.approx(0.9182958340544896)



def test_plug_in_entropy_refuses_nat_message() -> None:
    # NaT in a datetime message was counted as a symbol, and in an object array it split equal dates
    # into different codes. It is now refused by name as a missing value.
    from finlab.entropy import plug_in_entropy

    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="missing values"):
        plug_in_entropy([d1, pd.NaT, d2, d1], 1)
    with pytest.raises(ValueError, match="missing values"):
        plug_in_entropy(pd.Series(pd.to_datetime(["2024-01-01", None, "2024-01-02", "2024-01-01"])), 1)
    assert plug_in_entropy([d1, d2, d1], 1) == pytest.approx(0.9182958340544896)
    assert plug_in_entropy(pd.Series([d1, d2, d1]), 1) == pytest.approx(0.9182958340544896)
