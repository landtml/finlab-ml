"""Tests for finlab.trials (trial registry feeding PBO and the deflated Sharpe ratio)."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Optional

import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from finlab.pbo import probability_of_backtest_overfitting
from finlab.stats import deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_ratio
from finlab.trials import TrialRecord, TrialRegistry


def _noise(seed: int, T: int = 64, mean: float = 0.0005) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 0.01, size=T)


def _arrays(n: int, T: int = 64, seed: int = 0) -> list[np.ndarray]:
    return [_noise(seed + i, T) for i in range(n)]


def _registry(arrays: list[np.ndarray]) -> TrialRegistry:
    reg = TrialRegistry()
    for i, x in enumerate(arrays):
        reg.record(f"trial-{i}", {"window": 10 + i}, x)
    return reg


def _hand_dsr(arrays: list[np.ndarray], n_obs: Optional[int] = None) -> float:
    """Deflated Sharpe ratio written out from its definition, without the registry."""
    N = len(arrays)
    srs = np.array([np.mean(x) / np.std(x, ddof=1) for x in arrays])
    best = int(np.argmax(srs))
    x = arrays[best]
    T = len(x) if n_obs is None else n_obs
    return deflated_sharpe_ratio(
        srs[best], N, np.var(srs, ddof=1), T, skew(x), kurtosis(x, fisher=False)
    )


# ---------------------------------------------------------------------------
# TrialRecord: construction and validation
# ---------------------------------------------------------------------------


def test_record_valid_fields() -> None:
    rec = TrialRecord(
        name="  fast  ",
        params={"b": 2, "a": "x", "c": None, "d": True, "e": 0.5},
        returns=[0.1, -0.2, 0.3],
    )
    assert rec.name == "fast"
    assert rec.params == (("a", "x"), ("b", 2), ("c", None), ("d", True), ("e", 0.5))
    assert rec.params_dict == {"a": "x", "b": 2, "c": None, "d": True, "e": 0.5}
    assert type(rec.params_dict["b"]) is int
    assert type(rec.params_dict["d"]) is bool
    assert rec.returns.dtype == np.float64
    assert rec.returns.ndim == 1
    np.testing.assert_array_equal(rec.returns, [0.1, -0.2, 0.3])


def test_record_accepts_pair_sequence_and_sorts_by_key() -> None:
    rec = TrialRecord("a", [("z", 1), ("m", 2.5)], [0.0, 1.0])
    assert rec.params == (("m", 2.5), ("z", 1))


def test_integer_returns_are_stored_as_float64() -> None:
    rec = TrialRecord("a", {}, np.array([1, 2, 3]))
    assert rec.returns.dtype == np.float64
    np.testing.assert_array_equal(rec.returns, [1.0, 2.0, 3.0])


def test_params_dict_is_a_fresh_copy() -> None:
    rec = TrialRecord("a", {"k": 1}, [0.0, 1.0])
    rec.params_dict["k"] = 99
    rec.params_dict["extra"] = 1
    assert rec.params_dict == {"k": 1}


def test_returns_are_copied_from_input() -> None:
    src = np.array([0.1, 0.2, 0.3])
    rec = TrialRecord("a", {}, src)
    src[0] = 99.0
    assert rec.returns[0] == 0.1


def test_returns_are_read_only_and_record_is_frozen() -> None:
    rec = TrialRecord("a", {}, [0.1, 0.2])
    with pytest.raises(ValueError, match="read-only"):
        rec.returns[0] = 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.returns = np.zeros(2)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"name": "", "params": {}, "returns": [0.1, 0.2]}, "non-empty"),
        ({"name": "   ", "params": {}, "returns": [0.1, 0.2]}, "non-empty"),
        ({"name": 3, "params": {}, "returns": [0.1, 0.2]}, "must be a str"),
        ({"name": "a", "params": {}, "returns": [0.1, float("nan")]}, "finite"),
        ({"name": "a", "params": {}, "returns": [0.1, float("inf")]}, "finite"),
        ({"name": "a", "params": {}, "returns": [0.1]}, "at least 2"),
        ({"name": "a", "params": {}, "returns": []}, "at least 2"),
        ({"name": "a", "params": {}, "returns": [[0.1, 0.2], [0.3, 0.4]]}, "1-D"),
        ({"name": "a", "params": {}, "returns": ["x", "y"]}, "real numbers"),
        ({"name": "a", "params": {"k": [1, 2]}, "returns": [0.1, 0.2]}, "must be str"),
        ({"name": "a", "params": {"k": {"x": 1}}, "returns": [0.1, 0.2]}, "must be str"),
        ({"name": "a", "params": {"k": float("nan")}, "returns": [0.1, 0.2]}, "must be finite"),
        ({"name": "a", "params": {"": 1}, "returns": [0.1, 0.2]}, "keys must be non-empty"),
        ({"name": "a", "params": {1: "x"}, "returns": [0.1, 0.2]}, "keys must be non-empty"),
        (
            {"name": "a", "params": [("k", 1), ("k", 2)], "returns": [0.1, 0.2]},
            "duplicate param key",
        ),
        ({"name": "a", "params": 5, "returns": [0.1, 0.2]}, "mapping or an iterable"),
    ],
)
def test_record_validation_errors(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrialRecord(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TrialRegistry: registration and the performance matrix
# ---------------------------------------------------------------------------


def test_registry_counts_and_keeps_registration_order() -> None:
    reg = TrialRegistry()
    assert reg.n_trials == 0
    assert reg.records == ()
    for i, x in enumerate(_arrays(3)):
        rec = reg.record(f"t{i}", {"i": i}, x)
        assert isinstance(rec, TrialRecord)
        assert reg.n_trials == i + 1
        assert reg.records[-1] is rec
    assert [r.name for r in reg.records] == ["t0", "t1", "t2"]


def test_records_is_a_tuple_snapshot() -> None:
    reg = _registry(_arrays(2))
    snap = reg.records
    assert isinstance(snap, tuple)
    reg.record("late", {}, _noise(99))
    assert len(snap) == 2
    assert reg.n_trials == 3


def test_duplicate_name_rejected_after_stripping() -> None:
    reg = TrialRegistry()
    reg.record("alpha", {}, _noise(1))
    with pytest.raises(ValueError, match="already registered"):
        reg.record("alpha", {"other": 1}, _noise(2))
    with pytest.raises(ValueError, match="already registered"):
        reg.record("  alpha ", {}, _noise(3))
    assert reg.n_trials == 1


def test_failed_record_is_not_registered() -> None:
    reg = TrialRegistry()
    with pytest.raises(ValueError):
        reg.record("bad", {}, [0.1])
    assert reg.n_trials == 0
    reg.record("bad", {}, _noise(1))  # the name is still free
    assert reg.n_trials == 1


def test_performance_matrix_columns_follow_registration_order() -> None:
    arrays = _arrays(4, T=32)
    reg = _registry(arrays)
    M = reg.performance_matrix()
    assert M.shape == (32, 4)
    assert M.dtype == np.float64
    for j, x in enumerate(arrays):
        np.testing.assert_array_equal(M[:, j], x)


def test_performance_matrix_is_independent_of_the_records() -> None:
    reg = _registry(_arrays(2, T=8))
    before = reg.records[0].returns.copy()
    reg.performance_matrix()[0, 0] = 123.0
    np.testing.assert_array_equal(reg.records[0].returns, before)


def test_performance_matrix_needs_two_trials_of_equal_length() -> None:
    reg = TrialRegistry()
    with pytest.raises(ValueError, match="at least 2 trials"):
        reg.performance_matrix()
    reg.record("a", {}, _noise(1, T=32))
    with pytest.raises(ValueError, match="at least 2 trials"):
        reg.performance_matrix()
    reg.record("b", {}, _noise(2, T=40))
    with pytest.raises(ValueError, match="same length"):
        reg.performance_matrix()


# ---------------------------------------------------------------------------
# Feeds PBO
# ---------------------------------------------------------------------------


def test_pbo_equals_direct_call_exactly() -> None:
    arrays = _arrays(5, T=64, seed=10)
    reg = _registry(arrays)
    got = reg.pbo(8)
    want = probability_of_backtest_overfitting(np.column_stack(arrays), 8)
    assert got.pbo == want.pbo
    assert got.n_combinations == want.n_combinations == math.comb(8, 4)
    assert np.array_equal(got.logits, want.logits)


def test_pbo_default_partitions_is_sixteen() -> None:
    arrays = _arrays(4, T=64, seed=20)
    reg = _registry(arrays)
    got = reg.pbo()
    want = probability_of_backtest_overfitting(np.column_stack(arrays), 16)
    assert got.n_combinations == math.comb(16, 8)
    assert got.pbo == want.pbo
    assert np.array_equal(got.logits, want.logits)


def test_pbo_uses_the_registry_count_as_n() -> None:
    arrays = _arrays(5, T=64, seed=30)
    reg = _registry(arrays)
    before = reg.pbo(8)
    extra = _noise(77, T=64)
    reg.record("extra", {}, extra)
    after = reg.pbo(8)
    want = probability_of_backtest_overfitting(np.column_stack(arrays + [extra]), 8)
    assert np.array_equal(after.logits, want.logits)
    assert not np.array_equal(before.logits, after.logits)


def test_pbo_rejects_partitions_that_do_not_divide_t() -> None:
    reg = _registry(_arrays(3, T=63, seed=35))
    with pytest.raises(ValueError):
        reg.pbo(8)


# ---------------------------------------------------------------------------
# Feeds the deflated Sharpe ratio
# ---------------------------------------------------------------------------


def test_dsr_matches_hand_computation() -> None:
    arrays = _arrays(6, T=60, seed=40)
    reg = _registry(arrays)
    dsr = reg.deflated_sharpe()
    assert 0.0 < dsr < 1.0
    assert dsr == _hand_dsr(arrays)


def test_adding_a_trial_changes_n_and_the_result() -> None:
    arrays = _arrays(6, T=60, seed=40)
    reg = _registry(arrays)
    before = reg.deflated_sharpe()
    extra = _noise(123, T=60, mean=0.0)
    reg.record("extra", {}, extra)
    after = reg.deflated_sharpe()
    assert reg.n_trials == 7
    assert after == _hand_dsr(arrays + [extra])
    assert after != before


def test_dsr_n_obs_override() -> None:
    arrays = _arrays(4, T=60, seed=50)
    reg = _registry(arrays)
    assert reg.deflated_sharpe(n_obs=500) == _hand_dsr(arrays, n_obs=500)
    assert reg.deflated_sharpe(n_obs=500) != reg.deflated_sharpe()


def test_dsr_selects_best_trial_when_lengths_differ() -> None:
    arrays = [_noise(60, T=40), _noise(61, T=90), _noise(62, T=25)]
    reg = _registry(arrays)
    assert reg.deflated_sharpe() == _hand_dsr(arrays)
    with pytest.raises(ValueError):
        reg.performance_matrix()


def test_dsr_single_trial_is_psr_against_zero() -> None:
    x = _noise(70, T=50, mean=0.001)
    reg = _registry([x])
    sr = sharpe_ratio(x, annualize=False)
    expected = probabilistic_sharpe_ratio(sr, 0.0, x.size, skew(x), kurtosis(x, fisher=False))
    assert reg.deflated_sharpe() == expected


def test_dsr_rejects_empty_registry_and_zero_variance_trial() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TrialRegistry().deflated_sharpe()
    reg = TrialRegistry()
    reg.record("flat", {}, np.full(10, 0.25))
    reg.record("noisy", {}, _noise(80, T=10))
    with pytest.raises(ValueError, match="zero return variance"):
        reg.deflated_sharpe()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _sample_registry() -> TrialRegistry:
    reg = TrialRegistry()
    reg.record(
        "grid/1",
        {"fast": 10, "slow": 50.0, "use_vol": True, "tag": None, "label": "a b"},
        _noise(90, T=32),
    )
    reg.record(
        "grid/2",
        {"fast": 20, "slow": 80.5, "use_vol": False, "tag": "x", "label": ""},
        _noise(91, T=32),
    )
    reg.record("grid/3", {}, _noise(92, T=32))
    return reg


def _assert_same_registry(a: TrialRegistry, b: TrialRegistry) -> None:
    assert a.n_trials == b.n_trials
    for ra, rb in zip(a.records, b.records, strict=True):
        assert ra.name == rb.name
        assert ra.params == rb.params
        assert [type(v) for _, v in ra.params] == [type(v) for _, v in rb.params]
        assert ra.returns.dtype == rb.returns.dtype == np.float64
        assert np.array_equal(ra.returns, rb.returns)


def test_json_round_trip_is_exact_and_stable() -> None:
    reg = _sample_registry()
    text = reg.to_json()
    back = TrialRegistry.from_json(text)
    _assert_same_registry(reg, back)
    assert back.to_json() == text
    assert back.deflated_sharpe() == reg.deflated_sharpe()


def test_dict_round_trip_is_exact_and_json_compatible() -> None:
    reg = _sample_registry()
    d = reg.to_dict()
    assert d["format"] == 1
    assert json.loads(json.dumps(d)) == d
    _assert_same_registry(reg, TrialRegistry.from_dict(d))


def test_record_dict_round_trip() -> None:
    rec = _sample_registry().records[0]
    back = TrialRecord.from_dict(rec.to_dict())
    assert back.name == rec.name
    assert back.params == rec.params
    assert np.array_equal(back.returns, rec.returns)


def test_floats_round_trip_bit_for_bit() -> None:
    values = [0.1, 1.0 / 3.0, 1e-300, -2.5e-17, 5e-324, 1.7976931348623157e308, -0.0]
    reg = TrialRegistry()
    reg.record("extreme", {"x": 0.1, "y": 1e-7}, values)
    back = TrialRegistry.from_json(reg.to_json()).records[0]
    assert np.array_equal(
        back.returns.view(np.uint64), np.array(values, dtype=np.float64).view(np.uint64)
    )
    assert back.params_dict == {"x": 0.1, "y": 1e-7}


@pytest.mark.parametrize("fmt", [2, 0, None, "1", 1.0, True])
def test_unknown_registry_format_is_rejected(fmt: object) -> None:
    d = _sample_registry().to_dict()
    d["format"] = fmt
    with pytest.raises(ValueError, match="format"):
        TrialRegistry.from_dict(d)
    with pytest.raises(ValueError, match="format"):
        TrialRegistry.from_json(json.dumps(d))


def test_missing_format_and_unknown_record_format_are_rejected() -> None:
    d = _sample_registry().to_dict()
    del d["format"]
    with pytest.raises(ValueError, match="format"):
        TrialRegistry.from_dict(d)
    rec_dict = _sample_registry().records[0].to_dict()
    rec_dict["format"] = 2
    with pytest.raises(ValueError, match="format"):
        TrialRecord.from_dict(rec_dict)


def test_from_dict_rejects_malformed_input() -> None:
    good = _sample_registry().to_dict()
    with pytest.raises(ValueError, match="missing key 'trials'"):
        TrialRegistry.from_dict({"format": 1})
    with pytest.raises(ValueError, match="already registered"):
        TrialRegistry.from_dict({"format": 1, "trials": [good["trials"][0], good["trials"][0]]})
    with pytest.raises(ValueError, match="missing key 'returns'"):
        TrialRegistry.from_dict({"format": 1, "trials": [{"format": 1, "name": "a", "params": {}}]})
    with pytest.raises(ValueError, match="mapping"):
        TrialRegistry.from_json("[1, 2]")


def test_from_json_rejects_non_finite_returns() -> None:
    text = json.dumps(
        {
            "format": 1,
            "trials": [
                {"format": 1, "name": "a", "params": {}, "returns": [0.1, float("nan")]},
            ],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        TrialRegistry.from_json(text)
