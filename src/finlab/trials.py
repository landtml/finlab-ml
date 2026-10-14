"""Trial registry: record each trial once and compute PBO and DSR from the record.

Every trial a researcher runs is registered with its parameters and its return
series. :class:`TrialRegistry` then builds the performance matrix for
:func:`finlab.pbo.probability_of_backtest_overfitting` and the inputs of
:func:`finlab.stats.deflated_sharpe_ratio` from that record, so N and the
performance matrix never come from a typed-in number. The record itself is not an
AFML construct; it supplies the inputs that AFML chapter 11 (PBO) and chapter 14
(deflated Sharpe) need::

    reg = TrialRegistry()
    reg.record("fast=10", {"fast": 10}, returns_fast_10)
    reg.record("fast=20", {"fast": 20}, returns_fast_20)
    pbo = reg.pbo(n_partitions=8).pbo
    dsr = reg.deflated_sharpe()

N is the number of registered trials, ``reg.n_trials``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import numpy as np
from scipy.stats import kurtosis, skew

from finlab.pbo import PBOResult, probability_of_backtest_overfitting
from finlab.stats import deflated_sharpe_ratio, sharpe_ratio

__all__ = ["TrialRecord", "TrialRegistry"]

ParamValue = Union[str, int, float, bool, None]
ParamsInput = Union[Mapping[str, ParamValue], Sequence[tuple[str, ParamValue]]]
ReturnsInput = Union[np.ndarray, Sequence[float]]

_FORMAT_VERSION = 1


@dataclass(frozen=True, eq=False)
class TrialRecord:
    """One registered trial: a name, its parameters and its return series.

    Parameters
    ----------
    name : str
        Label of the trial. Surrounding whitespace is removed, and the result
        must be non-empty.
    params : mapping or sequence of (key, value) pairs
        Keys are non-empty strings. Values are str, int, float, bool or None;
        floats must be finite. Stored as a tuple of pairs sorted by key, and
        read back as a new dict with :attr:`params_dict`.
    returns : array-like
        Per-period returns: at least two finite values. Stored as a read-only
        1-D float64 copy, so later changes to the input do not reach the record.

    Two records are equal when their names, their parameters (including each
    value's type) and their returns are equal. Equal records have equal hashes.

    Raises
    ------
    ValueError
        If any argument fails validation.
    """

    name: str
    params: ParamsInput
    returns: ReturnsInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name(self.name))
        object.__setattr__(self, "params", _check_params(self.params))
        object.__setattr__(self, "returns", _check_returns(self.returns, self.name))

    @property
    def params_dict(self) -> dict[str, ParamValue]:
        """Parameters as a new dict. Changing it does not change the record."""
        return dict(cast("tuple[tuple[str, ParamValue], ...]", self.params))

    def _key(self) -> tuple[str, tuple[tuple[str, str, ParamValue], ...], bytes]:
        # Python treats 1, 1.0 and True as equal, but they are different parameter
        # values for a researcher, so the type name is part of the key.
        pairs = cast("tuple[tuple[str, ParamValue], ...]", self.params)
        params = tuple((key, type(value).__name__, value) for key, value in pairs)
        return (self.name, params, _returns_key(cast(np.ndarray, self.returns)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrialRecord):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict with ``"format": 1``."""
        return {
            "format": _FORMAT_VERSION,
            "name": self.name,
            "params": self.params_dict,
            "returns": [float(v) for v in self.returns],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialRecord:
        """Rebuild a record from :meth:`to_dict` output, validating it as the constructor does."""
        data = _require_mapping(data, "trial record")
        _check_format(data, "trial record")
        return cls(
            name=_get(data, "name", "trial record"),
            params=_get(data, "params", "trial record"),
            returns=_get(data, "returns", "trial record"),
        )


class TrialRegistry:
    """Ordered record of the trials run in one research project.

    Trials are kept in registration order. :attr:`n_trials` is the N used by
    :meth:`pbo` and :meth:`deflated_sharpe`.

    Two registries are equal when they hold equal records in the same order,
    because the order sets the matrix columns and the tie-break in
    :meth:`deflated_sharpe`. A registry is mutable, so it is not hashable.
    """

    def __init__(self) -> None:
        self._records: list[TrialRecord] = []
        self._names: set[str] = set()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrialRegistry):
            return NotImplemented
        return self._records == other._records

    @property
    def n_trials(self) -> int:
        """Number of registered trials, N."""
        return len(self._records)

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        """Snapshot of the registered trials, in registration order."""
        return tuple(self._records)

    def record(self, name: str, params: ParamsInput, returns: ReturnsInput) -> TrialRecord:
        """Validate one trial, register it and return the record.

        Raises
        ------
        ValueError
            If the record is invalid or its name (after stripping whitespace)
            is already registered.
        """
        rec = TrialRecord(name=name, params=params, returns=returns)
        self._add(rec)
        return rec

    def _add(self, rec: TrialRecord) -> None:
        if rec.name in self._names:
            raise ValueError(f"trial name {rec.name!r} is already registered")
        self._records.append(rec)
        self._names.add(rec.name)

    def performance_matrix(self) -> np.ndarray:
        """Return the (T, N) matrix of returns, one column per trial.

        Columns follow registration order.

        Raises
        ------
        ValueError
            If there are fewer than two trials, or the trials do not all have
            the same number of returns.
        """
        if self.n_trials < 2:
            raise ValueError(f"performance_matrix needs at least 2 trials, have {self.n_trials}")
        first = self._records[0]
        T = np.asarray(first.returns).size
        for rec in self._records[1:]:
            size = np.asarray(rec.returns).size
            if size != T:
                raise ValueError(
                    f"trial {rec.name!r} has {size} returns, but trial {first.name!r} "
                    f"has {T}; all trials must have the same length"
                )
        return np.column_stack([np.asarray(rec.returns) for rec in self._records])

    def pbo(self, n_partitions: int = 16) -> PBOResult:
        """Probability of backtest overfitting over the registered trials.

        Calls :func:`finlab.pbo.probability_of_backtest_overfitting` on
        :meth:`performance_matrix`, so N is :attr:`n_trials`.
        """
        return probability_of_backtest_overfitting(
            self.performance_matrix(), n_partitions=n_partitions
        )

    def deflated_sharpe(self, n_obs: Optional[int] = None) -> float:
        """Deflated Sharpe ratio of the best registered trial.

        Parameters
        ----------
        n_obs : int, optional
            Number of return observations T in the PSR. Defaults to the number
            of returns of the selected trial.

        Returns
        -------
        float
            ``deflated_sharpe_ratio(sr, N, var_sr, T, skew, kurt)`` with N =
            :attr:`n_trials` and the inputs described below.

        Notes
        -----
        The selected trial is the one with the highest non-annualised Sharpe
        ratio (``sharpe_ratio(..., annualize=False)``); on a tie the earlier
        registration wins. ``var_sr`` is the sample variance (ddof=1) of the
        non-annualised Sharpe ratios of all registered trials, including the
        selected one. Skewness and non-excess kurtosis are the scipy estimates
        on the selected trial's returns, or 0 and 3 when that trial is identically
        zero. With one registered trial ``var_sr``
        is 0; the expected maximum is then 0, so the result is the PSR against
        zero skill.

        Raises
        ------
        ValueError
            If no trial is registered, or a trial is a nonzero constant (zero
            return variance with a nonzero mean, so its Sharpe ratio is infinite
            and the cross-trial variance is undefined). A trial that is identically
            zero has Sharpe ratio 0 and is not an error.
        """
        if self.n_trials == 0:
            raise ValueError("deflated_sharpe needs at least one registered trial")
        srs = np.array(
            [sharpe_ratio(np.asarray(rec.returns), annualize=False) for rec in self._records],
            dtype=np.float64,
        )
        not_finite = np.flatnonzero(~np.isfinite(srs))
        if not_finite.size:
            name = self._records[int(not_finite[0])].name
            raise ValueError(
                f"trial {name!r} has zero return variance, so its Sharpe ratio is not "
                "finite and the deflated Sharpe ratio is undefined"
            )
        best = int(np.argmax(srs))
        x = np.asarray(self._records[best].returns)
        n = self.n_trials
        var_sr = float(np.var(srs, ddof=1)) if n > 1 else 0.0
        # Skew and kurtosis of a constant series are undefined; a normal distribution's values stand in.
        if np.all(x == 0):
            g3, g4 = 0.0, 3.0
        else:
            g3, g4 = float(skew(x)), float(kurtosis(x, fisher=False))
        return deflated_sharpe_ratio(
            float(srs[best]),
            n,
            var_sr,
            x.size if n_obs is None else n_obs,
            g3,
            g4,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict with ``"format": 1`` and the trials in order."""
        return {"format": _FORMAT_VERSION, "trials": [rec.to_dict() for rec in self._records]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialRegistry:
        """Rebuild a registry from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            On an unknown format version, a malformed trial, or a duplicate name.
        """
        data = _require_mapping(data, "trial registry")
        _check_format(data, "trial registry")
        trials = _get(data, "trials", "trial registry")
        if not isinstance(trials, list):
            raise ValueError("trial registry 'trials' must be a list")
        registry = cls()
        for item in trials:
            registry._add(TrialRecord.from_dict(item))
        return registry

    def to_json(self) -> str:
        """Serialise to JSON text.

        Floats are written with repr precision, so they round-trip exactly.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> TrialRegistry:
        """Parse JSON text produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))


def _returns_key(returns: np.ndarray) -> bytes:
    # Adding 0.0 turns -0.0 into 0.0, which equals it, so both must share a byte string.
    # IEEE comparison makes NaN unequal to itself, which would make a record unequal to
    # itself, so every NaN maps to one canonical NaN before the bytes are taken.
    canonical = np.where(np.isnan(returns), np.nan, returns + 0.0)
    return canonical.tobytes()


def _check_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ValueError(f"trial name must be a str, got {type(name).__name__}")
    stripped = name.strip()
    if not stripped:
        raise ValueError("trial name must be non-empty")
    return stripped


def _check_params(params: Any) -> tuple[tuple[str, ParamValue], ...]:
    if isinstance(params, Mapping):
        pairs = list(params.items())
    else:
        try:
            items = list(params)
        except TypeError as exc:
            raise ValueError(
                "params must be a mapping or an iterable of (key, value) pairs"
            ) from exc
        pairs = []
        for item in items:
            # A string of length 2 would unpack into a key and a value, so only tuples and lists count.
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    f"params must be a mapping or an iterable of (key, value) pairs; got item {item!r}"
                )
            pairs.append((item[0], item[1]))
    checked: list[tuple[str, ParamValue]] = []
    seen: set[str] = set()
    for key, value in pairs:
        if not isinstance(key, str) or not key:
            raise ValueError(f"param keys must be non-empty str, got {key!r}")
        if key in seen:
            raise ValueError(f"duplicate param key {key!r}")
        seen.add(key)
        checked.append((str(key), _check_param_value(key, value)))
    checked.sort(key=lambda pair: pair[0])
    return tuple(checked)


def _check_param_value(key: str, value: Any) -> ParamValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"param {key!r} must be finite, got {value!r}")
        return float(value)
    if isinstance(value, str):
        return str(value)
    raise ValueError(
        f"param {key!r} must be str, int, float, bool or None, got {type(value).__name__}"
    )


def _check_returns(returns: Any, name: str) -> np.ndarray:
    # The shape is checked before the items: a DataFrame or 2-D array iterates by column label, so its
    # labels would be refused as string returns. An ndarray or DataFrame has ndim; a sequence does not.
    ndim = getattr(returns, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {tuple(returns.shape)}")
    try:
        items = list(returns)
    except TypeError as exc:
        raise ValueError(f"trial {name!r}: returns must be a sequence of real numbers") from exc
    for item in items:
        # np.array would turn "0.1" into 0.1 and True into 1.0; neither is a return.
        if isinstance(item, (str, bytes, bool, np.bool_)):
            raise ValueError(
                f"trial {name!r}: returns must be real numbers, got {type(item).__name__} {item!r}"
            )
    try:
        arr = np.array(returns, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be a sequence of real numbers") from exc
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError(f"returns must have at least 2 observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must contain only finite values")
    arr.setflags(write=False)
    return arr


def _require_mapping(data: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{what} must be a mapping, got {type(data).__name__}")
    return data


def _check_format(data: Mapping[str, Any], what: str) -> None:
    fmt = data.get("format")
    if type(fmt) is not int or fmt != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported {what} format {fmt!r}; this version reads format {_FORMAT_VERSION}"
        )


def _get(data: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in data:
        raise ValueError(f"{what} is missing key {key!r}")
    return data[key]
