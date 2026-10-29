"""A naive index against an aware one is refused with a message that names the timezone."""

import numpy as np
import pandas as pd
import pytest

from quantowl.bars import time_bars, volume_bars
from quantowl.labeling import add_vertical_barrier, get_bins, get_events
from quantowl.bet_sizing import average_active_signals
from quantowl.ensemble import SequentialBootstrapBagging
from quantowl.importance import _check_xy
from quantowl.microstructure import kyle_lambda
from quantowl.structural_breaks import cusum_test


class _FrequencyClassifier:
    """Predicts the training class frequencies; needs no scikit-learn in the core install."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.freq_ = np.array([np.mean(np.asarray(y) == c) for c in self.classes_])
        return self

    def predict_proba(self, X):
        return np.tile(self.freq_, (len(X), 1))
from quantowl._checks import _same_index, tz_mismatch_message
from quantowl.tuning import _as_arrays
from quantowl.weights import num_co_events, sample_weight_by_return

N = 30
AWARE = pd.date_range("2024-01-01", periods=N, freq="D", tz="America/New_York")
NAIVE = AWARE.tz_localize(None)


def _check(excinfo):
    msg = str(excinfo.value)
    assert ("tz-naive" in msg and "tz-aware" in msg) or "America/New_York" in msg


def _xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(N, 2)), index=AWARE)
    y = pd.Series(np.arange(N) % 2, index=NAIVE)
    return X, y


class _Stub:
    """Never fitted: the index check refuses the call before the base estimator runs."""


def test_ensemble_y():
    X, y = _xy()
    t1 = pd.Series(AWARE + pd.Timedelta(days=2), index=AWARE)
    with pytest.raises(ValueError) as e:
        SequentialBootstrapBagging(_Stub, 3, seed=1).fit(X, y, t1=t1)
    _check(e)


def test_tuning_sample_weight():
    X, _ = _xy()
    y = pd.Series(np.arange(N) % 2, index=AWARE)
    w = pd.Series(np.ones(N), index=NAIVE)
    with pytest.raises(ValueError) as e:
        _as_arrays(X, y, w)
    _check(e)


def test_importance_y():
    X, y = _xy()
    with pytest.raises(ValueError) as e:
        _check_xy(X, y)
    _check(e)


def test_kyle_prices():
    prices = pd.Series(np.cumsum(np.ones(N)) + 100.0, index=NAIVE)
    vol = pd.Series(np.tile([1.0, -1.0], N // 2), index=AWARE)
    with pytest.raises(ValueError) as e:
        kyle_lambda(prices, vol)
    _check(e)


def test_cusum_exog():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=N), index=AWARE)
    ex = pd.Series(rng.normal(size=N), index=NAIVE)
    with pytest.raises(ValueError) as e:
        cusum_test(s, exog=ex)
    _check(e)


def test_average_active_signals():
    sig = pd.Series(np.ones(N), index=NAIVE)
    t1 = pd.Series(AWARE + pd.Timedelta(days=2), index=AWARE)
    with pytest.raises(ValueError) as e:
        average_active_signals(sig, t1)
    _check(e)


def test_object_aware_index_same_instants():
    # An object index of aware Timestamps has no .tz, so it was read as naive and the same instants were refused.
    utc = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
    tokyo_obj = pd.Index(list(utc.tz_convert("Asia/Tokyo")), dtype=object)
    ends = pd.Series(utc + pd.Timedelta(days=2), index=utc)
    sig = np.arange(12.0)
    assert np.allclose(
        average_active_signals(pd.Series(sig, index=tokyo_obj), ends).to_numpy(),
        average_active_signals(pd.Series(sig, index=utc), ends).to_numpy(),
    )
    assert _same_index(tokyo_obj, utc)
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(12, 2)), index=utc)
    y = pd.Series(np.arange(12) % 2, index=utc)
    y_obj = pd.Series(y.to_numpy(), index=tokyo_obj)
    make = lambda: SequentialBootstrapBagging(_FrequencyClassifier, n_estimators=2, seed=1)
    assert np.allclose(
        make().fit(X, y_obj, t1=ends).predict_proba(X), make().fit(X, y, t1=ends).predict_proba(X)
    )


def test_object_aware_index_names_its_zone():
    # Naive against an object index of Tokyo Timestamps: the refusal must name Tokyo, not call it naive.
    naive = pd.date_range("2020-01-01", periods=12, freq="D")
    tokyo_obj = pd.Index(list(naive.tz_localize("Asia/Tokyo")), dtype=object)
    with pytest.raises(ValueError, match="Asia/Tokyo"):
        average_active_signals(pd.Series(np.arange(12.0), index=naive), pd.Series(naive, index=tokyo_obj))
    assert "Asia/Tokyo" in (tz_mismatch_message(naive, tokyo_obj, "a", "b") or "")


def test_kyle_lambda_same_instants_two_resolutions():
    # Same instants in another zone and at a different unit (s against the us default) must align.
    rng = np.random.default_rng(6)
    idx = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
    px = pd.Series(100 + rng.normal(size=12).cumsum(), index=idx)
    vol = pd.Series(rng.integers(1, 9, 12).astype(float), index=idx)
    vol_berlin_s = pd.Series(vol.to_numpy(), index=idx.tz_convert("Europe/Berlin").as_unit("s"))
    assert kyle_lambda(px, vol_berlin_s) == pytest.approx(kyle_lambda(px, vol), abs=1e-12)


def test_cusum_test_same_instants_two_resolutions():
    idx = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    y = pd.Series(rng.normal(size=40), index=idx)
    x = pd.DataFrame({"f": rng.normal(size=40)}, index=idx)
    x_berlin_s = x.set_axis(idx.tz_convert("Europe/Berlin").as_unit("s"))
    assert np.allclose(
        cusum_test(y, x_berlin_s).statistic, cusum_test(y, x).statistic, atol=1e-12, equal_nan=True
    )


def test_same_instants_two_units_accepted():
    # Prices at ns and volumes at s on the same UTC instants name the same bars; 2.3.3 refused them.
    idx = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    px = pd.Series(100 + np.arange(120.0) * 0.1, index=idx)
    expected = volume_bars(px, pd.Series(np.full(120, 10.0), index=idx), 40.0)
    vol_s = pd.Series(np.full(120, 10.0), index=idx.as_unit("s"))
    pd.testing.assert_frame_equal(volume_bars(px, vol_s, 40.0), expected)


def test_same_instants_two_zones_accepted():
    # The same instants in Asia/Tokyo name the same bars as UTC.
    idx = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    px = pd.Series(100 + np.arange(120.0) * 0.1, index=idx)
    expected = volume_bars(px, pd.Series(np.full(120, 10.0), index=idx), 40.0)
    vol_tokyo = pd.Series(np.full(120, 10.0), index=idx.tz_convert("Asia/Tokyo"))
    pd.testing.assert_frame_equal(volume_bars(px, vol_tokyo, 40.0), expected)


def test_mixed_zone_object_t1_spans_match_utc():
    # Labels from two zones concatenate to an object index or values; they must count as the UTC instants.
    grid = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    starts = grid[::3][:30]
    ends = starts + pd.Timedelta(hours=4)
    expected = num_co_events(grid, pd.Series(ends, index=starts))
    half = 15
    by_index = pd.concat(
        [pd.Series(ends[:half], index=starts[:half]), pd.Series(ends[half:], index=starts[half:].tz_convert("Asia/Tokyo"))]
    )
    assert by_index.index.dtype == object
    pd.testing.assert_series_equal(num_co_events(grid, by_index), expected)
    by_values = pd.concat(
        [pd.Series(ends[:half], index=starts[:half]), pd.Series(ends[half:].tz_convert("Asia/Tokyo"), index=starts[half:])]
    )
    assert by_values.dtype == object
    pd.testing.assert_series_equal(num_co_events(grid, by_values), expected)


def _zones_mixed(times):
    # Every other timestamp in Asia/Tokyo: the same instants, two zones in one object array.
    return [t if k % 2 == 0 else t.tz_convert("Asia/Tokyo") for k, t in enumerate(times)]


def test_mixed_zone_t_events_accepted():
    # Event starts from two zones, and mixed-zone vertical barriers, must give the all-UTC results.
    grid = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    close = pd.Series(100 + np.arange(120.0) * 0.1, index=grid)
    events = grid[::3][:30]
    expiry_utc = add_vertical_barrier(events, close, num_days=0.1)
    assert list(add_vertical_barrier(_zones_mixed(events), close, num_days=0.1)) == list(expiry_utc)
    target = pd.Series(0.02, index=events)
    expected_events = get_events(close, events, [1.0, 1.0], target, 0.0, vertical_barrier_times=expiry_utc)
    mixed_vb = pd.Series(pd.Index(_zones_mixed(expiry_utc.tolist()), dtype=object), index=events)
    pd.testing.assert_frame_equal(
        get_events(close, events, [1.0, 1.0], target, 0.0, vertical_barrier_times=mixed_vb), expected_events
    )
    mixed_events = get_events(close, pd.Index(_zones_mixed(events), dtype=object), [1.0, 1.0], target, 0.0)
    assert list(mixed_events["t1"]) == list(get_events(close, events, [1.0, 1.0], target, 0.0)["t1"])


def test_mixed_zone_get_bins_t1_accepted():
    grid = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    close = pd.Series(100 + np.arange(120.0) * 0.1, index=grid)
    events = grid[::3][:30]
    t1_utc = events + pd.Timedelta(hours=4)
    expected = get_bins(pd.DataFrame({"t1": t1_utc, "trgt": 0.02}, index=events), close)
    mixed = pd.DataFrame({"t1": pd.Index(_zones_mixed(t1_utc), dtype=object), "trgt": 0.02}, index=events)
    pd.testing.assert_frame_equal(get_bins(mixed, close), expected)


def test_object_aware_close_keeps_zone():
    # An object index of aware Timestamps must give the zone back, as the DatetimeIndex close does.
    grid = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    vals = 100 + np.arange(120.0) * 0.1
    events = grid[::3][:30]
    expected = add_vertical_barrier(events, pd.Series(vals, index=grid), num_days=0.1)
    obj_close = pd.Series(vals, index=pd.Index(list(grid), dtype=object))
    result = add_vertical_barrier(events, obj_close, num_days=0.1)
    assert result.dtype == expected.dtype
    assert str(result.dt.tz) == "UTC"
    assert list(result) == list(expected)


def test_object_aware_close_accepted_by_target_and_barrier():
    # Object indexes of aware Timestamps in the close zone must pass the zone check, as DatetimeIndex does.
    grid = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    close = pd.Series(100 + np.arange(120.0) * 0.1, index=grid)
    events = grid[::3][:30]
    obj = lambda idx: pd.Index(list(idx), dtype=object)
    expected = get_events(
        close, events, [1.0, 1.0], pd.Series(0.02, index=events), 0.0,
        vertical_barrier_times=pd.Series(events + pd.Timedelta(hours=3), index=events),
    )
    result = get_events(
        close, events, [1.0, 1.0], pd.Series(0.02, index=obj(events)), 0.0,
        vertical_barrier_times=pd.Series(events + pd.Timedelta(hours=3), index=obj(events)),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_object_aware_grid_accepted_by_sample_weight_by_return():
    grid = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC").as_unit("ns")
    close = pd.Series(100 + np.arange(120.0) * 0.1, index=grid)
    starts = grid[::3][:30]
    t1 = pd.Series(starts + pd.Timedelta(hours=4), index=starts)
    expected = sample_weight_by_return(grid, t1, close)
    result = sample_weight_by_return(pd.Index(list(grid), dtype=object), t1, close)
    pd.testing.assert_series_equal(result, expected)


def test_same_zone_different_tz_class_accepted():
    # Asia/Tokyo from zoneinfo and from dateutil are the same zone; the target must not be refused.
    from dateutil import tz as dateutil_tz

    grid = pd.date_range("2024-01-01", periods=120, freq="h", tz="Asia/Tokyo").as_unit("ns")
    close = pd.Series(100 + np.arange(120.0) * 0.1, index=grid)
    events = grid[::3][:30]
    expected = get_events(close, events, [1.0, 1.0], pd.Series(0.02, index=events), 0.0)
    dateutil_events = events.tz_convert(dateutil_tz.gettz("Asia/Tokyo"))
    result = get_events(close, dateutil_events, [1.0, 1.0], pd.Series(0.02, index=dateutil_events), 0.0)
    assert list(result["t1"]) == list(expected["t1"])


@pytest.mark.filterwarnings("ignore:the last time-bar interval")
def test_single_zone_object_index_accepted():
    # One zone held in an object index (UTC Timestamps) is a single zone, not a mixed one.
    idx = pd.date_range("2024-01-01 10:00:10", periods=40, freq="17s", tz="UTC")
    px = pd.Series(100 + np.arange(40.0) * 0.1, index=idx)
    obj = pd.Series(px.to_numpy(), index=pd.Index(list(idx), dtype=object))
    pd.testing.assert_frame_equal(time_bars(obj, freq="1min"), time_bars(px, freq="1min"))
