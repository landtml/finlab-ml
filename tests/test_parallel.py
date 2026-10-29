"""Tests for quantowl.parallel (AFML ch. 20)."""
import sys
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

from quantowl.parallel import lin_parts, mp_pandas_obj, nested_parts


def _square(molecule, offset=0):
    return pd.Series(np.asarray(molecule, dtype=float) ** 2 + offset, index=molecule)


@pytest.mark.parametrize("n,t", [(10, 3), (7, 7), (100, 4), (3, 8)])
def test_lin_parts_cover_range_monotonically(n, t):
    parts = lin_parts(n, t)
    assert parts[0] == 0 and parts[-1] == n
    assert np.all(np.diff(parts) >= 0)


@pytest.mark.parametrize("n,t", [(10, 3), (50, 4), (200, 8)])
def test_nested_parts_cover_range_monotonically(n, t):
    for upper in (False, True):
        parts = nested_parts(n, t, upper_triangle=upper)
        assert parts[0] == 0 and parts[-1] == n
        assert np.all(np.diff(parts) > 0)


def test_serial_and_parallel_agree():
    idx = pd.RangeIndex(1000)
    serial = mp_pandas_obj(_square, idx, num_threads=1, offset=1)
    parallel = mp_pandas_obj(_square, idx, num_threads=3, offset=1)
    pd.testing.assert_series_equal(serial, parallel)
    assert len(serial) == 1000
    # Equality alone would also pass if both paths returned the same wrong values.
    assert serial.index.equals(idx)
    np.testing.assert_array_equal(serial.to_numpy(), np.arange(1000.0) ** 2 + 1)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="fork DeprecationWarning starts in 3.12")
def test_parallel_call_emits_no_deprecation_warning():
    # CPython warns on fork() while another thread is alive, so keep one running
    # while the pool starts. Without it the old start method passes this test.
    stop = threading.Event()
    helper = threading.Thread(target=stop.wait)
    helper.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parallel = mp_pandas_obj(_square, pd.RangeIndex(200), num_threads=2, offset=1)
    finally:
        stop.set()
        helper.join()
    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, deprecations
    np.testing.assert_array_equal(parallel.to_numpy(), np.arange(200.0) ** 2 + 1)
