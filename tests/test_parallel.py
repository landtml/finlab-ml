"""Tests for finlab.parallel (AFML ch. 20)."""
import numpy as np
import pandas as pd
import pytest

from finlab.parallel import lin_parts, mp_pandas_obj, nested_parts


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
