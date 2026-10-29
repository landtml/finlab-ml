"""Test setup shared by every test module.

Numba caches compiled kernels on disk, and a cached caller can keep a callee it
embedded from an outdated source file. A stale entry then fails in ways that
depend on test order. So by default the suite gets a new, empty cache directory
on every run. Set ``QUANTOWL_TEST_NUMBA_CACHE`` to a directory to reuse one, for
example to check that a warm cache gives the same results.

This must run before numba is imported, so it sets the environment variable at
import time, before any ``quantowl`` module is loaded.
"""

import os
import tempfile

_cache_dir = os.environ.get("QUANTOWL_TEST_NUMBA_CACHE") or tempfile.mkdtemp(prefix="quantowl-numba-")
os.environ["NUMBA_CACHE_DIR"] = _cache_dir
