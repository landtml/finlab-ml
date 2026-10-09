"""Shared numba compilation settings.

Every hot loop in finlab is compiled to machine code with numba. Functions are
compiled lazily on first call and cached on disk (``cache=True``). ``fastmath``
is deliberately *off*: reassociating floating-point operations would make the
results depend on the compiler and the thread count, and the proofs in
``docs/proofs`` assume exact IEEE-754 semantics.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import numba

F = TypeVar("F", bound=Callable)

__all__ = ["jit", "pjit"]


def jit(func: F) -> F:
    """Compile ``func`` in nopython mode with on-disk caching."""
    return numba.njit(cache=True, nogil=True)(func)  # type: ignore[return-value]


def pjit(func: F) -> F:
    """Like :func:`jit`, but parallelised over ``numba.prange`` loops."""
    return numba.njit(cache=True, nogil=True, parallel=True)(func)  # type: ignore[return-value]
