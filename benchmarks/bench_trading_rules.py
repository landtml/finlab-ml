"""Benchmark: numba OTR grid vs a pure-Python loop over the same paths.

The naive loop is the transcription in ``quantowl.trading_rules``, run on the same
normals the kernel draws. The kernel draws its normals inside the timed call. The
naive loop reads normals drawn before timing, so its timing leaves out the draws.
The reported speedup is therefore a lower bound for a naive loop that draws its
own normals. The script also checks that both give the same means.

Not run by CI. Usage:
    PYTHONPATH=src python3 benchmarks/bench_trading_rules.py
"""

from __future__ import annotations

import numpy as np

from _timing import best_of
from quantowl.trading_rules import (
    _otr_exits_python,
    _pair_seeds,
    _standard_normals,
    otr_grid,
    ou_phi,
)

FORECAST = 0.0
HALF_LIFE = 5.0
SIGMA = 1.0
MAX_HP = 100
SEED = 0


def draw_normals(levels: np.ndarray, n_iter: int) -> list[np.ndarray]:
    """The normals each pair of the grid consumes, drawn with the kernel's seeds."""
    n_pairs = levels.size * levels.size
    seeds = _pair_seeds(SEED, n_pairs)
    return [_standard_normals(seeds[k], n_iter * (MAX_HP + 1)) for k in range(n_pairs)]


def naive_grid_means(levels: np.ndarray, n_iter: int, normals: list[np.ndarray]) -> np.ndarray:
    """Mean exit value per pair, from the pure-Python loop on the given normals."""
    phi = ou_phi(HALF_LIFE)
    pairs = [(pt, sl) for pt in levels for sl in levels]
    means = np.empty(len(pairs))
    for k, (pt, sl) in enumerate(pairs):
        exits, _ = _otr_exits_python(
            phi, FORECAST, SIGMA, 0.0, pt * SIGMA, sl * SIGMA, MAX_HP, n_iter, normals[k]
        )
        means[k] = exits.mean()
    return means


def main() -> None:
    n_iter = 2000
    levels = np.linspace(0.0, 4.0, 5)  # 5 x 5 grid

    def run_numba() -> np.ndarray:
        grid = otr_grid(
            FORECAST, HALF_LIFE, sigma=SIGMA, profit_taking=levels, stop_loss=levels,
            n_iter=n_iter, max_hp=MAX_HP, seed=SEED,
        )
        return grid["mean"].to_numpy()

    # The first call compiles the kernel, so it is outside the timing.
    otr_grid(FORECAST, HALF_LIFE, profit_taking=levels, stop_loss=levels, n_iter=2, seed=SEED)

    normals = draw_normals(levels, n_iter)  # outside the naive timing, as the module docstring says
    numba_means = run_numba()
    naive_means = naive_grid_means(levels, n_iter, normals)
    agree = bool(np.allclose(numba_means, naive_means, rtol=1e-12, atol=1e-15))

    numba_t = best_of(run_numba)
    naive_t = best_of(lambda: naive_grid_means(levels, n_iter, normals), repeats=1)

    print(f"grid {levels.size} x {levels.size}, n_iter={n_iter}, max_hp={MAX_HP}, half_life={HALF_LIFE}")
    print(f"{'naive (s)':>10} {'numba (s)':>10} {'speedup':>8} {'means agree':>12}")
    print(f"{naive_t:10.4f} {numba_t:10.4f} {naive_t / numba_t:7.1f}x {str(agree):>12}")


if __name__ == "__main__":
    main()
