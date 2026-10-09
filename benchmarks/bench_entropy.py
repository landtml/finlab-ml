"""Benchmark finlab.entropy against literal naive loops (not run by CI).

Usage: PYTHONPATH=src python3 benchmarks/bench_entropy.py
"""

from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np

from finlab.entropy import kontoyiannis_entropy, lempel_ziv_parse, plug_in_entropy


def best_of(fn, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def naive_plug_in(msg: list[int], w: int) -> float:
    lib = Counter(tuple(msg[i : i + w]) for i in range(len(msg) - w + 1))
    total = len(msg) - w + 1
    return -sum((c / total) * math.log2(c / total) for c in lib.values()) / w


def naive_match_length(msg: str, i: int, n: int) -> int:
    sub = ""
    for length in range(n):
        msg1 = msg[i : i + length + 1]
        for j in range(i - n, i):
            if msg1 == msg[j : j + length + 1]:
                sub = msg1
                break
    return len(sub) + 1


def naive_konto(msg: str) -> float:
    total, num = 0.0, 0
    for i in range(1, len(msg) // 2 + 1):
        total += math.log2(i + 1) / naive_match_length(msg, i, i)
        num += 1
    return total / num


def naive_lz(msg: list[int]) -> list[tuple[int, ...]]:
    i, lib = 1, [tuple(msg[:1])]
    while i < len(msg):
        for j in range(i, len(msg)):
            token = tuple(msg[i : j + 1])
            if token not in lib:
                lib.append(token)
                break
        i = j + 1
    return lib


def main() -> None:
    rng = np.random.default_rng(0)

    # Warm up all jitted kernels so compilation is excluded from timings.
    warm = rng.integers(0, 3, 200)
    plug_in_entropy(warm, 2)
    kontoyiannis_entropy(warm)
    kontoyiannis_entropy(warm, window=30)
    lempel_ziv_parse(warm)

    print("entropy: naive reference vs numba kernels (best of 3)")
    print(f"{'routine':<30}{'n':>7}{'naive s':>12}{'jit s':>12}{'speedup':>10}")

    msg = rng.integers(0, 4, 200_000).tolist()
    t_naive = best_of(lambda: naive_plug_in(msg, 2), repeat=1)
    t_jit = best_of(lambda: plug_in_entropy(msg, 2))
    print(f"{'plug_in_entropy (w=2)':<30}{len(msg):>7}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    bits = rng.integers(0, 2, 300)
    text = "".join(map(str, bits))
    t_naive = best_of(lambda: naive_konto(text), repeat=1)
    t_jit = best_of(lambda: kontoyiannis_entropy(bits))
    print(f"{'kontoyiannis (expanding)':<30}{300:>7}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    seq = rng.integers(0, 3, 3000).tolist()
    t_naive = best_of(lambda: naive_lz(seq), repeat=1)
    t_jit = best_of(lambda: lempel_ziv_parse(seq))
    print(f"{'lempel_ziv_parse':<30}{3000:>7}{t_naive:>12.4f}{t_jit:>12.6f}{t_naive / t_jit:>9.0f}x")

    print("\njit-only timing for larger inputs (no naive run):")
    big = rng.integers(0, 2, 5000)
    print(f"kontoyiannis expanding n=5000: {best_of(lambda: kontoyiannis_entropy(big), repeat=1):.3f} s")


if __name__ == "__main__":
    main()
