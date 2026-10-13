"""Reproduce the sequential-versus-standard bootstrap uniqueness means with finlab.weights.

Not run by CI. Usage:
    PYTHONPATH=src python3 benchmarks/repro_uniqueness.py

The design is the one of ``test_sequential_bootstrap_has_higher_average_uniqueness_than_standard``
in ``tests/test_weights.py``: 200 bars, 60 labels with spans of up to 30 bars, labels drawn
with seed 50. The proof note (docs/proofs/weights.md) states only "200 seeds on the same design".
It does not state the seed ranges or the sample size, so these are chosen here:

- sample size: 60 draws, the number of labels (the book's default);
- sequential bootstrap: seeds 0 to 199;
- standard bootstrap: ``numpy.random.default_rng(1000 + seed)``, seeds 0 to 199.

The script prints the mean and the standard deviation of the sample average uniqueness
over the 200 seeds for each method. Measured on 2026-10-13 on a shared 4-vCPU machine, it
printed means of 0.1963 (sequential) and 0.1913 (standard), with standard deviations of
0.0128 and 0.0146.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finlab.weights import indicator_matrix, sample_average_uniqueness, sequential_bootstrap

N_BARS = 200
N_LABELS = 60
MAX_SPAN = 30
LABEL_SEED = 50
N_SEEDS = 200


def random_labels(
    n_bars: int, n_labels: int, max_span: int, seed: int
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Label start times (index) and end times (values) on a business-day grid, and the grid."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-01", periods=n_bars)
    starts = np.sort(rng.choice(n_bars - max_span - 1, size=n_labels, replace=False))
    spans = rng.integers(0, max_span, size=n_labels)
    ends = np.minimum(starts + spans, n_bars - 1)
    return pd.Series(index[ends], index=index[starts]), index


def main() -> None:
    t1, index = random_labels(N_BARS, N_LABELS, MAX_SPAN, LABEL_SEED)
    m = indicator_matrix(index, t1).to_numpy()
    n = m.shape[1]

    seq_scores = np.empty(N_SEEDS)
    std_scores = np.empty(N_SEEDS)
    for seed in range(N_SEEDS):
        seq = sequential_bootstrap(m, n_samples=n, seed=seed)
        seq_scores[seed] = sample_average_uniqueness(m, seq)
        std = np.random.default_rng(1000 + seed).integers(0, n, size=n)
        std_scores[seed] = sample_average_uniqueness(m, std)

    print(f"design: {N_BARS} bars, {n} labels, spans below {MAX_SPAN} bars, "
          f"label seed {LABEL_SEED}, {N_SEEDS} seeds, sample size {n}")
    print(f"sequential bootstrap: mean {seq_scores.mean():.4f}  sd {seq_scores.std(ddof=1):.4f}")
    print(f"standard bootstrap:   mean {std_scores.mean():.4f}  sd {std_scores.std(ddof=1):.4f}")
    print(f"difference (sequential - standard): {seq_scores.mean() - std_scores.mean():.4f}")


if __name__ == "__main__":
    main()
