"""Tests for finlab.onc. ONC is from López de Prado, Lewis and Boudt (2019), not AFML."""
import numpy as np
import pandas as pd
import pytest

from finlab.onc import onc, silhouette_scores


def _block_corr(sizes, within=0.9, between=0.05, seed=0):
    """Correlation matrix with clear blocks and small noise, shuffled."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    c = np.where(truth[:, None] == truth[None, :], within, between).astype(float)
    noise = rng.normal(0, 0.02, size=(n, n))
    noise = (noise + noise.T) / 2
    c = np.clip(c + noise, -0.99, 0.99)
    np.fill_diagonal(c, 1.0)
    perm = rng.permutation(n)
    return c[np.ix_(perm, perm)], truth[perm]


def _naive_silhouette(d, labels):
    n = len(labels)
    out = np.zeros(n)
    for i in range(n):
        own = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not own:
            continue
        a = np.mean([d[i, j] for j in own])
        b = min(
            np.mean([d[i, j] for j in range(n) if labels[j] == c])
            for c in set(labels) if c != labels[i]
        )
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return out


def _same_partition(a, b):
    """True when two label vectors describe the same partition."""
    mapping = {}
    for x, y in zip(a, b):
        if mapping.setdefault(x, y) != y:
            return False
    return len(set(mapping.values())) == len(mapping)


def test_recovers_block_structure():
    corr, truth = _block_corr([4, 3, 5], seed=1)
    res = onc(corr, seed=0)
    assert res.n_clusters == 3
    assert _same_partition(res.labels, truth)


def test_silhouette_matches_naive_reference():
    corr, _ = _block_corr([3, 3, 4], seed=2)
    d = np.sqrt(0.5 * (1 - corr))
    labels = np.array([0, 1, 2] * 3 + [0, 1, 2, 0])[: corr.shape[0]]
    np.testing.assert_allclose(silhouette_scores(d, labels), _naive_silhouette(d, labels), atol=1e-12)


def test_silhouette_zero_for_singletons():
    d = np.sqrt(0.5 * (1 - np.eye(4) + 0.3 * (1 - np.eye(4))))
    s = silhouette_scores(d, [0, 1, 2, 3])
    np.testing.assert_array_equal(s, np.zeros(4))


def test_deterministic_for_fixed_seed():
    corr, _ = _block_corr([3, 4, 3], seed=3)
    a = onc(corr, seed=7)
    b = onc(corr, seed=7)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.scores == b.scores


def test_accepts_dataframe_and_returns_scores_for_each_k():
    corr, _ = _block_corr([3, 3, 3], seed=4)
    df = pd.DataFrame(corr, columns=[f"v{i}" for i in range(9)], index=[f"v{i}" for i in range(9)])
    res = onc(df, max_k=5, seed=0)
    assert sorted(res.scores) == [2, 3, 4, 5]


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((3, 2)),
        np.array([[1, 0.2, 0], [0.1, 1, 0], [0, 0, 1]]),  # asymmetric
        np.array([[0.9, 0.2, 0], [0.2, 1, 0], [0, 0, 1]]),  # non-unit diagonal
        np.array([[1, 2.0, 0], [2.0, 1, 0], [0, 0, 1]]),  # out of range
        np.eye(2),  # too few variables
    ],
)
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        onc(bad)


def test_max_k_out_of_range_rejected():
    corr, _ = _block_corr([3, 3, 3], seed=5)
    with pytest.raises(ValueError):
        onc(corr, max_k=9)
