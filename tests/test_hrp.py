"""Tests for finlab.hrp (AFML ch. 16, Hierarchical Risk Parity).

The fast quasi-diagonalization and recursive bisection are cross-checked
against recursive, list-based references on seeded random covariances. The
block-diagonal test checks the cluster-level inverse-variance allocation the
book derives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from finlab.hrp import correlation_distance, hrp_weights, quasi_diagonalize


# ---------------------------------------------------------------------------
# Naive references
# ---------------------------------------------------------------------------
def naive_quasi_diag(link: np.ndarray) -> list[int]:
    n = link.shape[0] + 1

    def expand(node: int) -> list[int]:
        if node < n:
            return [node]
        r = node - n
        return expand(int(link[r, 0])) + expand(int(link[r, 1]))

    return expand(2 * n - 2)


def naive_cluster_var(C: np.ndarray, items: list[int]) -> float:
    inv = np.array([1.0 / C[i, i] for i in items])
    wt = inv / inv.sum()
    sub = C[np.ix_(items, items)]
    return float(wt @ sub @ wt)


def naive_bisection(C: np.ndarray, order: list[int]) -> np.ndarray:
    """Recursive bisection over an explicit list of item indices (book's steps 1-4)."""
    w = {i: 1.0 for i in order}

    def rec(items: list[int]) -> None:
        if len(items) < 2:
            return
        half = len(items) // 2
        left, right = items[:half], items[half:]
        v1 = naive_cluster_var(C, left)
        v2 = naive_cluster_var(C, right)
        alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= 1.0 - alpha
        rec(left)
        rec(right)

    rec(list(order))
    n = C.shape[0]
    return np.array([w[i] for i in range(n)])


def naive_hrp(C: np.ndarray, method: str = "single") -> np.ndarray:
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    D = np.sqrt(0.5 * (1 - np.clip(R, -1, 1)))
    np.fill_diagonal(D, 0.0)
    link = sch.linkage(squareform(D, checks=False), method=method)
    order = naive_quasi_diag(link)
    return naive_bisection(C, order)


def random_cov(rng: np.random.Generator, n: int, T: int) -> np.ndarray:
    X = rng.normal(size=(T, n)) * rng.uniform(0.5, 3.0, size=n)
    return np.cov(X, rowvar=False).reshape(n, n)


# ---------------------------------------------------------------------------
# Distance and ordering
# ---------------------------------------------------------------------------
def test_correlation_distance_book_example_16_1() -> None:
    rho = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, -0.2], [0.2, -0.2, 1.0]])
    expected = np.array(
        [[0.0, 0.3873, 0.6325], [0.3873, 0.0, 0.7746], [0.6325, 0.7746, 0.0]]
    )
    np.testing.assert_allclose(correlation_distance(rho), expected, atol=5e-5)


def test_correlation_distance_range_and_dataframe() -> None:
    df = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    d = correlation_distance(df)
    assert isinstance(d, pd.DataFrame)
    assert d.loc["a", "b"] == pytest.approx(1.0)
    assert d.loc["a", "a"] == 0.0


@pytest.mark.parametrize("seed", range(10))
def test_quasi_diagonalize_is_permutation_and_matches_naive(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 25))
    C = random_cov(rng, n, T=3 * n)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    D = np.sqrt(0.5 * (1 - np.clip(R, -1, 1)))
    np.fill_diagonal(D, 0.0)
    link = sch.linkage(squareform(D, checks=False), "single")
    order = quasi_diagonalize(link)
    assert sorted(order) == list(range(n))  # a permutation of the input indices
    assert order == naive_quasi_diag(link)


def test_quasi_diagonalize_single_item() -> None:
    assert quasi_diagonalize(np.empty((0, 4))) == [0]


# ---------------------------------------------------------------------------
# HRP weights: properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(25))
def test_hrp_weights_nonnegative_and_sum_to_one(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 30))
    # Mix well-conditioned and singular (T < N) covariances.
    T = int(rng.choice([n // 2 + 1, 2 * n, 10 * n]))
    C = random_cov(rng, n, T=max(T, 2))
    w = hrp_weights(C)
    assert w.shape == (n,)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_hrp_matches_naive_reference(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(3, 20))
    C = random_cov(rng, n, T=4 * n)
    for method in ("single", "complete", "average"):
        np.testing.assert_allclose(
            hrp_weights(C, method=method), naive_hrp(C, method), rtol=1e-10, atol=1e-14
        )


def test_hrp_singular_covariance_is_allowed() -> None:
    # Rank 2 covariance on 5 assets: a quadratic optimizer cannot invert it.
    rng = np.random.default_rng(5)
    F = rng.normal(size=(200, 2))
    load = rng.normal(size=(5, 2))
    X = F @ load.T
    C = np.cov(X, rowvar=False)
    assert np.linalg.matrix_rank(C) == 2
    w = hrp_weights(C)
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)


def test_hrp_diagonal_covariance_is_inverse_variance() -> None:
    # AFML Appendix 16.A.2: IVP is optimal for diagonal covariance.
    variances = np.array([1.0, 4.0, 0.25, 9.0])
    w = hrp_weights(np.diag(variances))
    ivp = (1.0 / variances) / np.sum(1.0 / variances)
    np.testing.assert_allclose(w, ivp, rtol=1e-12)


def test_hrp_block_diagonal_matches_cluster_level_inverse_variance() -> None:
    # Two independent blocks of two assets each. Within a block the correlation
    # is 0.5, across blocks it is 0, so single linkage forms the two blocks first.
    # Book's bisection splits the blocks at the top level, then within each block
    # allocates inverse-variance. Hence:
    #   block weight  proportional to 1 / V_b,  V_b = w~' C_b w~,  w~ ~ 1/diag(C_b)
    #   asset weight  = block weight * (1/sigma_i^2) / sum_{j in b} (1/sigma_j^2)
    s1, s2, s3, s4 = 1.0, 2.0, 0.5, 3.0
    rho = 0.5
    blk_a = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    blk_b = np.array([[s3**2, rho * s3 * s4], [rho * s3 * s4, s4**2]])
    C = np.zeros((4, 4))
    C[:2, :2] = blk_a
    C[2:, 2:] = blk_b

    def block_var(B: np.ndarray) -> float:
        iv = 1.0 / np.diag(B)
        wt = iv / iv.sum()
        return float(wt @ B @ wt)

    Va, Vb = block_var(blk_a), block_var(blk_b)
    block_w = np.array([1.0 / Va, 1.0 / Vb])
    block_w /= block_w.sum()
    ivp_a = (1.0 / np.array([s1**2, s2**2])) / np.sum(1.0 / np.array([s1**2, s2**2]))
    ivp_b = (1.0 / np.array([s3**2, s4**2])) / np.sum(1.0 / np.array([s3**2, s4**2]))
    expected = np.concatenate([block_w[0] * ivp_a, block_w[1] * ivp_b])

    w = hrp_weights(C)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(w[:2].sum(), block_w[0], rtol=1e-10)
    np.testing.assert_allclose(w, expected, rtol=1e-10)


def test_hrp_dataframe_in_series_out_with_labels() -> None:
    rng = np.random.default_rng(9)
    C = pd.DataFrame(random_cov(rng, 4, T=50), index=list("abcd"), columns=list("abcd"))
    w = hrp_weights(C)
    assert isinstance(w, pd.Series)
    assert list(w.index) == list("abcd")
    assert w.sum() == pytest.approx(1.0)


def test_hrp_accepts_explicit_correlation() -> None:
    rng = np.random.default_rng(11)
    C = random_cov(rng, 6, T=60)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    np.testing.assert_allclose(hrp_weights(C, corr=R), hrp_weights(C))


def test_hrp_single_asset_and_bad_inputs() -> None:
    assert hrp_weights(np.array([[2.0]])).tolist() == [1.0]
    with pytest.raises(ValueError):
        hrp_weights(np.array([[0.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        hrp_weights(np.eye(3), method="ward")
