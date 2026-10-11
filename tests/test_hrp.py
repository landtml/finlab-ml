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


def test_hrp_single_asset_still_checks_corr_shape() -> None:
    with pytest.raises(ValueError, match="same shape"):
        hrp_weights(np.array([[0.04]]), corr=np.ones((3, 3)))
    with pytest.raises(ValueError, match="same shape"):
        hrp_weights(np.array([[0.04]]), corr=np.ones((1, 2)))


def test_hrp_single_asset_and_bad_inputs() -> None:
    assert hrp_weights(np.array([[2.0]])).tolist() == [1.0]
    with pytest.raises(ValueError):
        hrp_weights(np.array([[0.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        hrp_weights(np.eye(3), method="ward")


def test_quasi_diagonalize_single_row_must_merge_leaves() -> None:
    # Child ids 3 and 5 are not leaves of a two-leaf tree and not earlier rows.
    link = np.array([[3.0, 5.0, 0.5230936064205157, 2.0]])
    with pytest.raises(ValueError, match="child"):
        quasi_diagonalize(link)


def test_quasi_diagonalize_rejects_child_id_out_of_range() -> None:
    with pytest.raises(ValueError, match="child"):
        quasi_diagonalize(np.array([[0, 5, 1, 2]], dtype=float))


def test_quasi_diagonalize_rejects_cyclic_link_without_hanging() -> None:
    # Node 4 is the root and lists itself as a child; validation must raise before any walk.
    with pytest.raises(ValueError, match="child"):
        quasi_diagonalize(np.array([[0, 1, 1, 2], [4, 2, 1, 3]], dtype=float))


def test_quasi_diagonalize_rejects_child_used_twice() -> None:
    with pytest.raises(ValueError, match="more than once"):
        quasi_diagonalize(np.array([[0, 1, 1, 2], [0, 2, 1, 3]], dtype=float))


def test_quasi_diagonalize_valid_one_merge_is_unchanged() -> None:
    assert quasi_diagonalize(np.array([[0.0, 1.0, 0.7, 2.0]])) == [0, 1]


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_quasi_diagonalize_rejects_non_finite_link(bad) -> None:
    link = np.array(
        [
            [3.0, 5.0, 0.52, 2.0],
            [1.0, 2.0, 0.55, 2.0],
            [4.0, 6.0, 0.56, 3.0],
            [7.0, 8.0, 0.68, 5.0],
            [bad, 9.0, 0.69, 6.0],
        ]
    )
    with pytest.raises(ValueError, match="link"):
        quasi_diagonalize(link)


def test_hrp_rejects_non_psd_covariance() -> None:
    # This matrix has a negative eigenvalue; weights came out negative.
    from finlab.hrp import hrp_weights

    C = np.array(
        [
            [1.743, 0.433, 0.139, -0.184],
            [0.433, 0.536, 0.199, -0.872],
            [0.139, 0.199, 0.509, -0.718],
            [-0.184, -0.872, -0.718, 0.841],
        ]
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        hrp_weights(C)


def test_hrp_rejects_asymmetric_covariance() -> None:
    from finlab.hrp import hrp_weights

    C = np.array([[1.0, 0.5], [0.1, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        hrp_weights(C)


def test_hrp_singular_psd_covariance_still_accepted() -> None:
    # Rank-deficient but PSD: float noise on the zero eigenvalues must not trigger the check.
    from finlab.hrp import hrp_weights

    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 2))
    C = A @ A.T  # rank 2 out of 6, so four eigenvalues are zero up to float noise
    w = hrp_weights(C)
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("bad", [np.inf, np.nan, -2.0, 1.5])
def test_correlation_distance_rejects_non_finite_or_out_of_range(bad: float) -> None:
    # +inf mapped to distance 0, -2 was clipped silently, and NaN passed through.
    corr = np.array([[1.0, bad], [bad, 1.0]])
    with pytest.raises(ValueError, match="corr"):
        correlation_distance(corr)


def test_hrp_weights_nan_corr_fails_naming_corr() -> None:
    # NaN used to fail late inside scipy with a message that did not name corr.
    from finlab.hrp import hrp_weights

    C = np.diag([0.04, 0.09, 0.0625])
    R = np.array([[1.0, np.nan, 0.1], [np.nan, 1.0, 0.1], [0.1, 0.1, 1.0]])
    with pytest.raises(ValueError, match="corr"):
        hrp_weights(C, corr=R)


def test_hrp_weights_rejects_asymmetric_corr() -> None:
    # Only the upper triangle of corr was read, so lower-triangle edits did nothing.
    from finlab.hrp import hrp_weights

    C = np.diag([0.04, 0.09, 0.0625, 0.25])
    Ra = np.eye(4)
    Ra[0, 1] = Ra[2, 3] = 0.99
    Ra[2, 0] = Ra[3, 1] = 0.99
    with pytest.raises(ValueError, match="corr must be symmetric"):
        hrp_weights(C, corr=Ra)


def test_hrp_weights_weights_nonnegative_for_tiny_negative_eigenvalue() -> None:
    # A cluster variance could come out negative, pushing alpha above 1 and giving negative weights,
    # against the documented long-only contract.
    from finlab.hrp import hrp_weights

    b = -1 - 1e-10
    C = np.array([[1.0, b, 0, 0], [b, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    R = np.array([[1, 0.99, 0.1, 0.1], [0.99, 1, 0.1, 0.1], [0.1, 0.1, 1, 0.5], [0.1, 0.1, 0.5, 1.0]])
    w = hrp_weights(C, corr=R)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0)


def test_hrp_weights_single_asset_rejects_out_of_range_corr() -> None:
    # The N == 1 early return skipped the corr range check that runs for N >= 2, so an out-of-range
    # corr was accepted for one asset.
    with pytest.raises(ValueError, match=r"corr entries must lie in \[-1, 1\]"):
        hrp_weights(np.array([[2.0]]), np.array([[5.0]]))
    assert hrp_weights(np.array([[2.0]]), np.array([[1.0]])).tolist() == [1.0]


def test_hrp_weights_refuses_string_covariance_entries() -> None:
    # '1' and '0' were coerced to floats, so this gave [0.5, 0.5].
    from finlab.hrp import hrp_weights

    with pytest.raises(ValueError, match="must be real numbers"):
        hrp_weights([["1", "0"], ["0", "1"]])



def test_correlation_distance_refuses_non_unit_diagonal() -> None:
    # The diagonal was never checked, so self-distances came out as 0.63 and 0.5.
    from finlab.hrp import correlation_distance

    with pytest.raises(ValueError, match="unit diagonal"):
        correlation_distance([[0.5, 0.2], [0.2, 0.5]])
    d = correlation_distance([[1.0, 0.2], [0.2, 1.0]])
    assert np.all(np.diag(d) == 0.0)



def test_hrp_weights_refuses_partially_overlapping_corr_labels() -> None:
    # corr.loc[labels, labels] raised a KeyError that did not name the mismatch.
    from finlab.hrp import hrp_weights

    cov = pd.DataFrame([[1, 0.2], [0.2, 1]], index=["a", "b"], columns=["a", "b"])
    corr = pd.DataFrame([[1, 0.1], [0.1, 1]], index=["x", "y"], columns=["x", "y"])
    with pytest.raises(ValueError, match="missing labels"):
        hrp_weights(cov, corr=corr)


def test_hrp_weights_pin_refuses_clearly_non_psd_and_asymmetric_cov() -> None:
    # A real negative eigenvalue and a real asymmetry are still refused.
    from finlab.hrp import hrp_weights

    with pytest.raises(ValueError, match="positive semidefinite"):
        hrp_weights(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        hrp_weights(np.array([[4.0, 1.0], [1.001, 9.0]]))


def test_hrp_weights_accepts_float32_rounded_singular_cov() -> None:
    # A rank-deficient covariance (3 observations, 6 assets) stored in float32 has a smallest
    # eigenvalue of -2.6e-12, past the 1e-9 PSD floor, so a valid input was refused.
    from finlab.hrp import hrp_weights

    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, 6)) * 0.01
    C = np.cov(X, rowvar=False).astype(np.float32).astype(np.float64)
    w = hrp_weights(C)
    assert np.all(w >= 0.0)
    assert float(np.sum(w)) == pytest.approx(1.0)


def test_hrp_weights_accepts_float32_rounded_symmetry() -> None:
    # An asymmetry of 1e-8 relative (float32 storage) was refused by the 1e-9 rtol.
    from finlab.hrp import hrp_weights

    w = hrp_weights(np.array([[4.0, 1.0], [1.0 + 1e-8, 9.0]]))
    assert w == pytest.approx([0.69230769, 0.30769231], abs=1e-6)


def test_hrp_weights_refuses_non_unit_corr_diagonal_for_one_asset() -> None:
    # N = 1 skipped the unit-diagonal check that N = 2 runs through correlation_distance.
    from finlab.hrp import hrp_weights

    with pytest.raises(ValueError, match="unit diagonal"):
        hrp_weights(np.array([[0.04]]), corr=np.array([[0.5]]))
    with pytest.raises(ValueError, match="unit diagonal"):
        hrp_weights(np.array([[0.04, 0.01], [0.01, 0.09]]), corr=np.array([[0.5, 0.2], [0.2, 0.5]]))
    assert hrp_weights(np.array([[0.04]]), corr=np.array([[1.0]])).tolist() == [1.0]


def test_hrp_weights_refuses_non_psd_correlation_at_any_scale() -> None:
    # The PSD test used max|C| as its scale, so the refusal depended on the units of the assets. A
    # non-PSD correlation structure is refused at every scale.
    from finlab.hrp import hrp_weights

    R = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        hrp_weights(R)
    s = np.array([1e4, 1.0, 1e-4])
    with pytest.raises(ValueError, match="positive semidefinite"):
        hrp_weights(np.diag(s) @ R @ np.diag(s))


def test_hrp_weights_pin_psd_refusal_and_float32_singular_cov_at_scale() -> None:
    # A non-PSD matrix is still refused, and a float32-stored singular covariance is still accepted
    # when its assets are rescaled, so the normalized tolerance keeps both behaviours.
    from finlab.hrp import hrp_weights

    C = np.array(
        [
            [1.743, 0.433, 0.139, -0.184],
            [0.433, 0.536, 0.199, -0.872],
            [0.139, 0.199, 0.509, -0.718],
            [-0.184, -0.872, -0.718, 0.841],
        ]
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        hrp_weights(C)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, 6)) * 1e-4
    Cs = np.cov(X, rowvar=False).astype(np.float32).astype(np.float64)
    w = hrp_weights(Cs)
    assert np.all(w >= 0.0)
    assert float(np.sum(w)) == pytest.approx(1.0)


def test_correlation_distance_refuses_out_of_range_and_non_finite_after_tolerance() -> None:
    # A real out-of-range entry, inf and NaN are still refused, at the wider tolerance too.
    from finlab.hrp import correlation_distance

    for bad in (-2.0, 2.0, 1.01, np.inf, np.nan):
        with pytest.raises(ValueError):
            correlation_distance(np.array([[1.0, bad], [bad, 1.0]]))
def test_hrp_weights_accepts_float32_rounded_correlation() -> None:
    # A valid correlation stored in float32 has an entry of 1.0000001192 and a diagonal of
    # 0.99999988. The range and diagonal checks at 1e-9 refused it, while the covariance checks
    # already accept float32 rounding.
    from finlab.hrp import hrp_weights

    X = np.random.default_rng(7).standard_normal((60, 4)).astype(np.float32)
    C = np.cov(X, rowvar=False).astype(np.float32)
    s = np.sqrt(np.diag(C))
    R = (C / np.outer(s, s)).astype(np.float32)
    assert float(R.max()) > 1.0 + 1e-9
    w = hrp_weights(C.astype(np.float64), corr=R)
    w64 = hrp_weights(C.astype(np.float64), corr=R.astype(np.float64))
    assert w == pytest.approx(w64, abs=1e-5)
    assert float(np.sum(w)) == pytest.approx(1.0)


