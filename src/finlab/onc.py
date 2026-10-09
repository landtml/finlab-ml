"""Optimal Number of Clusters (ONC) for a correlation matrix.

ONC chooses the number of clusters for a correlation-based clustering, and is
used to group related instruments before hierarchical risk parity or feature
clustering. It is **not** part of AFML's text (the book's clustering uses
hierarchical linkage, see :mod:`finlab.hrp`); it is from López de Prado, Lewis
and Boudt (2019), "Clustering with ONC". Cited from that paper, not from AFML.

Method implemented here
-----------------------
1. Distance ``d_ij = sqrt(1/2 * (1 - rho_ij))`` from the correlation matrix.
2. For each candidate ``k`` in ``[2, max_k]``, cluster the *rows of the
   correlation matrix* with k-means (k-means++ seeding, ``n_init`` restarts,
   the best inertia kept).
3. Score each clustering by its mean silhouette on ``d``. Keep the ``k`` with the
   highest mean silhouette.

Simplification
--------------
The published ONC adds a repair step that moves poorly fitting members between
clusters and ranks candidate clusterings by a quality ratio. That step is not
implemented: this version selects ``k`` by mean silhouette only.

Scope
-----
Works on a symmetric correlation matrix with unit diagonal. Output is the cluster
label of each variable, the chosen ``k`` and its silhouette score.

Not covered
-----------
The repair step and quality-ratio ranking from the paper; consensus clustering;
and any claim that ONC recovers true clusters beyond the synthetic tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["ONCResult", "onc", "silhouette_scores"]


@dataclass(frozen=True)
class ONCResult:
    """Outcome of :func:`onc`.

    Attributes
    ----------
    labels : np.ndarray
        Cluster label (``0 .. k-1``) for each variable, in input order.
    n_clusters : int
        The selected number of clusters ``k``.
    silhouette : float
        Mean silhouette of the selected clustering, on the correlation distance.
    scores : dict[int, float]
        Mean silhouette for every candidate ``k`` that was evaluated.
    """

    labels: npt.NDArray[np.int64]
    n_clusters: int
    silhouette: float
    scores: dict[int, float]


def _distance(corr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))


def silhouette_scores(
    dist: npt.NDArray[np.float64], labels: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Per-variable silhouette ``s_i = (b_i - a_i) / max(a_i, b_i)``.

    ``a_i`` is the mean distance from ``i`` to the other members of its cluster,
    ``b_i`` the smallest mean distance from ``i`` to another cluster. A singleton
    cluster has ``s_i = 0`` by convention.
    """
    lab = np.asarray(labels)
    d = np.asarray(dist, dtype=np.float64)
    n = d.shape[0]
    uniq = np.unique(lab)
    if uniq.size < 2:
        return np.zeros(n)
    # Mean distance from each variable to each cluster, vectorized over clusters.
    onehot = (lab[:, None] == uniq[None, :]).astype(np.float64)  # (n, k)
    sizes = onehot.sum(axis=0)  # (k,)
    sums = d @ onehot  # (n, k): total distance from i to cluster c
    own = onehot.argmax(axis=1)
    own_size = sizes[own]
    a = np.where(own_size > 1, sums[np.arange(n), own] / np.maximum(own_size - 1, 1), 0.0)
    means = sums / sizes[None, :]
    means[np.arange(n), own] = np.inf
    b = means.min(axis=1)
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / np.where(denom > 0, denom, 1.0), 0.0)
    s = np.where(own_size > 1, s, 0.0)
    return s


def _kmeans(
    x: npt.NDArray[np.float64], k: int, n_init: int, rng: np.random.Generator, max_iter: int = 300
) -> npt.NDArray[np.int64]:
    """Lloyd's algorithm with k-means++ seeding; best of ``n_init`` restarts."""
    n = x.shape[0]
    best_labels = None
    best_inertia = np.inf
    for _ in range(n_init):
        # k-means++ seeding: first centre uniform, others proportional to D^2.
        centres = np.empty((k, x.shape[1]))
        centres[0] = x[rng.integers(n)]
        closest = ((x - centres[0]) ** 2).sum(axis=1)
        for c in range(1, k):
            total = closest.sum()
            if total <= 0:
                centres[c] = x[rng.integers(n)]
            else:
                centres[c] = x[rng.choice(n, p=closest / total)]
            closest = np.minimum(closest, ((x - centres[c]) ** 2).sum(axis=1))
        labels = np.zeros(n, dtype=np.int64)
        for _it in range(max_iter):
            dist2 = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist2.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if members.shape[0]:
                    centres[c] = members.mean(axis=0)
        inertia = ((x - centres[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    assert best_labels is not None
    return best_labels


def onc(
    corr: pd.DataFrame | npt.ArrayLike,
    max_k: int | None = None,
    n_init: int = 10,
    seed: int | None = 0,
) -> ONCResult:
    """Cluster the variables of a correlation matrix, choosing ``k`` by silhouette.

    Parameters
    ----------
    corr : DataFrame or 2-D array
        Symmetric correlation matrix with unit diagonal, entries in ``[-1, 1]``.
    max_k : int, optional
        Largest ``k`` to try. Defaults to ``N - 1``. Must satisfy ``2 <= max_k <= N - 1``.
    n_init : int, default 10
        k-means restarts per candidate ``k``.
    seed : int, optional
        Seed for reproducible restarts.

    Returns
    -------
    ONCResult

    Raises
    ------
    ValueError
        If the matrix is not square, symmetric, unit-diagonal and within ``[-1, 1]``,
        or if there are fewer than three variables.
    """
    c = corr.to_numpy(dtype=np.float64) if isinstance(corr, pd.DataFrame) else np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be a square matrix.")
    n = c.shape[0]
    if n < 3:
        raise ValueError("need at least three variables to choose k in [2, N-1].")
    if not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("corr must be symmetric.")
    if not np.allclose(np.diag(c), 1.0, atol=1e-10):
        raise ValueError("corr must have a unit diagonal.")
    if np.any(c < -1.0 - 1e-10) or np.any(c > 1.0 + 1e-10):
        raise ValueError("corr entries must lie in [-1, 1].")
    top = n - 1 if max_k is None else int(max_k)
    if not 2 <= top <= n - 1:
        raise ValueError(f"max_k must be in [2, N-1] = [2, {n - 1}], got {max_k!r}.")

    dist = _distance(c)
    rng = np.random.default_rng(seed)
    scores: dict[int, float] = {}
    best: tuple[float, int, npt.NDArray[np.int64]] | None = None
    for k in range(2, top + 1):
        labels = _kmeans(c, k, n_init, rng)
        mean_s = float(silhouette_scores(dist, labels).mean())
        scores[k] = mean_s
        # Strict improvement only: ties keep the smaller k.
        if best is None or mean_s > best[0]:
            best = (mean_s, k, labels)
    assert best is not None
    mean_s, k, labels = best
    return ONCResult(labels=labels, n_clusters=k, silhouette=mean_s, scores=scores)
