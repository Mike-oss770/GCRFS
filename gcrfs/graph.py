"""Sample-graph construction and multi-scale graph feature scores."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12


def row_l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize each sample row to unit L2 norm."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, EPS)


def build_knn_graph(Z: np.ndarray, k: int) -> sparse.csr_matrix:
    """Build the symmetrized adaptive-kernel kNN graph used by GCRFS."""
    k = min(max(1, int(k)), Z.shape[0] - 1)
    distances, neighbors = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    distances, neighbors = distances[:, 1:], neighbors[:, 1:]

    sigma = np.maximum(distances[:, -1], EPS)
    rows = np.repeat(np.arange(Z.shape[0]), k)
    cols = neighbors.ravel()
    values = np.exp(
        -(distances.ravel() ** 2)
        / (sigma[rows] * sigma[cols] + EPS)
    )
    directed = sparse.csr_matrix(
        (values, (rows, cols)), shape=(Z.shape[0], Z.shape[0])
    )
    graph = 0.5 * (directed + directed.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph.tocsr()


def laplacian_score(X: np.ndarray, graph: sparse.csr_matrix) -> np.ndarray:
    """Return one graph Laplacian score per feature; lower is better."""
    degree = np.asarray(graph.sum(axis=1)).ravel()
    total_degree = max(float(degree.sum()), EPS)
    centered = X - (degree[:, None] * X).sum(axis=0) / total_degree
    denominator = (degree[:, None] * centered * centered).sum(axis=0)
    numerator = denominator - (centered * (graph @ centered)).sum(axis=0)

    scores = np.full(X.shape[1], np.inf, dtype=float)
    valid = denominator > EPS
    scores[valid] = numerator[valid] / denominator[valid]
    return scores


def multiscale_laplacian_scores(
    X: np.ndarray, graph_features: np.ndarray, k_values: tuple[int, ...]
) -> np.ndarray:
    """Stack graph Laplacian scores for all requested neighborhood scales."""
    return np.column_stack([
        laplacian_score(X, build_knn_graph(graph_features, k))
        for k in k_values
    ])
