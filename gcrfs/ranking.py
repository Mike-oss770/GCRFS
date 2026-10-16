"""Structural-risk aggregation and graph-conditioned redundancy ranking."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata


EPS = 1e-12


def percentile_scores(scores: np.ndarray) -> np.ndarray:
    """Convert each score column to a percentile-like rank in [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    output = np.empty_like(scores)
    for column in range(scores.shape[1]):
        output[:, column] = (
            rankdata(scores[:, column], method="average") - 1
        ) / max(scores.shape[0] - 1, 1)
    return output


def absolute_correlation(X: np.ndarray) -> np.ndarray:
    """Return pairwise absolute feature correlation with a zero diagonal."""
    centered = X - X.mean(axis=0)
    normalized = centered / (np.linalg.norm(centered, axis=0, keepdims=True) + EPS)
    correlation = np.abs(normalized.T @ normalized)
    np.fill_diagonal(correlation, 0.0)
    return np.clip(correlation, 0.0, 1.0).astype(np.float32)


def aggregate_quality(A: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    """Compute q_j from normalized multi-scale structural risks."""
    reference_k12 = A[:, 2]
    entropic_risk = tau * (logsumexp(A / tau, axis=1) - np.log(A.shape[1]))
    return (1.0 - alpha) * reference_k12 + alpha * entropic_risk


def conditional_order(
    quality: np.ndarray,
    correlation: np.ndarray,
    lambda_div: float,
    conditional_depth: int,
    candidate_pool: int,
) -> np.ndarray:
    """Return a complete ordered-prefix ranking.

    The first ``conditional_depth`` positions are selected greedily from the
    lowest-risk candidate pool. Remaining positions follow structural risk.
    Ties are resolved by the original feature index.
    """
    n_features = len(quality)
    relative_quality = (rankdata(quality, method="average") - 1) / max(
        n_features - 1, 1
    )

    base_order = np.lexsort((np.arange(n_features), relative_quality))
    pool = base_order[: min(int(candidate_pool), n_features)]
    depth = min(max(1, int(conditional_depth)), len(pool))

    selected: list[int] = []
    selected_mask = np.zeros(n_features, dtype=bool)
    remaining = np.zeros(n_features, dtype=bool)
    remaining[pool] = True

    for _ in range(depth):
        candidates = np.flatnonzero(remaining)
        if selected:
            diversity = correlation[candidates][:, selected].mean(axis=1)
            objective = relative_quality[candidates] + lambda_div * diversity
        else:
            objective = relative_quality[candidates]
        pick = int(candidates[np.lexsort((candidates, objective))[0]])
        selected.append(pick)
        selected_mask[pick] = True
        remaining[pick] = False

    suffix = base_order[~selected_mask[base_order]]
    return np.asarray(selected + suffix.tolist(), dtype=np.int64)
