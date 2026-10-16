"""Public GCRFS selector and training-only preprocessing."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA

from .graph import multiscale_laplacian_scores, row_l2_normalize
from .ranking import (
    absolute_correlation,
    aggregate_quality,
    conditional_order,
    percentile_scores,
)


DEFAULT_K_VALUES = (5, 8, 12, 16, 20)
EPS = 1e-12


def _validate_training_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must be a 2-D matrix with at least 2 rows and 1 column")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


def _validate_feature_matrix(X: np.ndarray, n_features: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError("X must have the same number of features as the training matrix")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


class GCRFS:
    """Graph-Conditioned Redundancy Feature Selection.

    The selector is label-free.  ``fit`` must receive a training matrix only.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        tau: float = 0.5,
        lambda_div: float = 1.2,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        conditional_depth: int = 50,
        candidate_pool: int = 500,
        pca_components: int = 64,
        standardize: bool = True,
        random_state: int = 0,
    ) -> None:
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.lambda_div = float(lambda_div)
        self.k_values = tuple(int(k) for k in k_values)
        self.conditional_depth = int(conditional_depth)
        self.candidate_pool = int(candidate_pool)
        self.pca_components = int(pca_components)
        self.standardize = bool(standardize)
        self.random_state = int(random_state)

        if self.k_values != DEFAULT_K_VALUES:
            raise ValueError("k_values must be (5, 8, 12, 16, 20)")
        if self.tau <= 0 or self.candidate_pool < 1 or self.conditional_depth < 1:
            raise ValueError("tau, candidate_pool and conditional_depth must be positive")
        if not 0 <= self.alpha <= 1 or self.lambda_div < 0:
            raise ValueError("alpha must be in [0, 1] and lambda_div must be non-negative")

        self.ranking_: Optional[np.ndarray] = None
        self.structural_scores_: Optional[np.ndarray] = None
        self.normalized_scores_: Optional[np.ndarray] = None
        self.quality_scores_: Optional[np.ndarray] = None
        self.redundancy_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.pca_rank_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X_train: np.ndarray) -> "GCRFS":
        """Fit on training features and compute the complete feature ranking."""
        X = _validate_training_matrix(X_train)
        self.n_features_in_ = X.shape[1]

        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ <= EPS] = 1.0
            X_work = (X - self.mean_) / self.scale_
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.scale_ = np.ones(X.shape[1], dtype=float)
            X_work = X.copy()

        X_l2 = row_l2_normalize(X_work)
        self.pca_rank_ = min(self.pca_components, X_l2.shape[1], X_l2.shape[0] - 1)
        graph_features = row_l2_normalize(
            PCA(
                n_components=self.pca_rank_,
                svd_solver="randomized",
                random_state=self.random_state,
            ).fit_transform(X_l2)
        )

        structural = multiscale_laplacian_scores(
            X_l2, graph_features, self.k_values
        )
        normalized = percentile_scores(structural)
        quality = aggregate_quality(normalized, self.alpha, self.tau)
        correlation = absolute_correlation(X_l2)

        self.structural_scores_ = structural
        self.normalized_scores_ = normalized
        self.quality_scores_ = quality
        self.redundancy_ = correlation
        self.ranking_ = conditional_order(
            quality,
            correlation,
            self.lambda_div,
            self.conditional_depth,
            self.candidate_pool,
        )
        return self

    def rank(self) -> np.ndarray:
        """Return original feature indices, best first."""
        if self.ranking_ is None:
            raise RuntimeError("call fit(X_train) before rank()")
        return self.ranking_.copy()

    def select(self, X: np.ndarray, n_features: int) -> np.ndarray:
        """Return the first ``n_features`` original columns in ranking order."""
        if self.ranking_ is None or self.n_features_in_ is None:
            raise RuntimeError("call fit(X_train) before select()")
        X = _validate_feature_matrix(X, self.n_features_in_)
        n_features = int(n_features)
        if not 1 <= n_features <= self.n_features_in_:
            raise ValueError("n_features must be between 1 and the input dimension")
        return X[:, self.ranking_[:n_features]]

    def get_support(self, n_features: int) -> np.ndarray:
        """Return a boolean mask for the requested ordered prefix."""
        if self.ranking_ is None or self.n_features_in_ is None:
            raise RuntimeError("call fit(X_train) before get_support()")
        n_features = int(n_features)
        if not 1 <= n_features <= self.n_features_in_:
            raise ValueError("n_features must be between 1 and the input dimension")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.ranking_[:n_features]] = True
        return mask


def rank_features(X_train: np.ndarray, **kwargs: object) -> np.ndarray:
    """One-call convenience wrapper returning a GCRFS ranking."""
    return GCRFS(**kwargs).fit(X_train).rank()


__all__ = ["GCRFS", "rank_features"]
