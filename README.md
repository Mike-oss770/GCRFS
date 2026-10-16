# GCRFS

GCRFS is an ordered-prefix unsupervised feature selector with graph-conditioned redundancy across feature budgets.

## Repository layout

```text
GCRFS/
├── README.md
└── gcrfs/
    ├── __init__.py    # public API
    ├── selector.py    # GCRFS estimator and train-only preprocessing
    ├── graph.py       # kNN graphs and graph Laplacian scores
    └── ranking.py     # structural aggregation and conditional ordering
```

Only algorithm source files and this README are included. Datasets, results, paper files, and baseline implementations are intentionally omitted.

## Installation

```bash
pip install numpy scipy scikit-learn
```

## Quick start

```python
import numpy as np
from gcrfs import GCRFS

# Training samples only; labels are not used by GCRFS.
X_train = np.asarray(..., dtype=float)

selector = GCRFS()                  # formal conditional depth: 50
selector.fit(X_train)
ranking = selector.rank()           # original feature indices, best first
X_20 = selector.select(X_train, 20) # ordered prefix of 20 features
mask = selector.get_support(20)
```

For a preprocessed training fold, use `GCRFS(standardize=False)`. With the default `standardize=True`, z-score parameters are fitted only on the matrix passed to `fit`; never fit the selector on test data or pass labels to it.

## Formal settings

The default implementation uses the formal GCRFS settings:

- graph scales `k = (5, 8, 12, 16, 20)`;
- structural aggregation `alpha = 0.2`, `tau = 0.5`;
- conditional redundancy weight `lambda_div = 1.2`;
- candidate pool `min(500, d)`;
- PCA dimension capped at `min(64, n - 1, d)` for graph construction;
- PCA random state `0` and stable original-index tie breaking.

The structural score is computed from multi-scale graph Laplacian scores. The ordered prefix minimizes percentile structural risk plus average absolute correlation with already selected features. After `conditional_depth`, remaining features follow structural-risk order.

For the extended-depth variant, set:

```python
selector = GCRFS(conditional_depth=100)
```

The actual conditional depth is capped by the available candidate-pool size.

## Reproducible fold usage

For each fold, fit a new selector using only that fold's training features and apply `ranking[:m]` for each budget. Test labels may be used later by the evaluator, but are never passed to GCRFS.

## API

- `GCRFS(...).fit(X_train)` fits the selector and returns the selector.
- `.rank()` returns a complete permutation of feature indices.
- `.select(X, n_features)` returns the requested ordered prefix.
- `.get_support(n_features)` returns a boolean prefix mask.
- `rank_features(X_train, **kwargs)` is a one-call convenience wrapper.
