r"""Decision Tree models for classification and regression.

This subpackage provides pure NumPy implementations of Decision
Tree algorithms that recursively partition the feature space into
axis-aligned regions for prediction.

Available models
----------------
**Classification:**

- :class:`DTClassifier` — Predicts class labels by majority vote
  in each leaf node. Supports Gini impurity and Shannon entropy
  splitting criteria.

**Regression:**

- :class:`DTRegressor` — Predicts continuous target values as the
  mean of training samples in each leaf node. Uses variance
  reduction (MSE minimization) as the splitting criterion with
  cumulative sum optimization.

Common interface
----------------
All models implement:

- ``fit(X, y)`` — Build the decision tree from training data.
- ``predict(X)`` — Traverse the tree to make predictions for
  new samples.

Notes
-----
Decision Trees are non-parametric supervised learning algorithms
that partition the feature space by recursively selecting the
feature and threshold that best separates the data. They are
interpretable but prone to overfitting without proper constraints
(max depth, minimum samples per leaf, impurity decrease threshold).

Candidate thresholds are midpoints between consecutive sorted
feature values. The splitting process uses cumulative sum
precomputation for :math:`O(n \log n)` per-feature efficiency
in the regressor.

Examples
--------
>>> from pyml import DTClassifier, DTRegressor
>>> import numpy as np
>>>
>>> X = np.array([[1., 2.], [2., 3.], [3., 4.], [6., 7.]])
>>> y_cls = np.array([0, 0, 1, 1])
>>> y_reg = np.array([1.5, 2.0, 3.5, 6.0])
>>>
>>> clf = DTClassifier(depth=3, min_sample=2, algorithm='gini')
>>> clf.fit(X, y_cls)
>>> clf.predict(np.array([[2., 2.]]))
array([0])
>>>
>>> reg = DTRegressor(max_depth=3, min_split=2, min_leaf=1)
>>> reg.fit(X, y_reg)
>>> reg.predict(np.array([[2., 2.]]))
array([1.75])
"""

from .dt_classifier import DTClassifier
from .dt_regressor import DTRegressor

__all__ = [
    "DTClassifier",
    "DTRegressor",
]
