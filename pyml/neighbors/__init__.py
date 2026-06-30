r"""K-Nearest Neighbors for classification and regression.

This subpackage provides pure NumPy/SciPy implementations of the
K-Nearest Neighbors algorithm for both classification and regression
tasks, supporting multiple distance metrics and weighting strategies.

Available models
----------------
**Classification:**

- :class:`KNNClassifier` — Predicts class labels by majority vote
  among the :math:`k` nearest neighbors. Supports uniform and
  distance-weighted voting.

**Regression:**

- :class:`KNNRegressor` — Predicts continuous target values by
  (weighted) averaging the target values of the :math:`k` nearest
  neighbors. Supports uniform and distance-weighted aggregation.

Common interface
----------------
All models implement:

- ``fit(X, y)`` — Store training data (lazy learning, no
  computation performed).
- ``predict(X)`` — Compute distances to training data, find
  :math:`k` nearest neighbors, and aggregate their values.

Notes
-----
KNN is a non-parametric, instance-based (lazy) learning algorithm.
All computation is deferred to predict time, making it memory-intensive
but flexible. Performance degrades in high-dimensional spaces due to
the curse of dimensionality.

Distance computations are delegated to
:func:`scipy.spatial.distance.cdist`, supporting Euclidean,
Manhattan (cityblock), Chebyshev, and Cosine metrics.

Examples
--------
>>> from pyml import KNNClassifier, KNNRegressor
>>> import numpy as np
>>>
>>> X = np.array([[1., 2.], [2., 3.], [3., 4.], [6., 7.]])
>>> y_cls = np.array([0, 0, 1, 1])
>>> y_reg = np.array([1.5, 2.0, 3.5, 6.0])
>>>
>>> clf = KNNClassifier(k=3)
>>> clf.fit(X, y_cls)
>>> clf.predict(np.array([[2., 2.]]))
array([0])
>>>
>>> reg = KNNRegressor(k=3)
>>> reg.fit(X, y_reg)
>>> reg.predict(np.array([[2., 2.]]))
array([2.333...])
"""

from .knn_classifier import KNNClassifier
from .knn_regressor import KNNRegressor

__all__ = [
    "KNNClassifier",
    "KNNRegressor",
]
