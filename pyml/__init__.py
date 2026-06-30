r"""pyml: Pure Python Machine Learning.

A pure NumPy/SciPy implementation of classic machine learning
algorithms built from scratch for educational purposes. All
models follow a scikit-learn-compatible API with ``fit``,
``predict``, and ``transform`` methods where applicable.

Subpackages
-----------
linear_model
    Linear models for regression and classification with optional
    L1/L2 regularization (LinearRegression, LogisticRegression,
    Ridge, Lasso).
cluster
    Clustering algorithms including K-Means with K-Means++
    initialization and DBSCAN (Kmeans, DBSCAN).
neighbors
    K-Nearest Neighbors for classification and regression with
    multiple distance metrics and weighting strategies
    (KNNClassifier, KNNRegressor).
preprocess
    Feature scaling utilities including min-max, standardization,
    and robust scaling (MinMaxScaler, StandardScaler,
    RobustScaler).
tree
    Decision Tree models for classification and regression with
    configurable splitting criteria (DTClassifier, DTRegressor).

Available estimators
--------------------
**Regression:**
- :class:`LinearRegression` — Ordinary least squares with
  optional L1/L2 regularization.
- :class:`Ridge` — L2-regularized linear regression.
- :class:`Lasso` — L1-regularized linear regression with
  feature selection.
- :class:`KNNRegressor` — K-Nearest Neighbors regression.
- :class:`DTRegressor` — Decision Tree regression.

**Classification:**
- :class:`LogisticRegression` — Binary classifier with sigmoid
  activation and cross-entropy loss.
- :class:`KNNClassifier` — K-Nearest Neighbors classifier.
- :class:`DTClassifier` — Decision Tree classifier with Gini
  or entropy splitting.

**Clustering:**
- :class:`Kmeans` — K-Means clustering via Lloyd's algorithm.
- :class:`DBSCAN` — Density-based clustering with automatic
  outlier detection.

**Preprocessing:**
- :class:`MinMaxScaler` — Scale to [0, 1] range.
- :class:`StandardScaler` — Zero mean and unit variance.
- :class:`RobustScaler` — Median and IQR scaling, robust to
  outliers.

Notes
-----
All implementations prioritize educational clarity over
computational efficiency. They are designed to help understand
the inner workings of machine learning algorithms rather than
for production use. For production workloads, use scikit-learn
or similar optimized libraries.

Examples
--------
>>> from pyml import LinearRegression, Kmeans, StandardScaler
>>> import numpy as np
>>>
>>> # Regression
>>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
>>> y = np.array([3., 7., 11.])
>>> model = LinearRegression(learning_rate=0.01, max_iter=500)
>>> model.fit(X, y)
>>> model.predict(np.array([[7., 8.]]))
array([15.])
>>>
>>> # Clustering
>>> X_cluster = np.array([[1., 2.], [1., 4.], [10., 2.], [10., 4.]])
>>> kmeans = Kmeans(k=2, random_state=42)
>>> kmeans.fit(X_cluster)
>>> kmeans.predict(X_cluster)
array([0, 0, 1, 1])
>>>
>>> # Preprocessing
>>> scaler = StandardScaler()
>>> X_scaled = scaler.fit_transform(X)
"""

from .cluster import (
    DBSCAN,
    Kmeans,
)
from .linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from .neighbors import (
    KNNClassifier,
    KNNRegressor,
)
from .preprocess import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from .tree import (
    DTClassifier,
    DTRegressor,
)

__all__ = [
    "DBSCAN",
    "DTClassifier",
    "DTRegressor",
    "KNNClassifier",
    "KNNRegressor",
    "Kmeans",
    "Lasso",
    "LinearRegression",
    "LogisticRegression",
    "MinMaxScaler",
    "Ridge",
    "RobustScaler",
    "StandardScaler",
]
