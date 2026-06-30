r"""Clustering algorithms for unsupervised learning.

This subpackage provides pure NumPy/SciPy implementations of
fundamental clustering algorithms, supporting multiple distance
metrics and initialization strategies.

Available models
----------------
**Centroid-based clustering:**

- :class:`Kmeans` — K-Means clustering via Lloyd's algorithm with
  optional K-Means++ initialization. Partitions data into :math:`k`
  clusters by minimizing Within-Cluster Sum of Squares (WCSS).

**Density-based clustering:**

- :class:`DBSCAN` — Density-Based Spatial Clustering of Applications
  with Noise. Discovers clusters of arbitrary shape based on local
  density, automatically identifying outliers as noise.

**Initialization utilities:**

- :class:`KmeansPP` — K-Means++ centroid initialization strategy.
  Selects well-spread initial centroids via distance-weighted
  sampling for improved convergence.

Common interface
----------------
- ``fit(X)`` / ``fit_predict(X)`` — Train the model and obtain
  cluster assignments.
- ``predict(X)`` — Assign new samples to existing clusters
  (KMeans only; DBSCAN requires re-fitting).

Notes
-----
All distance computations are delegated to
:func:`scipy.spatial.distance.cdist`, supporting Euclidean,
Manhattan (cityblock), and Chebyshev metrics.

Examples
--------
>>> from pyml import Kmeans, DBSCAN
>>> import numpy as np
>>>
>>> X = np.array([[1., 2.], [1., 4.], [10., 2.], [10., 4.]])
>>>
>>> kmeans = Kmeans(k=2, random_state=42)
>>> kmeans.fit(X)
>>> kmeans.predict(X)
array([0, 0, 1, 1])
>>>
>>> dbscan = DBSCAN(eps=3.0, MinPts=2)
>>> dbscan.fit_predict(X)
array([0, 0, 1, 1])
"""

from ._kmeans_pp import KmeansPP
from .dbscan import DBSCAN
from .kmeans import Kmeans

__all__ = [
    "DBSCAN",
    "Kmeans",
    "KmeansPP",
]
