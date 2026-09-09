DBSCAN
======

DBSCAN groups together points that are densely packed, and marks
points in low-density regions as noise. Unlike :doc:`kmeans`, it does
not assume clusters are spherical, and it does not require you to
specify the number of clusters in advance.

Use it when your clusters have irregular shapes, or when your data
contains noise points that shouldn't be forced into any cluster. It
struggles when clusters have very different densities, since ``eps``
and ``min_samples`` apply uniformly across the whole dataset.

.. code-block:: python

    from pyml.cluster import DBSCAN

    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.cluster import DBSCAN, KMeans

    rng = np.random.default_rng(42)
    n = 100
    theta = np.linspace(0, np.pi, n)

    moon1 = np.column_stack([np.cos(theta), np.sin(theta)])
    moon2 = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)])
    X = np.vstack([moon1, moon2]) + rng.normal(0, 0.08, size=(2 * n, 2))

    kmeans_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X)
    dbscan_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.6)
    axes[0].set_title("KMeans (n_clusters=2)")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")

    axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap="viridis", alpha=0.6)
    axes[1].set_title("DBSCAN (eps=0.2)")
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X2")

    fig.suptitle("KMeans vs. DBSCAN on non-spherical clusters")
    fig.tight_layout()

Key parameters
--------------

- ``eps`` — the radius used to find neighboring points. Too small and
  most points end up labeled as noise; too large and separate
  clusters start merging into one.
- ``min_samples`` — how many neighbors (including the point itself) a
  point needs within ``eps`` to count as a core point. Higher values
  demand denser regions before forming a cluster, which makes the
  result more resistant to noise.
- ``metric`` — how distance between points is measured: ``euclidean``,
  ``chebyshev``, or ``cityblock``.

.. note::
    Points that don't belong to any cluster are labeled ``-1``. This
    is a real, meaningful label — unlike KMeans, DBSCAN doesn't force
    every point into a group.
