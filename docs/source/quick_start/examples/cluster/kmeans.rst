KMeans
======

KMeans partitions samples into ``n_clusters`` groups by repeatedly
assigning each point to its nearest centroid, then moving each
centroid to the mean of the points assigned to it.

Use it when you expect roughly spherical, evenly sized clusters and
you already know how many clusters you're looking for. It assumes
convex clusters and struggles with irregular shapes — see
:doc:`dbscan` for an alternative that doesn't make that assumption.

.. code-block:: python

    from pyml.cluster import KMeans

    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.cluster import KMeans

    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=(-3, -3), scale=1.0, size=(50, 2))
    X1 = rng.normal(loc=(3, 3), scale=1.0, size=(50, 2))
    X2 = rng.normal(loc=(0, 4), scale=1.0, size=(50, 2))
    X = np.vstack([X0, X1, X2])

    model = KMeans(n_clusters=3, random_state=42).fit(X)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", alpha=0.6)
    ax.scatter(
        model.centroids_[:, 0], model.centroids_[:, 1],
        marker="x", s=200, color="red", label="Centroids",
    )
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("KMeans on well-separated, spherical clusters")
    ax.legend()

Key parameters
--------------

- ``n_clusters`` — how many clusters to form. Must be chosen in
  advance; unlike :doc:`dbscan`, KMeans has no way to discover this
  on its own.
- ``max_iter``, ``tol`` — control how long the assignment/update loop
  runs before stopping. Same role as in the linear models.
- ``random_state`` — seeds the k-means++ initialization for
  reproducible results across runs.

.. note::
    KMeans assigns every point to a cluster, even outliers. If your
    data has noise points that don't belong to any real cluster,
    :doc:`dbscan` can label them separately instead of forcing them
    into the nearest group.
