Radius Regressor
================

Radius Regressor predicts a continuous target by finding every
training point within a fixed ``radius`` of a query point, and
averaging their target values — unlike :doc:`knn_regressor`, the
number of neighbors used varies from point to point.

Use it when your data isn't evenly spread out, and a fixed number of
neighbors (as in KNN) would pull in points that are actually far
away in sparse regions.

.. code-block:: python

    from pyml.neighbors import RadiusNeighborsRegressor

    model = RadiusNeighborsRegressor(radius=1.5, outlier_label=0.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.neighbors import RadiusNeighborsRegressor

    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
    y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 1.5, size=30)

    # Introduce a gap in the training data to show outlier_label in action
    mask = (X.ravel() < 4.0) | (X.ravel() > 5.5)
    X_gap, y_gap = X[mask], y[mask]

    model = RadiusNeighborsRegressor(radius=1.0, outlier_label=0.0).fit(X_gap, y_gap)
    X_line = np.linspace(0, 10, 200).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(X_gap, y_gap, alpha=0.5, label="Training data")
    ax.plot(X_line, model.predict(X_line), color="red", label="radius=1.0")
    ax.axvspan(4.0, 5.5, color="gray", alpha=0.15, label="No training data")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("RadiusNeighborsRegressor: outlier_label fills a data gap")
    ax.legend()

Key parameters
--------------

- ``radius`` — how far to search around each query point. Too small
  and some points may have no neighbors at all; too large and it
  starts to behave like a global average.
- ``metric`` — how distance between points is measured: ``euclidean``,
  ``chebyshev``, or ``cityblock``.
- ``weights`` — ``"uniform"`` gives every neighbor equal weight in the
  average; ``"distance"`` lets closer neighbors count for more.
- ``outlier_label`` — what to predict when no training point falls
  within the radius, for that specific query. Leave as ``None`` to
  raise an error instead — useful when a silent fallback prediction
  would be worse than failing loudly. Set it to a specific number to
  have those points fall back to that value instead of crashing the
  whole batch.
