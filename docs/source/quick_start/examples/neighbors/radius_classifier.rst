Radius Classifier
====================

Radius Classifier predicts a class label by finding every training
point within a fixed ``radius`` of a query point, and taking a
majority vote among their labels — unlike :doc:`knn_classifier`,
the number of neighbors used varies from point to point.

Use it when your data isn't evenly spread out, and a fixed number of
neighbors (as in KNN) would pull in points that are actually far
away in sparse regions.

.. code-block:: python

    from pyml.neighbors import RadiusNeighborsClassifier

    model = RadiusNeighborsClassifier(radius=1.5, outlier_label=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.neighbors import RadiusNeighborsClassifier

    rng = np.random.default_rng(42)
    n = 100
    X0 = rng.normal(loc=(-2, -2), scale=1.5, size=(n // 2, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.5, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    # Carve out a gap clearly larger than the model's radius, so grid
    # points near the center genuinely have no neighbors within reach.
    gap_radius = 3.0
    model_radius = 1.0
    mask = np.linalg.norm(X, axis=1) > gap_radius
    X_gap, y_gap = X[mask], y[mask]

    model = RadiusNeighborsClassifier(radius=model_radius, outlier_label=0).fit(X_gap, y_gap)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    preds = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, preds, levels=1, cmap="RdBu", alpha=0.5)
    ax.scatter(X_gap[y_gap == 0, 0], X_gap[y_gap == 0, 1], label="Class 0", edgecolor="k")
    ax.scatter(X_gap[y_gap == 1, 0], X_gap[y_gap == 1, 1], label="Class 1", edgecolor="k")
    circle = plt.Circle((0, 0), gap_radius, color="black", fill=False, linestyle="--", linewidth=1.5, label="No training data")
    ax.add_patch(circle)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("RadiusNeighborsClassifier: outlier_label fills a data gap")
    ax.legend()

Key parameters
--------------

- ``radius`` — how far to search around each query point. Too small
  and some points may have no neighbors at all; too large and it
  starts to behave like a global majority vote.
- ``metric`` — how distance between points is measured: ``euclidean``,
  ``chebyshev``, or ``cityblock``.
- ``weights`` — ``"uniform"`` gives every neighbor an equal vote;
  ``"distance"`` lets closer neighbors count for more.
- ``outlier_label`` — the label to predict when no training point
  falls within the radius, for that specific query. Leave as ``None``
  to raise an error instead — useful when a silent fallback
  prediction would be worse than failing loudly. Set it to a specific
  class to have those points fall back to that label instead of
  crashing the whole batch. Note that since the fallback value is
  itself a valid class label, the affected region won't stand out
  visually — only the training data (or lack of it) tells you it
  happened.
