KNN Classifier
==============

KNN Classifier predicts a class label by finding the ``n_neighbors``
closest training points to a query point, and taking a majority vote
among their labels.

Use it as a simple, non-parametric baseline for classification — it
makes no assumptions about the shape of the decision boundary, at the
cost of having to search the training data at every prediction.

.. code-block:: python

    from pyml.neighbors import KNNClassifier

    model = KNNClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.neighbors import KNNClassifier

    rng = np.random.default_rng(42)
    n = 100
    X0 = rng.normal(loc=(-2, -2), scale=1.5, size=(n // 2, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.5, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    model = KNNClassifier(n_neighbors=5).fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    preds = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, preds, levels=1, cmap="RdBu", alpha=0.5)
    ax.scatter(X0[:, 0], X0[:, 1], label="Class 0", edgecolor="k")
    ax.scatter(X1[:, 0], X1[:, 1], label="Class 1", edgecolor="k")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("KNNClassifier decision boundary (k=5)")
    ax.legend()

Key parameters
--------------

- ``n_neighbors`` — how many neighbors vote on each prediction. Too
  few and the boundary gets noisy; too many and it oversmooths,
  blurring the distinction between classes.
- ``metric`` — how distance between points is measured: ``euclidean``,
  ``chebyshev``, or ``cityblock``.
- ``weights`` — ``"uniform"`` gives every neighbor an equal vote;
  ``"distance"`` lets closer neighbors count for more.
