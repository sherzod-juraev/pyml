KNN Regressor
=============

KNN Regressor predicts a continuous target by finding the
``n_neighbors`` closest training points to a query point, and
averaging their target values.

Use it as a simple, non-parametric baseline for regression — it makes
no assumptions about the shape of the underlying function, at the
cost of having to search the training data at every prediction.

.. code-block:: python

    from pyml.neighbors import KNNRegressor

    model = KNNRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.neighbors import KNNRegressor

    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
    y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 1.5, size=30)

    model_k3 = KNNRegressor(n_neighbors=3).fit(X, y)
    model_k15 = KNNRegressor(n_neighbors=15).fit(X, y)
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5, label="Training data")
    ax.plot(X_line, model_k3.predict(X_line), label="n_neighbors=3")
    ax.plot(X_line, model_k15.predict(X_line), label="n_neighbors=15")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("KNNRegressor: effect of n_neighbors")
    ax.legend()

Key parameters
--------------

- ``n_neighbors`` — how many neighbors are averaged for each
  prediction. Fewer neighbors follow the data more closely but can
  overfit to noise; more neighbors smooth the prediction but can miss
  real structure.
- ``metric`` — how distance between points is measured: ``euclidean``,
  ``chebyshev``, or ``cityblock``.
- ``weights`` — ``"uniform"`` gives every neighbor equal weight in the
  average; ``"distance"`` lets closer neighbors count for more.
