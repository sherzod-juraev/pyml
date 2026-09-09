Lasso Classifier
================

Lasso Classifier is :doc:`logistic_regression` with an L1 penalty
added to the coefficients — the same relationship :doc:`lasso` has
to :doc:`linear_regression`. Its penalty can push coefficients to
exactly zero, effectively removing those features from the model
entirely.

Use it when you suspect only some of your features are actually
useful for separating the classes, and you want the model to
identify which ones automatically.

.. code-block:: python

    from pyml.linear_model import LassoClassifier

    model = LassoClassifier(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.linear_model import LassoClassifier

    rng = np.random.default_rng(42)
    n = 100
    X0 = rng.normal(loc=(-2, -2), scale=1.5, size=(n // 2, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.5, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    model = LassoClassifier(alpha=1.0, max_iter=5000).fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs = model.predict_proba(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.6)
    ax.scatter(X0[:, 0], X0[:, 1], label="Class 0", edgecolor="k")
    ax.scatter(X1[:, 0], X1[:, 1], label="Class 1", edgecolor="k")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("LassoClassifier decision boundary")
    ax.legend()

Key parameters
--------------

- ``alpha`` — how strongly coefficients are pushed toward zero.
  Higher values zero out more coefficients; ``alpha=0`` is
  equivalent to plain :doc:`logistic_regression`.
- ``learning_rate``, ``max_iter``, ``fit_intercept`` — same as in
  :doc:`linear_regression`.

.. note::
    If every coefficient ends up at zero, ``alpha`` is likely too
    high — try lowering it.
