Elastic Net Classifier
======================

Elastic Net Classifier combines the penalties of
:doc:`ridge_classifier` and :doc:`lasso_classifier`: it shrinks
coefficients toward zero like Ridge, while still being able to drop
some features entirely like Lasso.

Use it when you suspect some features are irrelevant (as with Lasso)
but your remaining features are also correlated with each other (as
with Ridge) — a case where Lasso alone tends to pick one feature
from a correlated group somewhat arbitrarily.

.. code-block:: python

    from pyml.linear_model import ElasticNetClassifier

    model = ElasticNetClassifier(alpha=1.0, l1_ratio=0.5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.linear_model import ElasticNetClassifier

    rng = np.random.default_rng(42)
    n = 100
    X0 = rng.normal(loc=(-2, -2), scale=1.5, size=(n // 2, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.5, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    model = ElasticNetClassifier(alpha=1.0, l1_ratio=0.5, max_iter=5000).fit(X, y)

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
    ax.set_title("ElasticNetClassifier decision boundary")
    ax.legend()

Key parameters
--------------

- ``alpha`` — overall penalty strength, same role as in
  :doc:`ridge_classifier`/:doc:`lasso_classifier`.
- ``l1_ratio`` — the mix between the two penalties. ``l1_ratio=1``
  behaves like pure :doc:`lasso_classifier`; ``l1_ratio=0`` behaves
  like pure :doc:`ridge_classifier`.
- ``learning_rate``, ``max_iter``, ``fit_intercept`` — same as in
  :doc:`linear_regression`.
