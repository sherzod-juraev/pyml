Elastic Net
===========

Elastic Net combines the penalties of :doc:`ridge` and :doc:`lasso`:
it shrinks coefficients toward zero like Ridge, while still being
able to drop some features entirely like Lasso.

Use it when you suspect some features are irrelevant (as with Lasso)
but your remaining features are also correlated with each other (as
with Ridge) — a case where Lasso alone tends to pick one feature
from a correlated group somewhat arbitrarily.

.. code-block:: python

    from pyml.linear_model import ElasticNet

    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.linear_model import ElasticNet, Lasso, Ridge

    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
    y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 3.0, size=30)

    ridge = Ridge(alpha=1.0, max_iter=5000).fit(X, y)
    lasso = Lasso(alpha=1.0, max_iter=5000).fit(X, y)
    elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000).fit(X, y)
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5, label="Training data")
    ax.plot(X_line, ridge.predict(X_line), label="Ridge")
    ax.plot(X_line, lasso.predict(X_line), label="Lasso")
    ax.plot(X_line, elastic.predict(X_line), label="Elastic Net")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Elastic Net vs. Ridge vs. Lasso")
    ax.legend()

Key parameters
--------------

- ``alpha`` — overall penalty strength, same role as in
  :doc:`ridge`/:doc:`lasso`.
- ``l1_ratio`` — the mix between the two penalties. ``l1_ratio=1``
  behaves like pure :doc:`lasso`; ``l1_ratio=0`` behaves like pure
  :doc:`ridge`.
- ``learning_rate``, ``max_iter``, ``fit_intercept`` — same as in
  :doc:`linear_regression`.
