Lasso
=====

Lasso regression, like :doc:`ridge`, penalizes large coefficients —
but its penalty can push coefficients to exactly zero, effectively
removing those features from the model entirely.

Use it when you suspect only some of your features are actually
useful, and you want the model to identify which ones automatically.

.. code-block:: python

    from pyml.linear_model import Lasso

    model = Lasso(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.linear_model import LinearRegression, Lasso

    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
    y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 3.0, size=30)

    ols = LinearRegression(max_iter=5000).fit(X, y)
    lasso_low = Lasso(alpha=0.1, max_iter=5000).fit(X, y)
    lasso_high = Lasso(alpha=2.0, max_iter=5000).fit(X, y)
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5, label="Training data")
    ax.plot(X_line, ols.predict(X_line), label="No regularization")
    ax.plot(X_line, lasso_low.predict(X_line), label="alpha=0.1")
    ax.plot(X_line, lasso_high.predict(X_line), label="alpha=2.0")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Lasso: effect of alpha")
    ax.legend()

Key parameters
--------------

- ``alpha`` — how strongly coefficients are pushed toward zero.
  Higher values zero out more coefficients; ``alpha=0`` is
  equivalent to plain :doc:`linear_regression`.
- ``learning_rate``, ``max_iter``, ``fit_intercept`` — same as in
  :doc:`linear_regression`.

.. note::
    If every coefficient ends up at zero, ``alpha`` is likely too
    high — try lowering it.
