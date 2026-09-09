Ridge
=====

Ridge regression is ordinary linear regression with a penalty added
for large coefficients. This discourages the model from relying too
heavily on any single feature, which makes it more stable when your
features are correlated with each other or you have many features
relative to the amount of data.

Use it when you suspect :doc:`linear_regression` is overfitting, but
you want to keep every feature in the model (compare with
:doc:`lasso`, which can drop features entirely).

.. code-block:: python

    from pyml.linear_model import Ridge

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.linear_model import LinearRegression, Ridge

    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
    y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 3.0, size=30)

    ols = LinearRegression(max_iter=5000).fit(X, y)
    ridge_low = Ridge(alpha=0.1, max_iter=5000).fit(X, y)
    ridge_high = Ridge(alpha=10.0, max_iter=5000).fit(X, y)
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5, label="Training data")
    ax.plot(X_line, ols.predict(X_line), label="No regularization")
    ax.plot(X_line, ridge_low.predict(X_line), label="alpha=0.1")
    ax.plot(X_line, ridge_high.predict(X_line), label="alpha=10.0")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Ridge: effect of alpha")
    ax.legend()

Key parameters
--------------

- ``alpha`` — how strongly large coefficients are penalized. Higher
  values shrink coefficients closer to zero (but never exactly to
  zero); ``alpha=0`` is equivalent to plain
  :doc:`linear_regression`.
- ``learning_rate``, ``max_iter``, ``fit_intercept`` — same as in
  :doc:`linear_regression`.
