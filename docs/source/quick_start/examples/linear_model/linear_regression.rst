Linear Regression
==================

Linear regression finds the straight line (or hyperplane, with more
than one feature) that best fits your data — the one that minimizes
the average squared distance between the line and your actual data
points.

Use it when you expect a roughly linear relationship between your
features and the target, and you don't have reason to suspect
overfitting (if you do, see :doc:`ridge` or :doc:`lasso` instead).

.. code-block:: python

    from pyml.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.linear_model import LinearRegression

    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
    y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 1.5, size=30)

    model = LinearRegression(max_iter=5000).fit(X, y)
    X_line = np.linspace(0, 10, 100).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.6, label="Training data")
    ax.plot(X_line, model.predict(X_line), color="red", label="Fitted line")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("LinearRegression")
    ax.legend()

Key parameters
--------------

- ``learning_rate`` — how big a step the model takes while learning.
  Too high and it can overshoot and fail to converge; too low and
  training takes longer than necessary.
- ``max_iter`` — the maximum number of training steps. If your model
  hasn't converged, try increasing this.
- ``fit_intercept`` — whether the line is allowed to shift up or down
  (an intercept), or must pass through the origin. Almost always
  left as ``True``.
