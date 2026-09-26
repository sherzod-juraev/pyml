Decision Tree Regressor
=======================

Decision Tree Regressor predicts a continuous target by recursively
splitting the feature space on the feature and threshold that most
reduce target variance, then predicting the mean target value within
each resulting region.

Use it when the relationship between features and target has sharp
changes or plateaus that a smooth curve wouldn't capture well — unlike
the linear models, it makes no assumption about linearity, and needs
no feature standardization.

.. code-block:: python

    from pyml.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.tree import DecisionTreeRegressor

    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 10, size=(60, 1)), axis=0)
    y = np.sin(X).ravel() + rng.normal(0, 0.1, size=60)

    model = DecisionTreeRegressor(max_depth=3).fit(X, y)
    X_line = np.linspace(0, 10, 200).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.6, label="Training data")
    ax.plot(
        X_line,
        model.predict(X_line),
        color="red",
        label="DecisionTreeRegressor (max_depth=3)"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("DecisionTreeRegressor fit")
    ax.legend(loc="best")

Key parameters
--------------

- ``max_depth`` — how many levels of splits the tree is allowed to
  make. Too shallow and predictions look like a coarse staircase; too
  deep and it can fit noise in the training data.
- ``min_samples`` — smallest number of samples a node needs before a
  split is even considered. Raising it prevents splits based on a
  handful of points.
- ``min_impurity_decrease`` — smallest variance reduction a split must
  achieve to be kept. Raising it prunes splits that barely help.

.. note::
    Predictions are always a step function — flat within each leaf
    region — since every point in a region gets the same mean target
    value. This is different from the smooth curves the linear models
    produce.
