Decision Tree Classifier
========================

Decision Tree Classifier predicts a class label by recursively
splitting the feature space on the feature and threshold that most
reduce Gini impurity, until each region is (nearly) pure or a
stopping condition is reached.

Use it when you want a model whose decisions are easy to trace back
to specific feature thresholds, or when the relationship between
features and target isn't linear — unlike the linear models, it makes
no assumption about how features and target relate, and needs no
feature standardization.

.. code-block:: python

    from pyml.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.tree import DecisionTreeClassifier

    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(50, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.0, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    model = DecisionTreeClassifier(max_depth=3).fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
    )
    grid_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, grid_pred.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("DecisionTreeClassifier decision boundary (max_depth=3)")

Key parameters
--------------

- ``max_depth`` — how many levels of splits the tree is allowed to
  make. Too shallow and it underfits; too deep and it can fit noise
  in the training data.
- ``min_samples`` — smallest number of samples a node needs before a
  split is even considered. Raising it prevents splits based on a
  handful of points.
- ``min_impurity_decrease`` — smallest impurity reduction a split
  must achieve to be kept. Raising it prunes splits that barely help.

.. note::
    The tree only splits when doing so actually reduces impurity — it
    won't use up its full ``max_depth`` if fewer splits already
    separate the classes well.
