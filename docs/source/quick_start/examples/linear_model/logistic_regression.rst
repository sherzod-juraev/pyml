Logistic Regression
====================

Logistic regression predicts the probability that a sample belongs
to one of two classes, based on a linear combination of its
features passed through a sigmoid function. Despite the name, it's
a classification model, not a regression one.

Use it as your starting point for binary classification: it's
simple, fast, and gives you a probability, not just a label.

.. code-block:: python

    from pyml.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    n = 100
    X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    model = LogisticRegression(max_iter=5000).fit(X, y)

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
    ax.set_title("LogisticRegression decision boundary")
    ax.legend()

Key parameters
--------------

- ``learning_rate``, ``max_iter``, ``fit_intercept`` — same role as
  in :doc:`linear_regression`.

.. note::
    If you suspect overfitting, see :doc:`ridge_classifier` or
    :doc:`lasso_classifier`.
