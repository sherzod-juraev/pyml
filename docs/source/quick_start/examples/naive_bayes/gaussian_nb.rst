GaussianNB
==========

GaussianNB is a probabilistic classifier based on Bayes' theorem. It
assumes that, within each class, every feature follows a normal
(Gaussian) distribution — and uses that assumption to estimate how
likely a new point is to belong to each class.

Use it when your features are continuous, roughly bell-shaped within
each class, and you want a fast, simple baseline before trying more
complex models.

.. code-block:: python

    from pyml.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.naive_bayes import GaussianNB

    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(50, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.0, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    model = GaussianNB().fit(X, y)

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
    ax.set_title("GaussianNB decision boundary")

.. note::
    GaussianNB has no hyperparameters to tune — fitting is a single
    closed-form pass over the data (per-class mean and variance),
    not an iterative optimization. If a feature is severely
    non-Gaussian within a class (e.g. multimodal or heavily skewed),
    predictions may be unreliable — consider transforming that
    feature or using a different model.
