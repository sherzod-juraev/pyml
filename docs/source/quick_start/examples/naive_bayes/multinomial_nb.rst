MultinomialNB
=============

MultinomialNB is suited to count data — features that represent how
many times something occurred, like word frequencies in a
bag-of-words representation. Within each class, it estimates each
feature's probability as its relative frequency, with Laplace
smoothing to avoid zero probabilities for features never seen in a
class.

Use it when your features are counts (not continuous measurements)
and you want a fast, simple baseline — the classic example is text
classification with a bag-of-words or TF-IDF representation.

.. code-block:: python

    from pyml.naive_bayes import MultinomialNB

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.naive_bayes import MultinomialNB

    rng = np.random.default_rng(42)
    X0 = rng.poisson(lam=[8, 1], size=(50, 2))
    X1 = rng.poisson(lam=[1, 8], size=(50, 2))
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.array([0] * 50 + [1] * 50)

    model = MultinomialNB().fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(0, 15, 200),
        np.linspace(0, 15, 200),
    )
    grid_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, grid_pred.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    ax.set_xlabel("Count of word A")
    ax.set_ylabel("Count of word B")
    ax.set_title("MultinomialNB decision boundary")

.. note::
    ``alpha`` controls Laplace smoothing. Higher values assign more
    probability mass to unseen feature/class combinations;
    ``alpha=0`` disables smoothing and can produce ``-inf``
    log-probabilities for a feature never observed in a class.
