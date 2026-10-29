Quick Start
===========

pyml follows the same fit/predict convention as scikit-learn: every
estimator implements ``fit(X, y)`` to learn parameters from data, and
``predict(X)`` to generate predictions from a fitted model.

A first model
-------------

The example below trains a k-nearest neighbors classifier on the Iris dataset. It needs the
companion package pyml-datasets, see :doc:`installation`.

.. code-block:: python

    from pyml.model_selection import train_test_split
    from pyml.neighbors import KNNClassifier
    from pyml_datasets import load_iris

    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNNClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(model.score(X_test, y_test))

``score(X, y)`` returns the accuracy for classifiers and the R² for regressors.

See what the model learned
--------------------------

A model with two features can be drawn. The plot below keeps only the petal length and the
petal width of the Iris flowers, fits the same classifier, and colors the plane by the class
that the model predicts at each point. The training samples are circles and the test samples
are triangles.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    from pyml.model_selection import train_test_split
    from pyml.neighbors import KNNClassifier
    from pyml_datasets import load_iris

    data = load_iris()
    X = data.X[:, 2:]  # petal length and petal width
    X_train, X_test, y_train, y_test = train_test_split(X, data.y, test_size=0.2, random_state=42)

    model = KNNClassifier(n_neighbors=5).fit(X_train, y_train)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
    )
    regions = model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots()
    colors = ListedColormap(["C0", "C1", "C2"])
    ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5, 2.5], cmap=colors, alpha=0.25)
    for label, name in enumerate(data.target_names):
        train = X_train[y_train == label]
        test = X_test[y_test == label]
        ax.scatter(train[:, 0], train[:, 1], c=f"C{label}", edgecolor="k", label=name)
        ax.scatter(test[:, 0], test[:, 1], c=f"C{label}", edgecolor="k", marker="^")
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title(f"KNNClassifier, test accuracy {model.score(X_test, y_test):.2f}")
    ax.legend(loc="upper left")

.. note::
    The examples for the individual models focus on the API itself, so they use placeholder
    variables like ``X_train``, ``y_train``, and ``X_test`` to represent your own data.

Next steps
----------

Continue with the installation guide, or browse the worked examples for every model.

.. toctree::
    :hidden:

    installation
    examples/index

.. only:: html

    .. grid:: 1 2 2 2
        :gutter: 3
        :padding: 0

        .. grid-item-card:: Installation
            :link: installation
            :link-type: doc
            :text-align: center
            :shadow: sm

            How to install pyml and its dependencies.

        .. grid-item-card:: Examples
            :link: examples/index
            :link-type: doc
            :text-align: center
            :shadow: sm

            Worked examples for every model in pyml.
