Getting Started
===============

This page installs pyml and fits a first model. Every estimator in pyml follows the same
convention: ``fit`` learns from data, ``predict`` (or ``transform`` for preprocessing) uses what
was learned, and ``score`` measures how good the result is.

Install pyml
------------

pyml needs Python 3.12, 3.13 or 3.14. NumPy 1.26 or newer and SciPy 1.11.2 or newer are
installed automatically.

.. note::

   pyml is a personal project and is not published on PyPI, so it is installed directly from
   GitHub.

.. tab-set::

   .. tab-item:: pip

      Install the latest version of the ``main`` branch:

      .. code-block:: console

         $ pip install git+https://github.com/sherzod-juraev/pyml.git

      To install a fixed release, add its tag after ``@``, replacing ``vX.Y.Z`` with a tag
      from the `Releases <https://github.com/sherzod-juraev/pyml/releases>`_ page. Pinning a
      tag keeps your results reproducible, because the code no longer changes under you.

      .. code-block:: console

         $ pip install git+https://github.com/sherzod-juraev/pyml.git@vX.Y.Z

   .. tab-item:: uv

      With `uv <https://docs.astral.sh/uv/>`_, add pyml to your own project. This also records
      it in your ``pyproject.toml``:

      .. code-block:: console

         $ uv add git+https://github.com/sherzod-juraev/pyml.git

      To use a fixed release, pass its tag with ``--tag``, replacing ``vX.Y.Z`` with a tag from
      the `Releases <https://github.com/sherzod-juraev/pyml/releases>`_ page. Pinning a tag
      keeps your results reproducible, because the code no longer changes under you.

      .. code-block:: console

         $ uv add git+https://github.com/sherzod-juraev/pyml.git --tag vX.Y.Z

      To install into the active virtual environment only, without a project, use
      ``uv pip install`` with the same address, and ``@vX.Y.Z`` after it for a release:

      .. code-block:: console

         $ uv pip install git+https://github.com/sherzod-juraev/pyml.git

Install the example datasets
----------------------------

The examples in this documentation use the classic datasets from the companion package
`pyml-datasets <https://pyml-datasets.readthedocs.io/en/latest/>`_. It is not needed to use
pyml itself, only to run the examples as they are written:

.. tab-set::

   .. tab-item:: pip

      .. code-block:: console

         $ pip install git+https://github.com/sherzod-juraev/pyml-datasets.git

   .. tab-item:: uv

      .. code-block:: console

         $ uv add git+https://github.com/sherzod-juraev/pyml-datasets.git

Check the installation
----------------------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: console

         $ python -c "import pyml; print(pyml.__version__)"

   .. tab-item:: uv

      .. code-block:: console

         $ uv run python -c "import pyml; print(pyml.__version__)"

This prints the version of the installed release.

Fit a first model
-----------------

The example below trains a k-nearest neighbors classifier on the Iris dataset.

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
