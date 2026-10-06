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

.. note::
    The examples for the individual models focus on the API itself, so they use placeholder
    variables like ``X_train``, ``y_train``, and ``X_test`` to represent your own data.

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
