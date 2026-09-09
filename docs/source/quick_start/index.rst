Quick Start
===========

pyml follows the same fit/predict convention as scikit-learn: every
estimator implements ``fit(X, y)`` to learn parameters from data, and
``predict(X)`` to generate predictions from a fitted model.

.. note::
    Examples throughout this documentation use placeholder variables
    like ``X_train``, ``y_train``, and ``X_test`` to represent your
    own data. No dataset-loading code is shown — the focus here is
    on the API itself, not on where the data comes from.

.. code-block:: python

    from pyml.linear_model import Ridge

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

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
