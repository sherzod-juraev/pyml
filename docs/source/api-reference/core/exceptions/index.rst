Exceptions
==========

.. toctree::
    :hidden:

    base
    fitting
    validation

.. only:: html

    .. grid:: 1 2 2 3
        :gutter: 3
        :padding: 0

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Base
            ^^^^
            The root exception, PymlError, from which all other pyml
            exceptions derive.

            +++
            .. button-ref:: base
                :ref-type: doc
                :color: primary
                :expand:

                Open Base

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Fitting
            ^^^^^^^
            Errors related to the fit/predict lifecycle, such as calling
            predict before fit.

            +++
            .. button-ref:: fitting
                :ref-type: doc
                :color: primary
                :expand:

                Open Fitting

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Validation
            ^^^^^^^^^^
            Errors related to invalid input, parameters, or shape
            mismatches.

            +++
            .. button-ref:: validation
                :ref-type: doc
                :color: primary
                :expand:

                Open Validation
