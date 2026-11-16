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

            .. card-header::

                Base

            The root exception, PymlError, from which all other pyml
            exceptions derive.

            .. card-footer::

                .. button-ref:: base
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Base

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Fitting

            Errors related to the fit/predict lifecycle, such as calling
            predict before fit.

            .. card-footer::

                .. button-ref:: fitting
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Fitting

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Validation

            Errors related to invalid input, parameters, or shape
            mismatches.

            .. card-footer::

                .. button-ref:: validation
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Validation
