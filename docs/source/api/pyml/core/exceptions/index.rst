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

        .. grid-item-card:: Base
            :link: base
            :link-type: doc
            :text-align: center
            :shadow: sm

            The root exception, PymlError, from which all other pyml
            exceptions derive.

        .. grid-item-card:: Fitting
            :link: fitting
            :link-type: doc
            :text-align: center
            :shadow: sm

            Errors related to the fit/predict lifecycle, such as calling
            predict before fit.

        .. grid-item-card:: Validation
            :link: validation
            :link-type: doc
            :text-align: center
            :shadow: sm

            Errors related to invalid input, parameters, or shape
            mismatches.
