Naive Bayes
===========

.. toctree::
    :hidden:

    gaussian-nb
    multinomial-nb

.. only:: html

    .. grid:: 1 2 2 2
        :gutter: 3
        :padding: 0

        .. grid-item-card::
            :text-align: center
            :shadow: md

            GaussianNB
            ^^^^^^^^^^
            Classification assuming each feature is normally
            distributed within each class.

            +++
            .. button-ref:: gaussian-nb
                :ref-type: doc
                :color: primary
                :expand:

                Open GaussianNB

        .. grid-item-card::
            :text-align: center
            :shadow: md

            MultinomialNB
            ^^^^^^^^^^^^^
            Classification assuming each feature is a count (e.g.
            word frequencies), estimated via relative frequency with
            Laplace smoothing.

            +++
            .. button-ref:: multinomial-nb
                :ref-type: doc
                :color: primary
                :expand:

                Open MultinomialNB
