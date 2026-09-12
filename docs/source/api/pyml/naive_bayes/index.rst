Naive Bayes
===========

.. toctree::
    :hidden:

    gaussian_nb
    multinomial_nb

.. only:: html

    .. grid:: 1 2 2 2
        :gutter: 3
        :padding: 0

        .. grid-item-card:: GaussianNB
            :link: gaussian_nb
            :link-type: doc
            :text-align: center
            :shadow: sm

            Classification assuming each feature is normally
            distributed within each class.

        .. grid-item-card:: MultinomialNB
            :link: multinomial_nb
            :link-type: doc
            :text-align: center
            :shadow: sm

            Classification assuming each feature is a count (e.g.
            word frequencies), estimated via relative frequency with
            Laplace smoothing.
