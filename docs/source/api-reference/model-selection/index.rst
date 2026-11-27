Model Selection
===============

.. toctree::
    :hidden:

    train-test-split
    kfold
    cross-val-score

.. only:: html

    .. grid:: 1 2 2 3
        :gutter: 3
        :padding: 0

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Train Test Split

            Split feature matrices and target arrays into random train
            and test subsets for robust model evaluation.

            .. card-footer::

                .. button-ref:: train-test-split
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Train Test Split

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                K Fold

            Split sample indices into consecutive train and validation folds,
            so that every sample is validated exactly once.

            .. card-footer::

                .. button-ref:: kfold
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open K Fold

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Cross Val Score

            Score an estimator or a pipeline on the validation folds of a
            cross-validation, with a new unfitted copy for every fold.

            .. card-footer::

                .. button-ref:: cross-val-score
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Cross Val Score
