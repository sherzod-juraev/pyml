API Reference
=============

Full reference documentation for every class and function in pyml,
organized by module rather than alphabetically: supervised models first,
then unsupervised, then utilities. The same order is used in
:doc:`/examples/index`. Use the sidebar search or
:kbd:`Ctrl+K` to jump directly to a specific model.

.. toctree::
    :hidden:

    core/index
    linear-model/index
    naive-bayes/index
    neighbors/index
    tree/index
    ensemble/index
    cluster/index
    preprocessing/index
    pipeline/index
    model-selection/index
    metrics/index

.. only:: html

    .. grid:: 1 2 2 3
        :gutter: 3
        :class-container: sd-mb-4

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Core

            Shared base classes, exceptions, protocols, and type aliases
            used across all pyml estimators, and clone, which copies an
            estimator without what it has learned.

            .. card-footer::

                .. button-ref:: core/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Core

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Linear & Logistic Regression

            Gradient-descent-based regression and classification: OLS,
            Ridge, Lasso, Elastic Net, and their logistic counterparts.

            .. card-footer::

                .. button-ref:: linear-model/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Linear Models

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Naive Bayes

            Probabilistic classification via Bayes' theorem: GaussianNB
            for continuous features, MultinomialNB for count data.

            .. card-footer::

                .. button-ref:: naive-bayes/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Naive Bayes

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Nearest Neighbors

            Distance-based regression and classification: KNN and
            Radius Neighbors.

            .. card-footer::

                .. button-ref:: neighbors/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Nearest Neighbors

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Tree

            Decision tree regression and classification via recursive,
            impurity-reducing splits.

            .. card-footer::

                .. button-ref:: tree/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Tree

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Ensemble

            Random forest regression and classification: averaged decision
            trees grown on bootstrap samples with random feature subsets.

            .. card-footer::

                .. button-ref:: ensemble/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Ensemble

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Cluster

            Unsupervised clustering algorithms: KMeans (with k-means++
            initialization) and DBSCAN.

            .. card-footer::

                .. button-ref:: cluster/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Cluster

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Preprocessing

            Feature scaling transformers: StandardScaler, MinMaxScaler, and
            RobustScaler.

            .. card-footer::

                .. button-ref:: preprocessing/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Preprocessing

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Pipeline

            Chains transformers and a final model or clusterer into one
            estimator.

            .. card-footer::

                .. button-ref:: pipeline/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Pipeline

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Model Selection

            Train and test splitting, K-fold cross-validation, and
            cross-validated scores for model evaluation.

            .. card-footer::

                .. button-ref:: model-selection/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Model Selection

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Metrics

            Regression and classification evaluation metrics: MSE, :math:`R^2`,
            accuracy, precision, recall, F1.

            .. card-footer::

                .. button-ref:: metrics/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Metrics
