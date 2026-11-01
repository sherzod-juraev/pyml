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
    cluster/index
    preprocessing/index
    model-selection/index
    metrics/index

.. only:: html

    .. grid:: 1 2 2 3
        :gutter: 3
        :class-container: sd-mb-4

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Core
            ^^^^
            Shared base classes, exceptions, and type aliases used
            across all pyml estimators.

            +++
            .. button-ref:: core/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Core

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Linear & Logistic Regression
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Gradient-descent-based regression and classification: OLS,
            Ridge, Lasso, Elastic Net, and their logistic counterparts.

            +++
            .. button-ref:: linear-model/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Linear Models

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Naive Bayes
            ^^^^^^^^^^^
            Probabilistic classification via Bayes' theorem: GaussianNB
            for continuous features, MultinomialNB for count data.

            +++
            .. button-ref:: naive-bayes/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Naive Bayes

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Nearest Neighbors
            ^^^^^^^^^^^^^^^^^
            Distance-based regression and classification: KNN and
            Radius Neighbors.

            +++
            .. button-ref:: neighbors/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Nearest Neighbors

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Tree
            ^^^^
            Decision tree regression and classification via recursive,
            impurity-reducing splits.

            +++
            .. button-ref:: tree/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Tree

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Cluster
            ^^^^^^^
            Unsupervised clustering algorithms: KMeans (with k-means++
            initialization) and DBSCAN.

            +++
            .. button-ref:: cluster/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Cluster

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Preprocessing
            ^^^^^^^^^^^^^
            Feature scaling transformers: StandardScaler, MinMaxScaler, and
            RobustScaler.

            +++
            .. button-ref:: preprocessing/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Preprocessing

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Model Selection
            ^^^^^^^^^^^^^^^
            Splitting feature matrices and target arrays into train and
            test subsets for model evaluation.

            +++
            .. button-ref:: model-selection/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Model Selection

        .. grid-item-card::
            :text-align: center
            :shadow: md

            Metrics
            ^^^^^^^
            Regression and classification evaluation metrics: MSE, R^2,
            accuracy, precision, recall, F1.

            +++
            .. button-ref:: metrics/index
                :ref-type: doc
                :color: primary
                :expand:

                Open Metrics
