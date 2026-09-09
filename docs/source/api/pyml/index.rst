API Reference
=============

Full reference documentation for every class and function in pyml,
organized by module. Each page includes parameter descriptions,
mathematical derivations, and a worked example.

.. toctree::
    :hidden:

    core/index
    linear_model/index
    neighbors/index
    cluster/index
    preprocessing/index
    model_selection/index
    metrics/index

.. only:: html

    .. grid:: 1 2 2 3
        :gutter: 3
        :class-container: sd-mb-4

        .. grid-item-card:: Core
            :link: core/index
            :link-type: doc
            :text-align: center
            :shadow: md

            Shared base classes, exceptions, and type aliases used
            across all pyml estimators.

        .. grid-item-card:: Linear & Logistic Regression
            :link: linear_model/index
            :link-type: doc
            :text-align: center
            :shadow: md

            Gradient-descent-based regression and classification: OLS,
            Ridge, Lasso, Elastic Net, and their logistic counterparts.

        .. grid-item-card:: Nearest Neighbors
            :link: neighbors/index
            :link-type: doc
            :text-align: center
            :shadow: md

            Distance-based regression and classification: KNN and
            Radius Neighbors.

        .. grid-item-card:: Cluster
            :link: cluster/index
            :link-type: doc
            :text-align: center
            :shadow: md

            Unsupervised clustering algorithms: KMeans (with k-means++
            initialization) and DBSCAN.

        .. grid-item-card:: Preprocessing
            :link: preprocessing/index
            :link-type: doc
            :text-align: center
            :shadow: md

            Feature scaling transformers: StandardScaler, MinMaxScaler, and
            RobustScaler.

        .. grid-item-card:: Model Selection
            :link: model_selection/index
            :link-type: doc
            :text-align: center
            :shadow: md

            Splitting feature matrices and target arrays into train and
            test subsets for model evaluation.

        .. grid-item-card:: Metrics
            :link: metrics/index
            :link-type: doc
            :text-align: center
            :shadow: md

            Regression and classification evaluation metrics: MSE, R^2,
            accuracy, precision, recall, F1.
