Core
====

.. toctree::
    :hidden:

    base/index
    exceptions/index
    protocols
    distance-metrics

.. only:: html

    .. grid:: 1 2 2 2
        :gutter: 3
        :padding: 0

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Base

            Base estimator classes (Regressor, Classifier, Transformer,
            Clusterer) that concrete models inherit from.

            .. card-footer::

                .. button-ref:: base/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Base

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Exceptions

            The pyml exception hierarchy, rooted at PymlError.

            .. card-footer::

                .. button-ref:: exceptions/index
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Exceptions

        .. grid-item-card::
            :text-align: center
            :shadow: md

            .. card-header::

                Protocols

            The method sets that parts of pyml rely on, such as what a
            Pipeline needs from its steps.

            .. card-footer::

                .. button-ref:: protocols
                    :ref-type: doc
                    :color: primary
                    :expand:

                    Open Protocols
