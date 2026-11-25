Protocols
=========

A protocol names the methods that one part of pyml needs from another, so
the classes that provide them do not have to inherit from it. The
:doc:`/api-reference/pipeline/index`, for example, accepts any
transformer and any model that has the methods listed here.

The protocols can also be used with ``isinstance``. That check only looks
that the methods exist by name and does not compare their signatures.

Supports Params
---------------

.. autoclass:: pyml.core.protocols.SupportsParams
    :members:
    :show-inheritance:

Transformer Like
----------------

.. autoclass:: pyml.core.protocols.TransformerLike
    :members:
    :show-inheritance:

Supervised Like
---------------

.. autoclass:: pyml.core.protocols.SupervisedLike
    :members:
    :show-inheritance:

Clusterer Like
--------------

.. autoclass:: pyml.core.protocols.ClustererLike
    :members:
    :show-inheritance:

Predictable Clusterer Like
--------------------------

.. autoclass:: pyml.core.protocols.PredictableClustererLike
    :members:
    :show-inheritance:

Supports Predict Proba
----------------------

.. autoclass:: pyml.core.protocols.SupportsPredictProba
    :members:
    :show-inheritance:
