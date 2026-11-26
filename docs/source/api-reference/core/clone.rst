Clone
=====

``clone`` builds a new estimator with the same parameters and none of what the
original learned from data. It copies a meta-estimator such as
:doc:`/api-reference/pipeline/index` as a whole, so the estimators inside it are
copied too and the copy shares nothing with the original.

.. autofunction:: pyml.core.clone.clone
