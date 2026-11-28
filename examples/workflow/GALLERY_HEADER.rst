Machine Learning Workflow
=========================

The tools around a model, not a model itself. These pages show how to split the data, chain the
preprocessing and a model into one estimator, and measure the result. They use the same models
as the rest of the examples, only as something to measure.

Where to start
--------------

* **Cross-Validation** measures a model on several different splits instead of one, with
  :class:`~pyml.model_selection.KFold`, :class:`~pyml.pipeline.Pipeline` and
  :func:`~pyml.model_selection.cross_val_score`. Start here when you want to compare two models.
