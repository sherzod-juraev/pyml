Neighbors
=========

Models that predict from the training samples that lie closest to the new one. They learn
nothing in ``fit``: the training data is kept, and every prediction looks it up. Because the
answer depends on distances, scale the features first.

Where to start
--------------

* Pick **K nearest neighbors** when you want the same number of neighbors for every sample.
  It is the safer default. :class:`~pyml.neighbors.KNNClassifier` predicts a class and
  :class:`~pyml.neighbors.KNNRegressor` predicts a number.
* Pick **radius neighbors** when a distance has a meaning of its own and the data is spread
  evenly. :class:`~pyml.neighbors.RadiusNeighborsClassifier` and
  :class:`~pyml.neighbors.RadiusNeighborsRegressor` use every sample within the radius, so
  decide what to do with a sample that has none.
