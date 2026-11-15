Ensemble
========

Models that answer with the vote or the average of many decision trees. Each tree is grown on a
different random sample of the training data, so the trees make different mistakes, and the
combined answer is steadier than that of any single tree. The trees need no scaling of the
features, and the forest cannot be read question by question as one tree can.

Where to start
--------------

* :class:`~pyml.ensemble.RandomForestClassifier` predicts a class by a vote of the trees. It is
  the place to start: it is compared with a single tree on the penguins of the Tree family, and it
  shows how the vote of many trees gives the probability of each class.
* :class:`~pyml.ensemble.RandomForestRegressor` predicts a number as the average of the trees. It
  shows that a forest can use deep trees without the loss on new data that a single deep tree
  suffers, and that, like a tree, it cannot answer outside the range of the targets it has seen.

In both, ``max_depth`` is required and ``n_estimators`` is the number of trees. Judge a forest on
data that it has not seen, as a single tree.
