Tree
====

Models that predict by asking a series of yes/no questions about the features. Every sample
follows the questions down the tree to a leaf, and the leaf holds the answer. A tree needs no
scaling of the features, and the fitted model can be read as a short list of rules.

Where to start
--------------

* :class:`~pyml.tree.DecisionTreeClassifier` predicts a class. It is the place to start, since the
  questions of a tree are easiest to follow when the answer is a species or a yes or no.
* :class:`~pyml.tree.DecisionTreeRegressor` predicts a number as a staircase of averages. It also
  shows what a tree cannot do: it never gives an answer outside the range of the training targets.

In both, ``max_depth`` is the choice that matters. A tree that is too deep fits the training data by
heart, so judge it on data that it has not seen.
