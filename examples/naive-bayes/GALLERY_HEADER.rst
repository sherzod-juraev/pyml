Naive Bayes
===========

Naive Bayes classifiers use Bayes' rule. For a new sample they ask how likely each class is on
its own, and how likely the measured features are inside that class, and they pick the class
where the product is the largest. To keep the second part simple they assume that the features
are independent of each other. That is the "naive" part. It is rarely true, and the models still
work well as fast baselines, because the fit is a single pass over the data and needs few samples.

The two models differ in what they assume about a feature. Each page below fits one of them on a
real dataset from `pyml-datasets <https://pyml-datasets.readthedocs.io/en/latest/>`_ and shows
where the independence assumption helps and where it costs.

Where to start
--------------

* Open **Gaussian Naive Bayes** for features that are measurements, such as lengths and weights.
  It models each feature inside a class with a normal distribution, and its page shows the idea
  step by step.
* Open **Multinomial Naive Bayes** for features that are counts, such as the number of times a
  word appears in a text. It models how much of its total each class puts into each feature.

There is nothing to iterate, so there is no ``learning_rate``, and Gaussian Naive Bayes does not
need standardized features. Multinomial Naive Bayes needs the opposite care: its features must
not be negative, so they must not be standardized.
