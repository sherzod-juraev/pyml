Cluster
=======

Clustering looks for groups in data that has no labels. Nobody tells the model which sample
belongs where, it has to find the structure from the distances between the samples. That makes
the result harder to judge than a prediction, so each page below compares the groups with what
is known about the data, and shows how much the answer depends on the settings. Both pages use
real datasets from `pyml-datasets <https://pyml-datasets.readthedocs.io/en/latest/>`_.

Where to start
--------------

* Open **K Means** first. It is the simplest, and it shows the ideas that the other model
  shares: distances, scaling and the question of how many groups there are.
* Open **DBSCAN** when you do not know how many groups there should be, when the groups are not
  round, or when some samples belong to no group at all.

The two models answer differently. K Means is told the number of groups and puts every sample
in one of them. DBSCAN is told what a crowded neighborhood is, finds the number of groups
itself, and leaves the samples in sparse places as noise.

Both measure distances, so the features must be standardized first. Each page shows what
happens when they are not.
