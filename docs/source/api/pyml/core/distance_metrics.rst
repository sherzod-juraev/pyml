Distance Metrics
=================

Reference for the distance metrics used across pyml — currently by
the :doc:`/api/pyml/neighbors/index` models and
:doc:`/api/pyml/cluster/dbscan`. Each metric defines a different
notion of "distance" between two points
:math:`x = (x_1, \dots, x_n)` and :math:`y = (y_1, \dots, y_n)`.

.. _metric-euclidean:

Euclidean
---------

Straight-line ("as the crow flies") distance:

.. math::
    d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}

The default choice for continuous, similarly-scaled features.

.. _metric-manhattan:

Manhattan (City Block)
-------------------------

Sum of absolute differences along each axis — the distance a taxi
would drive on a grid of city blocks, rather than in a straight
line:

.. math::
    d(x, y) = \sum_{i=1}^{n} |x_i - y_i|

Less sensitive to outliers in a single feature than Euclidean,
since differences aren't squared.

.. _metric-chebyshev:

Chebyshev
---------

The greatest single-axis difference — distance measured as if
diagonal movement costs the same as axis-aligned movement (like a
king's move in chess):

.. math::
    d(x, y) = \max_{i} |x_i - y_i|

Useful when the worst-case difference across any one dimension
matters more than the combined difference across all of them.
