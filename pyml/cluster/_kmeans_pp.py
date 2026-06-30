r"""K-Means++ centroid initialization strategy.

This module implements the K-Means++ algorithm for selecting
well-spread initial centroids via distance-weighted sampling,
significantly improving both convergence speed and final
clustering quality compared to uniform random initialization.

Classes
-------
KmeansPP
    Distance-weighted centroid initializer for K-Means clustering.

Notes
-----
The K-Means++ initialization is the default in scikit-learn's
``KMeans`` and is widely regarded as the standard approach for
initial centroid selection in practice.
"""

from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist


class KmeansPP:
    r"""K-Means++ Centroid Initialization.

    Implements the **K-Means++** initialization strategy which selects
    initial centroids via distance-weighted sampling, significantly
    improving both convergence speed and solution quality compared to
    uniform random initialization.

    **Algorithm:**

    1. Choose the first centroid :math:`\mu_1` uniformly at random
       from the training data.

    2. For each subsequent centroid :math:`\mu_i`, :math:`i = 2, \ldots, k`:

       a. Compute the minimum distance of each point to the nearest
          already-chosen centroid:

          .. math::

              D(x) = \min_{j < i} \, d(x, \mu_j)

        b. Sample the next centroid with probability proportional to
           the squared distance:

           .. math::

              P(x)=\frac{D(x)^2}{\sum_{x'} D(x')^2+\varepsilon}

       where :math:`\varepsilon = 10^{-12}` prevents division by zero.

    This strategy ensures that initial centroids are **well-spread**,
    reducing the likelihood of poor local minima.

    Parameters
    ----------
    k : int, default=3
        Number of centroids to initialize.
    metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used to compute distances between points:

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \sqrt{\sum_{j=1}^{p}(x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan):

          .. math::

              d(x, x') = \sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

          .. math::

              d(x, x') = \max_{j} |x_j - x'_j|

    random_state : int or None, default=None
        Seed for reproducible centroid initialization. Passed to
        ``numpy.random.default_rng``. If ``None``, randomness is
        not controlled.

    Attributes
    ----------
    centroids_ : np.ndarray of shape (k, n_features)
        Initialized centroids after calling ``initialize``. Each
        row is one centroid vector sampled from the training data.

    Notes
    -----
    **Why K-Means++?**

    Standard K-Means with random initialization is sensitive to
    the initial centroid positions — poor initialization can lead
    to slow convergence or suboptimal local minima. K-Means++
    mitigates this by ensuring initial centroids are spread across
    the data space:

    - The first centroid is chosen uniformly.
    - Each subsequent centroid is sampled with probability
      proportional to the squared distance from the nearest
      existing centroid.

    This creates a well-dispersed starting configuration that
    typically leads to faster convergence and lower final inertia.

    **Time complexity**

    Each initialization step computes :math:`O(m \cdot n \cdot p)`
    pairwise distances where :math:`m` is the number of centroids
    selected so far, :math:`n` is the number of samples, and
    :math:`p` is the number of features. For :math:`k` centroids,
    total complexity is :math:`O(k \cdot n \cdot p)`.

    Examples
    --------
    >>> from pyml import KmeansPP
    >>> import numpy as np
    >>>
    >>> X = np.array([[1., 2.], [1., 4.], [1., 0.],
    ...               [10., 2.], [10., 4.], [10., 0.]])
    >>>
    >>> pp = KmeansPP(k=3, metric='euclidean', random_state=42)
    >>> pp.initialize(X)
    >>> pp.centroids_
    array([[10.,  0.],
           [ 1.,  0.],
           [10.,  4.]])
    """

    def __init__(
        self,
        k: int = 3,
        metric: Literal["euclidean", "chebyshev", "cityblock"] = "euclidean",
        random_state: int | None = None,
    ) -> None:
        r"""Initialize the K-Means++ centroid selector.

        Parameters
        ----------
        k : int, default=3
            Number of centroids to select from the training data.
        metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
            Distance metric for computing point-to-centroid distances:

            - ``'euclidean'`` — Euclidean (L2) distance.
            - ``'cityblock'`` — Manhattan (L1) distance.
            - ``'chebyshev'`` — Chebyshev (L∞) distance.

        random_state : int or None, default=None
            Seed for the random number generator. Controls the random
            selection of the first centroid and the weighted sampling
            of subsequent centroids. Set to an integer for reproducible
            results.

        Returns
        -------
        None
        """
        self.k = k
        self.metric = metric
        self.random_state = random_state

    def initialize(self, X: npt.NDArray[np.float64]) -> Self:
        r"""Initialize :math:`k` centroids using the K-Means++ strategy.

        Selects the first centroid uniformly at random, then iteratively
        samples subsequent centroids with probability proportional
        to the squared distance from the nearest existing centroid. Points farther
        from all current centroids are more likely to be selected,
        ensuring good coverage of the data space.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data from which centroids are sampled. Centroids
            are chosen from the rows of this matrix.

        Returns
        -------
        self : KmeansPP
            The instance with the ``centroids_`` attribute populated.
            Enables method chaining.

        Notes
        -----
        The algorithm uses ``numpy.random.default_rng`` with the stored
        ``random_state`` for all stochastic operations. The probability
        distribution in each iteration is computed via :meth:`__cal_probab`.
        """
        rng = np.random.default_rng(self.random_state)
        self.centroids_ = np.zeros(shape=(self.k, X.shape[1]), dtype=X.dtype)
        first_ind = rng.integers(0, X.shape[0])
        self.centroids_[0, :] = X[first_ind, :].copy()
        for i in range(1, self.k):
            probability = self.__cal_probab(self.centroids_[:i, :], X)
            ind = rng.choice(X.shape[0], p=probability)
            self.centroids_[i, :] = X[ind, :].copy()
        return self

    def __cal_probab(
        self, centroids_: npt.NDArray[np.float64], X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Compute the sampling probability distribution for the next centroid.

        For each point :math:`x_i` in the dataset, computes the minimum
        distance to any already-selected centroid, then normalizes these
        distances to form a valid probability distribution over all points:

        .. math::

            D(x_i) = \min_{j} \, d(x_i, \mu_j)

        .. math::

            P(x_i)=
            \frac{D(x_i)^2}
            {\sum_{l=1}^{n} D(x_l)^2 + 10^{-12}}

        A small epsilon (:math:`10^{-12}`) is added to the denominator
        to prevent division by zero when all points coincide with existing
        centroids.

        Parameters
        ----------
        centroids_ : np.ndarray of shape (m, n_features)
            The :math:`m` centroids already selected in previous
            iterations, where :math:`1 \leq m < k`.
        X : np.ndarray of shape (n_samples, n_features)
            Full training data. Each row is a candidate for the next
            centroid.

        Returns
        -------
        probability : np.ndarray of shape (n_samples,)
            Normalized probability distribution over all training
            samples. Points farther from existing centroids have
            higher probability mass. Elements sum to 1.
        """
        distances = cdist(centroids_, X, metric=self.metric)
        min_distances = np.min(distances, axis=0) ** 2
        probability = min_distances / (np.sum(min_distances) + 1e-12)
        return probability
