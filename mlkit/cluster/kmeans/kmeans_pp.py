import numpy as np
from typing import Literal, Self
from scipy.spatial.distance import cdist


class KmeansPP:
    """
    K-Means++ Centroid Initialization.

    Implements the **K-Means++** initialization strategy which selects
    initial centroids via distance-weighted sampling, significantly
    improving both convergence speed and solution quality compared to
    uniform random initialization.

    **Algorithm:**

    1. Choose the first centroid :math:`\\mu_1` uniformly at random
       from the training data.

    2. For each subsequent centroid :math:`\\mu_i`, :math:`i = 2, \\ldots, k`:

       a. Compute the minimum distance of each point to the nearest
          already-chosen centroid:

          .. math::

              D(x) = \\min_{j < i} \\, d(x, \\mu_j)

       b. Sample the next centroid with probability proportional to
          :math:`D(x)`:

          .. math::

              P(x) = \\frac{D(x)}{\\sum_{x'} D(x') + \\varepsilon}

       where :math:`\\varepsilon = 10^{-12}` prevents division by zero.

    This strategy ensures that initial centroids are **well-spread**,
    reducing the likelihood of poor local minima.

    Parameters
    ----------
    k : int, default=3
        Number of centroids to initialize.
    metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used to compute distances between points.

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \\sqrt{\\sum_{j=1}^{p}(x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan):

          .. math::

              d(x, x') = \\sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

    Parameters
    ----------
    k : int, default=3
        Number of centroids to initialize.
    metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used to compute distances between points.

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \\sqrt{\\sum_{j=1}^{p}(x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan):

          .. math::

              d(x, x') = \\sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:
     Examples
    --------
    >>> pp = KmeansPP(k=3, metric='euclidean', random_state=42)
    >>> pp.initialize(X_train)
    >>> initial_centroids = pp.centroids_
    """

    def __init__(
            self,
            k: int = 3,
            metric: Literal['euclidean', 'chebyshev', 'cityblock'] = 'euclidean',
            random_state: int | None = None
    ) -> None:

        self.k = k
        self.centroids_ = None
        self.metric = metric
        self.random_state = random_state

    def initialize(self, X: np.ndarray) -> Self:
        """
        Initialize :math:`k` centroids using the K-Means++ strategy.

        Selects the first centroid uniformly at random, then iteratively
        samples subsequent centroids with probability proportional to
        their squared distance from the nearest existing centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data from which centroids are sampled.

        Returns
        -------
        self : KmeansPP
            Instance with initialized ``centroids_``.
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

    def __cal_probab(self, centroids_: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute the sampling probability distribution for the next centroid.

        For each point :math:`x`, computes its minimum distance to any
        already-selected centroid, then normalizes to form a valid
        probability distribution:

        .. math::

            D(x_i) = \\min_{j} \\, d(x_i, \\mu_j)

            P(x_i) = \\frac{D(x_i)}{\\sum_{l} D(x_l) + 10^{-12}}

        Points far from existing centroids are sampled with higher
        probability, promoting spread-out initialization.

        Parameters
        ----------

        centroids\\_ : np.ndarray of shape (m, n_features)
            The :math:`m` centroids already selected.
        X : np.ndarray of shape (n_samples, n_features)
            Full training data.

        Returns
        -------
        probability : np.ndarray of shape (n_samples,)
            Normalized probability distribution over all training samples.
        """

        distances = cdist(centroids_, X, metric=self.metric)
        min_distances = np.min(distances, axis=0)
        probability = min_distances / (np.sum(min_distances) + 1e-12)
        return probability