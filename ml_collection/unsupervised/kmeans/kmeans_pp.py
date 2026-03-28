import numpy as np
from typing import Literal
from scipy.spatial.distance import cdist


class KmeansPP:
    """
    K-Means++ initialization algorithm.

    This class is responsible for initializing cluster centroids using
    the K-Means++ strategy, which improves convergence by selecting
    initial centroids based on distance-weighted probability.

    Parameters
    ----------
    k : int, default=3
        Number of clusters (centroids) to initialize.

    metric : {'euclidean', 'chebyshev', 'cityblock'}, default='euclidean'
        Distance metric used to compute distances between points.

    random_state : int or None, default=None
        Seed for random number generator to ensure reproducibility.

    Attributes
    ----------
    centroids : np.ndarray of shape (k, n_features)
        Initialized cluster centers.
    """
    def __init__(
            self,
            k: int = 3,
            metric: Literal['euclidean', 'chebyshev', 'cityblock'] = 'euclidean',
            random_state: int | None = None
    ):

        self.k = k
        self.centroids = None
        self.metric = metric
        self.random_state = random_state

    def initialize(self, X: np.ndarray) -> 'KmeansPP':
        """
        Initialize centroids using K-Means++ algorithm.

        The first centroid is chosen randomly. Each subsequent centroid is
        selected with probability proportional to the distance from the
        nearest existing centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : KmeansPP
            Fitted instance with initialized centroids.
        """

        rng = np.random.default_rng(self.random_state)
        self.centroids = np.zeros(shape=(self.k, X.shape[1]), dtype=X.dtype)
        first_ind = rng.integers(0, X.shape[0])
        self.centroids[0, :] = X[first_ind, :].copy()
        for i in range(1, self.k):
            probability = self.__cal_probab(self.centroids[:i, :], X)
            ind = rng.choice(X.shape[0], p=probability)
            self.centroids[i, :] = X[ind, :].copy()
        return self

    def __cal_probab(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute probability distribution for selecting next centroid.

        Probability is proportional to the minimum distance of each point
        to the nearest existing centroid.

        Parameters
        ----------
        centroids : np.ndarray of shape (m, n_features)
            Already selected centroids.

        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        probability : np.ndarray of shape (n_samples,)
            Probability distribution over samples.
        """

        distances = cdist(centroids, X, metric=self.metric)
        min_distances = np.min(distances, axis=0)
        probability = min_distances / (np.sum(min_distances) + 1e-12)
        return probability