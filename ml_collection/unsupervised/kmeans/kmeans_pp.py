import numpy as np
from typing import Literal
from scipy.spatial.distance import cdist


class KmeansPP:

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

        distances = cdist(centroids, X, metric=self.metric)
        min_distances = np.min(distances, axis=0)
        probability = min_distances / (np.sum(min_distances) + 1e-12)
        return probability