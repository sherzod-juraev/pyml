import numpy as np
from typing import Literal
from scipy.spatial.distance import cdist
from .kmeans_pp import KmeansPP
from ml_collection.exception import NotFitted


class Kmeans:

    def __init__(
            self,
            k: int = 3,
            max_iter: int = 100,
            tol: float | int = 1e-3,
            init: Literal['uniform', 'kmeans++'] = 'kmeans++',
            metric: Literal['euclidean', 'chebyshev', 'cityblock'] = 'euclidean',
            random_state: int | None = None
    ):

        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.metric = metric
        self.random_state = random_state
        self.__fitted = False

    def __initialize_centroids(self, X: np.ndarray) -> None:

        if self.init == 'uniform':
            rng = np.random.default_rng(self.random_state)
            ind = rng.choice(X.shape[0], size=self.k, replace=False)
            self.centroids = X[ind, :].copy()
        else:
            kmeanspp = KmeansPP(k=self.k, metric=self.metric, random_state=self.random_state)
            kmeanspp.initialize(X)
            self.centroids = kmeanspp.centroids

    def fit(self, X: np.ndarray) -> 'Kmeans':

        for i in range(self.max_iter):
            labels = self.__cal_labels(X)
            old_centroids = self.centroids.copy()
            self.__update_centroids(X, labels)
            if np.all(np.abs(old_centroids - self.centroids) < self.tol):
                break
        self.__fitted = True
        return self

    def __cal_labels(self, X: np.ndarray) -> np.ndarray:

        distances = cdist(X, self.centroids, metric=self.metric)
        labels = np.argmin(distances, axis=1)
        return labels

    def __update_centroids(self, X: np.ndarray, labels: np.ndarray) -> None:

        for i in range(self.k):
            neighbors = X[labels == i]
            if neighbors.shape[0] != 0:
                self.centroids[i, :] = neighbors.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.__fitted:
            raise NotFitted("Kmeans not fitted yet")
        labels = self.__cal_labels(X)
        return labels