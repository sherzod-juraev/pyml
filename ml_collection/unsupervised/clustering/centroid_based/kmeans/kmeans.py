import numpy as np
from typing import Literal
from scipy.spatial.distance import cdist
from .kmeans_pp import KmeansPP
from ml_collection.exception import NotFitted


class Kmeans:
    """
    K-Means clustering algorithm.

    This implementation supports both random (uniform) initialization
    and K-Means++ initialization. Clustering is performed by iteratively
    assigning points to the nearest centroid and updating centroids
    until convergence.

    Parameters
    ----------
    k : int, default=3
        Number of clusters.

    max_iter : int, default=100
        Maximum number of iterations.

    tol : float, default=1e-3
        Tolerance for convergence. Training stops when centroid
        movement is less than this value.

    init : {'uniform', 'kmeans++'}, default='kmeans++'
        Method for centroid initialization.

    metric : {'euclidean', 'chebyshev', 'cityblock'}, default='euclidean'
        Distance metric used for clustering.

    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    centroids : np.ndarray of shape (k, n_features)
        Cluster centers after fitting.

    __fitted : bool
        Indicates whether the model has been fitted.
    """
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
        """
        Fit the K-Means model to the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : Kmeans
            Fitted model.
        """

        self.__initialize_centroids(X)
        for i in range(self.max_iter):
            labels = self.__cal_labels(X)
            old_centroids = self.centroids.copy()
            self.__update_centroids(X, labels)
            if np.all(np.abs(old_centroids - self.centroids) < self.tol):
                break
        self.__fitted = True
        return self

    def __cal_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster indices for each sample.
        """
        centroids = self.centroids if self.centroids.ndim == 2 else np.array([self.centroids])
        distances = cdist(X, centroids, metric=self.metric)
        labels = np.argmin(distances, axis=1)
        return labels

    def __update_centroids(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Update centroids as the mean of assigned points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        labels : np.ndarray of shape (n_samples,)
            Cluster assignments.
        """

        for i in range(self.k):
            neighbors = X[labels == i]
            if neighbors.shape[0] != 0:
                self.centroids[i, :] = neighbors.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted cluster indices.

        Raises
        ------
        NotFitted
            If the model has not been fitted yet.
        """

        if not self.__fitted:
            raise NotFitted("Kmeans not fitted yet")
        labels = self.__cal_labels(X)
        return labels