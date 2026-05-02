import numpy as np
from typing import Literal
from scipy.spatial.distance import cdist
from mlkit.exc import NotFitted


class KNNRegression:
    """
    KNNRegression - k-Nearest Neighbors regress model.

    Implements classic KNN regress with optional distance-based weighting.
    Predictions can be computed using uniform weights or inversely proportional to distance.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to consider.

    metric : {'euclidean', 'chebyshev', 'cityblock'}, default='euclidean'
        Distance metric to use:
        - 'euclidean' : Euclidean distance
        - 'chebyshev' : Chebyshev distance
        - 'manhattan' : Manhattan distance (corresponds to 'cityblock' in scipy cdist)

    weighting : {'uniform', 'distance'}, default='uniform'
        Weighting method for neighbors:
        - 'uniform'  : all neighbors contribute equally
        - 'distance' : closer neighbors contribute more (inverse of distance)

    Attributes
    ----------
    X_ : np.ndarray
        Training feature data stored after fitting.
    y_ : np.ndarray
        Training target values stored after fitting.
    __fitted : bool
        Flag indicating whether the model has been fitted.
    """

    def __init__(
            self,
            k: int = 3,
            metric: Literal['euclidean', 'chebyshev', 'cityblock'] = 'euclidean',
            weighting: Literal['uniform', 'distance'] = 'uniform'
    ):

        self.k = k
        self.metric = metric
        self.weighting = weighting
        self.__fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNRegression':
        """
        Fit the KNN regress model using training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature vectors.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNNRegression
            Fitted instance.
        """

        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        self.__fitted = True
        return self

    def __predict_regression(self, neighbor_ind: np.ndarray, neighbor_dist: np.ndarray) -> np.ndarray:
        """
        Compute regress predictions based on neighbor indices and distances.

        Parameters
        ----------
        neighbor_ind : np.ndarray of shape (n_samples, k)
            Indices of the k nearest neighbors for each sample.
        neighbor_dist : np.ndarray of shape (n_samples, k)
            Distances of the k nearest neighbors for each sample.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """

        if self.weighting == 'uniform':
            return np.mean(self.y_[neighbor_ind], axis=1)
        weights = 1 / (neighbor_dist + 1e-12)
        y_pred = np.sum(self.y_[neighbor_ind] * weights, axis=1) / np.sum(weights, axis=1)
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new data points using the fitted KNN model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Input samples for which to predict target values.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        NotFitted
            If the model has not been fitted yet.
        """

        if not self.__fitted:
            raise NotFitted("KNNRegression not fitted yet")
        X = np.asarray(X)
        X = np.array([X]) if X.ndim == 1 else X
        dists = cdist(X, self.X_, metric=self.metric)
        neighbor_ind = np.argpartition(dists, kth=self.k - 1, axis=1)[:,:self.k]
        neighbor_dist = np.take_along_axis(dists, neighbor_ind, axis=1)
        return self.__predict_regression(neighbor_ind, neighbor_dist)