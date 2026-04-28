from scipy.spatial.distance import cdist
from scipy import stats
from typing import Literal
import numpy as np
from ml_collection.exception import NotFitted


class KNNClassifier:
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to consider for classification.
    metric : {'euclidean', 'cityblock', 'chebyshev', 'cosine'}, default='euclidean'
        Distance metric to use for calculating distances between points.
    weighting : {'uniform', 'distance'}, default='uniform'
        Weighting method for neighbor votes:
        - 'uniform': each neighbor has equal weight
        - 'distance': closer neighbors have higher weight (1 / distance)

    Attributes
    ----------
    X_train : np.ndarray
        Training feature matrix stored after fit.
    y_train : np.ndarray
        Training labels stored after fit.
    __fitted : bool
        Flag indicating whether the classifier has been fitted.
    """

    def __init__(
            self,
            k: int = 3,
            metric: Literal['euclidean', 'cityblock', 'chebyshev', 'cosine'] = 'euclidean',
            weighting: Literal['uniform', 'distance'] = 'uniform'
    ):

        self.k = k
        self.metric = metric
        self.weighting = weighting
        self.__fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the KNN classifier with training data.

        Parameters
        ----------
        X : np.ndarray
            Training data features of shape (n_samples, n_features).
        y : np.ndarray
            Training data labels of shape (n_samples,).

        Returns
        -------
        self : KNNClassifier
            Fitted classifier instance.
        """

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.__fitted = True
        return self

    def __predict_label(
            self,
            neigh_ind: np.ndarray,
            neigh_dist: np.ndarray
    ) -> np.ndarray:
        """
        Predict labels for given neighbor indices and distances.

        Parameters
        ----------
        neigh_ind : np.ndarray
            Array of neighbor indices for each sample.
        neigh_dist : np.ndarray
            Array of distances corresponding to each neighbor.

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels for each sample.
        """

        labels = self.y_train[neigh_ind]
        if self.weighting == 'uniform':
            y_pred = stats.mode(labels, axis=1, keepdims=False)[0]
            return y_pred
        weights = 1 / (neigh_dist + 1e-12)
        y_pred = np.zeros(shape=neigh_dist.shape[0])
        for i in range(neigh_dist.shape[0]):
            uniq_labels = np.unique(labels[i])
            votes = np.zeros(shape=uniq_labels.shape[0])
            for j in range(uniq_labels.shape[0]):
                votes[j] = np.sum(weights[i, labels[i] == uniq_labels[j]])
            y_pred[i] = uniq_labels[np.argmax(votes)]
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels of shape (n_samples,).

        Raises
        ------
        NotFitted
            If the classifier has not been fitted yet.
        """

        if not self.__fitted:
            raise NotFitted(f"KNNClassifier not fitted yet")
        X = np.asarray(X)
        X = np.array([X]) if X.ndim == 1 else X
        dist = cdist(X, self.X_train, self.metric)
        neigh_ind = np.argpartition(dist, self.k - 1, axis=1)[:, :self.k]
        neigh_dist = np.take_along_axis(dist, neigh_ind, axis=1)
        return self.__predict_label(neigh_ind, neigh_dist)