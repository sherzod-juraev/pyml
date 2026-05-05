import numpy as np
from typing import Literal, Self
from scipy.spatial.distance import cdist
from ...exc import NotFitted


class KNNRegression:
    """
    K-Nearest Neighbors Regressor.

    Predicts continuous target values by averaging the target values
    of the :math:`k` nearest neighbors in the training set.
    Given a query point :math:`x`, the algorithm:

    1. Computes distances from :math:`x` to all training points.
    2. Selects the :math:`k` closest neighbors.
    3. Aggregates their target values (uniform or distance-weighted).

    **Uniform weighting:**

    .. math::

        \\hat{y} = \\frac{1}{k} \\sum_{i=1}^{k} y_{(i)}

    **Distance weighting:**

    .. math::

        \\hat{y} = \\frac{\\sum_{i=1}^{k} w_i \\cdot y_{(i)}}
                         {\\sum_{i=1}^{k} w_i},
        \\quad w_i = \\frac{1}{d(x,\\, x_{(i)}) + \\varepsilon}

    where :math:`\\varepsilon = 10^{-12}` prevents division by zero
    for exact matches.

    Parameters
    ----------

    k : int, default=3
        Number of nearest neighbors to consider for prediction.
    metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used to find nearest neighbors.


        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \\sqrt{\\sum_{j=1}^{p} (x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan distance):

          .. math::

              d(x, x') = \\sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

          .. math::

              d(x, x') = \\max_{j} |x_j - x'_j|

    weighting : {'uniform', 'distance'}, default='uniform'
        Weighting strategy for neighbor aggregation.

        - ``'uniform'`` — all neighbors contribute equally.
        - ``'distance'`` — closer neighbors contribute more,
          weighted by inverse distance.

    Attributes
    ----------
    X\\_ : np.ndarray of shape (n_samples, n_features)
        Training feature matrix stored after fitting.
    y\\_ : np.ndarray of shape (n_samples,)
        Training target values stored after fitting.

    Notes
    -----
    KNN regression is a non-parametric, lazy learning algorithm —
    no explicit training occurs. All computation happens at predict time,
    making it memory-intensive but flexible.

    Performance degrades in high-dimensional spaces due to the
    **curse of dimensionality**: distances become increasingly uniform
    as dimensionality grows.

    ``argpartition`` is used instead of full sorting for efficiency:
    it finds the :math:`k` smallest distances in :math:`O(n)` rather
    than :math:`O(n \\log n)`.

    Examples
    --------
    >>> model = KNNRegression(k=5, metric='euclidean', weighting='distance')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Store training data for use during prediction.

        KNN is a lazy learner — no model is built during fit.
        Training data is stored and used directly at predict time.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self : KNNRegression
            Fitted instance with stored training data.
        """

        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        self.__fitted = True
        return self

    def __predict_regression(self, neighbor_ind: np.ndarray, neighbor_dist: np.ndarray) -> np.ndarray:
        """
        Aggregate neighbor target values into predictions.

        For ``'uniform'`` weighting computes the simple mean:

        .. math::

            \\hat{y} = \\frac{1}{k} \\sum_{i=1}^{k} y_{(i)}

        For ``'distance'`` weighting computes the weighted mean:

        .. math::

            \\hat{y} = \\frac{\\sum_{i=1}^{k} w_i \\cdot y_{(i)}}
                             {\\sum_{i=1}^{k} w_i},
            \\quad w_i = \\frac{1}{d_i + 10^{-12}}

        Parameters
        ----------
        neighbor_ind : np.ndarray of shape (n_samples, k)
            Indices of the k nearest neighbors for each query point.
        neighbor_dist : np.ndarray of shape (n_samples, k)
            Distances to the k nearest neighbors for each query point.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted continuous target values.
        """

        if self.weighting == 'uniform':
            return np.mean(self.y_[neighbor_ind], axis=1)
        weights = 1 / (neighbor_dist + 1e-12)
        y_pred = np.sum(self.y_[neighbor_ind] * weights, axis=1) / np.sum(weights, axis=1)
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new data points.

        Computes pairwise distances between query points and training
        data, selects the :math:`k` nearest neighbors using
        ``argpartition``, then aggregates their target values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Query points. 1-D input is automatically reshaped to
            ``(1, n_features)``.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted continuous target values.

        Raises
        ------
        NotFitted
            If called before fitting the model.
        """

        if not self.__fitted:
            raise NotFitted(self)
        X = np.asarray(X)
        X = np.array([X]) if X.ndim == 1 else X
        dists = cdist(X, self.X_, metric=self.metric)
        neighbor_ind = np.argpartition(dists, kth=self.k - 1, axis=1)[:,:self.k]
        neighbor_dist = np.take_along_axis(dists, neighbor_ind, axis=1)
        return self.__predict_regression(neighbor_ind, neighbor_dist)