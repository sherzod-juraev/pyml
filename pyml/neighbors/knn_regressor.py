r"""K-Nearest Neighbors Regressor.

This module provides a pure NumPy/SciPy implementation of the KNN
regression algorithm with support for multiple distance metrics
and weighting strategies.

Classes
-------
KNNRegressor
    Lazy-learning regressor using (weighted) average of k nearest
    neighbors' target values.

Notes
-----
KNN is a non-parametric, instance-based learning algorithm. No
explicit training occurs — all computation is deferred to predict
time. Uses ``argpartition`` for efficient neighbor selection in
:math:`O(n)` time rather than full sorting.
"""

from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from ..exc import NotFittedError


class KNNRegressor:
    r"""K-Nearest Neighbors Regressor.

    Predicts continuous target values by averaging the target values
    of the :math:`k` nearest neighbors in the training set.
    Given a query point :math:`x`, the algorithm:

    1. Computes distances from :math:`x` to all training points.
    2. Selects the :math:`k` closest neighbors.
    3. Aggregates their target values (uniform or distance-weighted).

    **Uniform weighting:**

    .. math::

        \hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_{(i)}

    **Distance weighting:**

    .. math::

        \hat{y} = \frac{\sum_{i=1}^{k} w_i \cdot y_{(i)}}
                         {\sum_{i=1}^{k} w_i},
        \quad w_i = \frac{1}{d(x, x_{(i)}) + \varepsilon}

    where :math:`\varepsilon = 10^{-12}` prevents division by zero
    for exact matches.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to consider for prediction.
    metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used to find nearest neighbors:

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan distance):

          .. math::

              d(x, x') = \sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

          .. math::

              d(x, x') = \max_{j} |x_j - x'_j|

    weighting : {'uniform', 'distance'}, default='uniform'
        Weighting strategy for neighbor aggregation:

        - ``'uniform'`` — all neighbors contribute equally.
        - ``'distance'`` — closer neighbors contribute more,
          weighted by inverse distance.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples, n_features)
        Training feature matrix stored after fitting.
    y_ : np.ndarray of shape (n_samples,)
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
    than :math:`O(n \log n)`.

    Examples
    --------
    >>> from pyml import KNNRegressor
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1., 2.], [2., 3.], [3., 4.], [6., 7.]])
    >>> y_train = np.array([1.5, 2.0, 3.5, 6.0])
    >>>
    >>> model = KNNRegressor(k=3, metric='euclidean', weighting='distance')
    >>> model.fit(X_train, y_train)
    >>> model.predict(np.array([[2., 2.]]))
    array([2.166...])
    """

    def __init__(
        self,
        k: int = 3,
        metric: Literal["euclidean", "chebyshev", "cityblock"] = "euclidean",
        weighting: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        r"""Initialize the KNN Regressor with hyperparameters.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors to consider for prediction.
            Must be positive and not exceed the number of training samples.
        metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
            Distance metric for computing pairwise distances between
            query points and training data.
        weighting : {'uniform', 'distance'}, default='uniform'
            Strategy for weighting neighbor contributions to the
            regression prediction.

        Returns
        -------
        None
        """
        self.k = k
        self.metric = metric
        self.weighting = weighting
        self.__fitted = False

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[Any]) -> Self:
        r"""Store training data for use during prediction.

        KNN is a lazy learner — no model is built during fit.
        Training data is stored and used directly at predict time
        for distance computation and neighbor lookup.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self : KNNRegressor
            Fitted instance with stored training data. Enables method
            chaining: ``model.fit(X, y).predict(X_test)``.
        """
        self.X_: npt.NDArray[np.float64] = np.asarray(X)
        self.y_: npt.NDArray[np.float64] = np.asarray(y)
        self.__fitted = True
        return self

    def __predict_regression(
        self, neighbor_ind: npt.NDArray[np.intp], neighbor_dist: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Aggregate neighbor target values into predictions.

        For ``'uniform'`` weighting computes the simple mean:

        .. math::

            \hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_{(i)}

        For ``'distance'`` weighting computes the weighted mean:

        .. math::

            \hat{y} = \frac{\sum_{i=1}^{k} w_i \cdot y_{(i)}}
                             {\sum_{i=1}^{k} w_i},
            \quad w_i = \frac{1}{d_i + 10^{-12}}

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
        if self.weighting == "uniform":
            return cast(npt.NDArray[np.float64], np.mean(self.y_[neighbor_ind], axis=1))
        weights = 1 / (neighbor_dist + 1e-12)
        y_pred = np.sum(self.y_[neighbor_ind] * weights, axis=1) / np.sum(
            weights, axis=1
        )
        return y_pred

    def check_k(self) -> None:
        r"""Validate that k is within the valid range.

        Ensures that :math:`0 < k \leq n\_samples` where
        :math:`n\_samples` is the number of training samples.

        Raises
        ------
        ValueError
            If ``k`` is less than or equal to 0, or if ``k`` exceeds
            the number of training samples.
        """
        n_samples: int = self.X_.shape[0]
        if self.k > n_samples or self.k <= 0:
            raise ValueError(
                f"Expected 0 < n_neighbors <= n_samples, but n_samples = {n_samples}, "
                f"n_neighbors = {self.k}."
            )

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Predict target values for new data points.

        Computes pairwise distances between query points and training
        data via :func:`scipy.spatial.distance.cdist`, selects the
        :math:`k` nearest neighbors using ``argpartition``, then
        aggregates their target values via :meth:`__predict_regression`.

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
        NotFittedError
            If ``predict`` is called before ``fit``. The model must
            be fitted before making predictions.
        ValueError
            If ``k`` is invalid (exceeds training set size or is
            non-positive), raised via :meth:`check_k`.
        """
        if not self.__fitted:
            raise NotFittedError(self)
        self.check_k()
        X = np.asarray(X)
        X = np.array([X]) if X.ndim == 1 else X
        dists = cdist(X, self.X_, metric=self.metric)
        neighbor_ind = np.argpartition(dists, kth=self.k - 1, axis=1)[:, : self.k]
        neighbor_dist = np.take_along_axis(dists, neighbor_ind, axis=1)
        return self.__predict_regression(neighbor_ind, neighbor_dist)
