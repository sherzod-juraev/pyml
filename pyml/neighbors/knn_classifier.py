r"""K-Nearest Neighbors Classifier.

This module provides a pure NumPy/SciPy implementation of the KNN
classification algorithm with support for multiple distance metrics
and weighting strategies.

Classes
-------
KNNClassifier
    Lazy-learning classifier using majority vote among k nearest
    neighbors.

Notes
-----
KNN is a non-parametric, instance-based learning algorithm. No
explicit training occurs — all computation is deferred to predict
time. Uses ``argpartition`` for efficient neighbor selection in
:math:`O(n)` time rather than full sorting.
"""

from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from ..exc import NotFittedError


class KNNClassifier:
    r"""K-Nearest Neighbors Classifier.

    Predicts class labels by majority vote among the :math:`k` nearest
    neighbors in the training set. Given a query point :math:`x`,
    the algorithm:

    1. Computes distances from :math:`x` to all training points.
    2. Selects the :math:`k` closest neighbors.
    3. Assigns the class with the highest total vote weight.

    **Uniform weighting** — each neighbor votes equally:

    .. math::

        \hat{y} = \arg\max_{c} \sum_{i=1}^{k}
        \mathbf{1}[y_{(i)} = c]

    **Distance weighting** — closer neighbors vote more strongly:

    .. math::

        \hat{y} = \arg\max_{c} \sum_{i=1}^{k}
        w_i \cdot \mathbf{1}[y_{(i)} = c],
        \quad w_i = \frac{1}{d(x, x_{(i)}) + \varepsilon}

    where :math:`\varepsilon = 10^{-12}` prevents division by zero
    for exact matches.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to consider for classification.
    metric : {'euclidean', 'cityblock', 'chebyshev', 'cosine'}, default='euclidean'
        Distance metric used to find nearest neighbors:

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \sqrt{\sum_{j=1}^{p}(x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan):

          .. math::

              d(x, x') = \sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

          .. math::

              d(x, x') = \max_j |x_j - x'_j|

        - ``'cosine'`` — angular similarity:

          .. math::

              d(x, x') = 1 - \frac{x \cdot x'}{\|x\| \|x'\|}

    weighting : {'uniform', 'distance'}, default='uniform'
        Weighting strategy for neighbor votes:

        - ``'uniform'`` — all neighbors contribute equally.
        - ``'distance'`` — closer neighbors have higher influence,
          weighted by inverse distance.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples, n_features)
        Training feature matrix stored after fitting.
    y_ : np.ndarray of shape (n_samples,)
        Training class labels stored after fitting.

    Notes
    -----
    KNN is a non-parametric, lazy learning algorithm — no explicit
    model is trained. All computation happens at predict time.

    ``argpartition`` is used instead of full sorting for efficiency:
    selecting :math:`k` nearest neighbors costs :math:`O(n)` rather
    than :math:`O(n \log n)`.

    Performance degrades in high-dimensional spaces due to the
    **curse of dimensionality**: distances become increasingly uniform
    as the number of features grows.

    Examples
    --------
    >>> from pyml import KNNClassifier
    >>> import numpy as np
    >>>
    >>> X_train = np.array([[1., 2.], [2., 3.], [3., 4.], [6., 7.]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>>
    >>> model = KNNClassifier(k=3, metric='euclidean', weighting='distance')
    >>> model.fit(X_train, y_train)
    >>> model.predict(np.array([[2., 2.]]))
    array([0])
    """

    def __init__(
        self,
        k: int = 3,
        metric: Literal["euclidean", "cityblock", "chebyshev", "cosine"] = "euclidean",
        weighting: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        r"""Initialize the KNN Classifier with hyperparameters.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors to consider for classification.
            Must be positive and not exceed the number of training samples.
        metric : {'euclidean', 'cityblock', 'chebyshev', 'cosine'}, default='euclidean'
            Distance metric for computing pairwise distances between
            query points and training data.
        weighting : {'uniform', 'distance'}, default='uniform'
            Strategy for weighting neighbor contributions to the
            classification decision.

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
            Training class labels.

        Returns
        -------
        self : KNNClassifier
            Fitted instance with stored training data. Enables method
            chaining: ``model.fit(X, y).predict(X_test)``.
        """
        self.X_: npt.NDArray[np.float64] = np.asarray(X)
        self.y_: npt.NDArray[Any] = np.asarray(y)
        self.__fitted = True
        return self

    def __predict_label(
        self, neigh_ind: npt.NDArray[np.intp], neigh_dist: npt.NDArray[np.float64]
    ) -> npt.NDArray[Any]:
        r"""Aggregate neighbor votes into predicted class labels.

        For ``'uniform'`` weighting, uses majority vote via
        ``numpy.unique`` with counts.

        For ``'distance'`` weighting, computes weighted votes per
        unique class:

        .. math::

            \text{vote}(c) = \sum_{i: y_{(i)}=c} w_i,
            \quad w_i = \frac{1}{d_i + 10^{-12}}

        and selects the class with the highest total weight.

        Parameters
        ----------
        neigh_ind : np.ndarray of shape (n_samples, k)
            Indices of the k nearest neighbors for each query point.
        neigh_dist : np.ndarray of shape (n_samples, k)
            Distances to the k nearest neighbors for each query point.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        labels = self.y_[neigh_ind]
        y_pred = np.zeros(shape=labels.shape[0], dtype=self.y_.dtype)
        if self.weighting == "uniform":
            for i in range(labels.shape[0]):
                unique_labels, counts = np.unique(labels[i], return_counts=True)
                y_pred[i] = unique_labels[np.argmax(counts)]
            return y_pred
        weights: npt.NDArray[np.float64] = 1 / (neigh_dist + 1e-12)
        for i in range(neigh_dist.shape[0]):
            uniq_labels = np.unique(labels[i])
            votes = np.zeros(shape=uniq_labels.shape[0])
            for j in range(uniq_labels.shape[0]):
                votes[j] = np.sum(weights[i, labels[i] == uniq_labels[j]])
            y_pred[i] = uniq_labels[np.argmax(votes)]
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

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        r"""Predict class labels for new data points.

        Computes pairwise distances between query points and training
        data via :func:`scipy.spatial.distance.cdist`, selects the
        :math:`k` nearest neighbors using ``argpartition``, then
        aggregates their votes via :meth:`__predict_label`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Query points. 1-D input is automatically reshaped to
            ``(1, n_features)``.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.

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
        dist = cdist(X, self.X_, self.metric)
        neigh_ind = np.argpartition(dist, self.k - 1, axis=1)[:, : self.k]
        neigh_dist = np.take_along_axis(dist, neigh_ind, axis=1)
        return self.__predict_label(neigh_ind, neigh_dist)
