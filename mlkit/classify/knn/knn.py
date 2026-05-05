from scipy.spatial.distance import cdist
from scipy import stats
from typing import Literal, Self
import numpy as np
from mlkit.exc import NotFitted


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    Predicts class labels by majority vote among the :math:`k` nearest
    neighbors in the training set. Given a query point :math:`x`,
    the algorithm:

    1. Computes distances from :math:`x` to all training points.
    2. Selects the :math:`k` closest neighbors.
    3. Assigns the class with the highest total vote weight.

    **Uniform weighting** — each neighbor votes equally:

    .. math::

        \\hat{y} = \\arg\\max_{c} \\sum_{i=1}^{k}
        \\mathbf{1}[y_{(i)} = c]

    **Distance weighting** — closer neighbors vote more strongly:

    .. math::

        \\hat{y} = \\arg\\max_{c} \\sum_{i=1}^{k}
        w_i \\cdot \\mathbf{1}[y_{(i)} = c],
        \\quad w_i = \\frac{1}{d(x, x_{(i)}) + \\varepsilon}

    where :math:`\\varepsilon = 10^{-12}` prevents division by zero
    for exact matches.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to consider for classification.
    metric : {'euclidean', 'cityblock', 'chebyshev', 'cosine'}, default='euclidean'
        Distance metric used to find nearest neighbors.

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \\sqrt{\\sum_{j=1}^{p}(x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan):

          .. math::

              d(x, x') = \\sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

          .. math::

              d(x, x') = \\max_j |x_j - x'_j|

        - ``'cosine'`` — angular similarity:

          .. math::

              d(x, x') = 1 - \\frac{x \\cdot x'}{\\|x\\| \\|x'\\|}

    weighting : {'uniform', 'distance'}, default='uniform'
        Weighting strategy for neighbor votes.

        - ``'uniform'`` — all neighbors contribute equally.
        - ``'distance'`` — closer neighbors have higher influence,
          weighted by inverse distance.

    Attributes
    ----------
    X\\_ : np.ndarray of shape (n_samples, n_features)
        Training feature matrix stored after fitting.
    y\\_ : np.ndarray of shape (n_samples,)
        Training class labels stored after fitting.

    Notes
    -----
    KNN is a non-parametric, lazy learning algorithm — no explicit
    model is trained. All computation happens at predict time.

    ``argpartition`` is used instead of full sorting for efficiency:
    selecting :math:`k` nearest neighbors costs :math:`O(n)` rather
    than :math:`O(n \\log n)`.

    Performance degrades in high-dimensional spaces due to the
    **curse of dimensionality**: distances become increasingly uniform
    as the number of features grows.

    Examples
    --------
    >>> model = KNNClassifier(k=5, metric='euclidean', weighting='distance')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(
            self,
            k: int = 3,
            metric: Literal['euclidean', 'cityblock', 'chebyshev', 'cosine'] = 'euclidean',
            weighting: Literal['uniform', 'distance'] = 'uniform'
    ) -> None:

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
            Training class labels.

        Returns
        -------
        self : KNNClassifier
            Fitted instance with stored training data.
        """

        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        self.__fitted = True
        return self

    def __predict_label(
            self,
            neigh_ind: np.ndarray,
            neigh_dist: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate neighbor votes into predicted class labels.

        For ``'uniform'`` weighting uses majority vote via
        ``scipy.stats.mode``. For ``'distance'`` weighting computes
        weighted votes per unique class:

        .. math::

            \\text{vote}(c) = \\sum_{i: y_{(i)}=c} w_i,
            \\quad w_i = \\frac{1}{d_i + 10^{-12}}

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
        Predict class labels for new data points.

        Computes pairwise distances between query points and training
        data, selects the :math:`k` nearest neighbors using
        ``argpartition``, then aggregates their votes.

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
        NotFitted
            If called before fitting the model.
        """

        if not self.__fitted:
            raise NotFitted(self)
        X = np.asarray(X)
        X = np.array([X]) if X.ndim == 1 else X
        dist = cdist(X, self.X_, self.metric)
        neigh_ind = np.argpartition(dist, self.k - 1, axis=1)[:, :self.k]
        neigh_dist = np.take_along_axis(dist, neigh_ind, axis=1)
        return self.__predict_label(neigh_ind, neigh_dist)