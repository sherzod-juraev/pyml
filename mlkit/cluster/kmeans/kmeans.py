import numpy as np
from typing import Literal, Self
from scipy.spatial.distance import cdist
from .kmeans_pp import KmeansPP
from ...exc import NotFitted
import warnings


class Kmeans:
    """
    K-Means Clustering using Lloyd's Algorithm.

    Partitions :math:`n` samples into :math:`k` disjoint clusters by
    minimizing the **Within-Cluster Sum of Squares (WCSS)**:

    .. math::

        J = \\sum_{j=1}^{k} \\sum_{x_i \\in C_j}
        \\| x_i - \\mu_j \\|^2

    where :math:`\\mu_j` is the centroid of cluster :math:`C_j`.

    Each iteration consists of two steps:

    1. **Assignment** — assign each sample to the nearest centroid:

       .. math::

           C_j = \\{ x_i : \\arg\\min_{l} \\, d(x_i, \\mu_l) = j \\}

    2. **Update** — recompute each centroid as the mean of its members:

       .. math::

           \\mu_j := \\frac{1}{|C_j|} \\sum_{x_i \\in C_j} x_i

    Parameters
    ----------
    k : int, default=3
        Number of clusters.
    max_iter : int, default=100
        Maximum number of Lloyd iterations.
    tol : float, default=1e-3
        Convergence tolerance. Training stops when the maximum centroid
        displacement (measured by ``metric``) falls below this value:

        :math:`\\max_j \\, d(\\mu_j^{new}, \\mu_j^{old}) < tol`

    init : {'kmeans++', 'uniform'}, default='kmeans++'
        Centroid initialization strategy.

        - ``'kmeans++'`` — distance-weighted probabilistic initialization,
          reduces sensitivity to poor starting points.
        - ``'uniform'`` — :math:`k` centroids chosen uniformly at random
          from the training data.

     metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used for both assignment and convergence check.

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \\sqrt{\\sum_{j=1}^{p} (x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan):

          .. math::

              d(x, x') = \\sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

          .. math::

              d(x, x') = \\max_j |x_j - x'_j|


    random_state : int or None, default=None
        Seed for the random number generator. Set for reproducibility.

    Attributes
    ----------
    centroids\\_ : np.ndarray of shape (k, n_features)
        Cluster centers after fitting.

    Notes
    -----
    K-Means is **not guaranteed** to find the global optimum of :math:`J`.
    The result depends on initialization. ``'kmeans++'`` initialization
    significantly improves convergence speed and solution quality.

    Empty clusters are handled by reinitializing the centroid to the
    training point furthest from any existing centroid, and a
    ``RuntimeWarning`` is emitted.

    Examples
    --------
    >>> model = Kmeans(k=3, init='kmeans++', random_state=42)
    >>> model.fit(X_train)
    >>> labels = model.predict(X_test)
    """

    def __init__(
            self,
            k: int = 3,
            max_iter: int = 100,
            tol: float | int = 1e-3,
            init: Literal['uniform', 'kmeans++'] = 'kmeans++',
            metric: Literal['euclidean', 'chebyshev', 'cityblock'] = 'euclidean',
            random_state: int | None = None
    ) -> None:

        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids_ = None
        self.metric = metric
        self.random_state = random_state
        self.__fitted = False

    def initialize_centroids_(self, X: np.ndarray) -> None:
        """
        Initialize cluster centroids using the chosen strategy.

        For ``'uniform'`` — selects :math:`k` random samples without
        replacement. For ``'kmeans++'`` — delegates to
        :class:`KmeansPP` for distance-weighted initialization.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data from which centroids are chosen.
        """
        if self.init == 'uniform':
            rng = np.random.default_rng(self.random_state)
            ind = rng.choice(X.shape[0], size=self.k, replace=False)
            self.centroids_ = X[ind, :].copy()
        else:
            kmeanspp = KmeansPP(k=self.k, metric=self.metric, random_state=self.random_state)
            kmeanspp.initialize(X)
            self.centroids_ = kmeanspp.centroids_

    def fit(self, X: np.ndarray) -> Self:
        """
        Fit the K-Means model to training data.

        Initializes centroids, then alternates between assigning samples
        to the nearest centroid and updating centroids until convergence
        or ``max_iter`` is reached.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : Kmeans
            Fitted model with learned ``centroids_``.
        """

        self.initialize_centroids_(X)
        for i in range(self.max_iter):
            labels = self.cal_labels(X)
            C_old = self.centroids_.copy()
            self.update_centroids_(X, labels)
            if  self.convergence(C_old):
                break
        self.__fitted = True
        return self

    def convergence(self, C_old: np.ndarray) -> bool:
        """
        Check whether centroids have converged.

        Computes displacement of each centroid since the last iteration
        using the chosen ``metric``, and returns ``True`` if all
        displacements are below ``tol``:

        .. math::

            \\forall j: \\, d(\\mu_j^{new}, \\mu_j^{old}) < tol

        Parameters
        ----------
        C_old : np.ndarray of shape (k, n_features)
            Centroid positions from the previous iteration.

        Returns
        -------
        converged : bool
            ``True`` if all centroids moved less than ``tol``.
        """

        dif = self.centroids_ - C_old
        if self.metric == 'euclidean':
            dist = np.linalg.norm(dif, ord=2, axis=1)
        elif self.metric == 'chebyshev':
            dist = np.max(np.abs(dif), axis=1)
        else:
            dist = np.linalg.norm(dif, ord=1, axis=1)
        if np.all(dist < self.tol):
            return True
        return False

    def cal_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.

        Computes pairwise distances between all samples and all
        centroids, then assigns each sample to the closest one:

        .. math::

            \\text{label}_i = \\arg\\min_{j} \\, d(x_i, \\mu_j)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to assign.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster index for each sample.
        """

        centroids_ = self.centroids_ if self.centroids_.ndim == 2 else np.array([self.centroids_])
        distances = cdist(X, centroids_, metric=self.metric)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids_(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Update each centroid as the mean of its assigned points.

        .. math::

            \\mu_j := \\frac{1}{|C_j|} \\sum_{x_i \\in C_j} x_i

        If a cluster is empty, its centroid is reinitialized to the
        training point with the maximum minimum distance to any
        existing centroid, and a ``RuntimeWarning`` is emitted.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        labels : np.ndarray of shape (n_samples,)
            Current cluster assignments.
        """

        for i in range(self.k):
            neighbors = X[labels == i]
            if neighbors.shape[0] != 0:
                self.centroids_[i] = neighbors.mean(axis=0)
            else:
                dist = cdist(self.centroids_, X, metric=self.metric)
                self.centroids_[i] = X[np.argmax(np.min(dist, axis=0))].copy()
                warnings.warn(f"Cluster {i} is empty. Reinitializing centroid.", RuntimeWarning)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Assigns each sample to the nearest centroid learned during
        fitting using :meth:`cal_labels`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to assign to clusters.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted cluster indices in :math:`\\{0, 1, \\ldots, k-1\\}`.

        Raises
        ------
        NotFitted
            If called before fitting the model.
        """

        if not self.__fitted:
            raise NotFitted(self)
        labels = self.cal_labels(X)
        return labels