r"""K-Means clustering using Lloyd's Algorithm.

This module provides a pure NumPy implementation of the K-Means
clustering algorithm with support for multiple distance metrics
and initialization strategies.

Classes
-------
Kmeans
    K-Means clustering with Lloyd's iterative optimization.

Notes
-----
K-Means minimizes the Within-Cluster Sum of Squares (WCSS) via
alternating assignment and update steps. The algorithm converges
to a local minimum — result quality depends on initialization.
K-Means++ initialization is strongly recommended over uniform
random selection.
"""

import warnings
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from ..exc import NotFittedError
from ._kmeans_pp import KmeansPP


class Kmeans:
    r"""K-Means Clustering using Lloyd's Algorithm.

    Partitions :math:`n` samples into :math:`k` disjoint clusters by
    minimizing the **Within-Cluster Sum of Squares (WCSS)**:

    .. math::

        J = \sum_{j=1}^{k} \sum_{x_i \in C_j}
        \| x_i - \mu_j \|^2

    where :math:`\mu_j` is the centroid of cluster :math:`C_j`.

    Each iteration consists of two steps:

    1. **Assignment** — assign each sample to the nearest centroid:

       .. math::

           C_j = \{ x_i : \arg\min_{l} \, d(x_i, \mu_l) = j \}

    2. **Update** — recompute each centroid as the mean of its members:

       .. math::

           \mu_j := \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i

    Parameters
    ----------
    k : int, default=3
        Number of clusters to form.
    max_iter : int, default=100
        Maximum number of Lloyd iterations.
    tol : float, default=1e-3
        Convergence tolerance. Training stops when the maximum centroid
        displacement (measured by ``metric``) falls below this value:

        .. math::

            \max_j \, d(\mu_j^{new}, \mu_j^{old}) < tol

    init : {'kmeans++', 'uniform'}, default='kmeans++'
        Centroid initialization strategy:

        - ``'kmeans++'`` — distance-weighted probabilistic initialization,
          reduces sensitivity to poor starting points.
        - ``'uniform'`` — :math:`k` centroids chosen uniformly at random
          from the training data.

    metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used for both assignment and convergence check:

        - ``'euclidean'`` — L2 norm:

          .. math::

              d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}

        - ``'cityblock'`` — L1 norm (Manhattan):

          .. math::

              d(x, x') = \sum_{j=1}^{p} |x_j - x'_j|

        - ``'chebyshev'`` — L∞ norm:

          .. math::

              d(x, x') = \max_j |x_j - x'_j|

    random_state : int or None, default=None
        Seed for the random number generator. Set to an integer for
        reproducible centroid initialization.

    Attributes
    ----------
    centroids_ : np.ndarray of shape (k, n_features)
        Learned cluster centers after fitting. Each row is the mean
        of all points assigned to that cluster.

    Notes
    -----
    **Local minima and initialization**

    K-Means is **not guaranteed** to find the global optimum of
    :math:`J`. The result depends heavily on initialization.
    ``'kmeans++'`` initialization significantly improves both
    convergence speed and solution quality by spreading initial
    centroids across the data space.

    **Empty cluster handling**

    If a cluster receives no points during an iteration (possible
    with poor initialization or high :math:`k`), its centroid is
    reinitialized to the training point furthest from any existing
    centroid, and a ``RuntimeWarning`` is emitted.

    **Convergence criterion**

    Training stops when the maximum displacement of any centroid
    falls below ``tol``, measured using the chosen ``metric``.
    This is checked via :meth:`convergence`.

    **Choosing :math:`k`**

    The number of clusters :math:`k` must be specified in advance.
    In practice, methods like the elbow method or silhouette
    analysis are used to select :math:`k`. This implementation
    does not provide automatic :math:`k` selection.

    Examples
    --------
    >>> from pyml import Kmeans
    >>> import numpy as np
    >>>
    >>> X = np.array([[1., 2.], [1., 4.], [1., 0.],
    ...               [10., 2.], [10., 4.], [10., 0.]])
    >>>
    >>> model = Kmeans(k=3, init='kmeans++', random_state=42)
    >>> model.fit(X)
    >>> model.centroids_
    array([[10.,  2.],
           [ 1.,  2.],
           [10.,  0.]])
    >>> model.predict(np.array([[5., 3.]]))
    array([0])
    """

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 100,
        tol: float | int = 1e-3,
        init: Literal["uniform", "kmeans++"] = "kmeans++",
        metric: Literal["euclidean", "chebyshev", "cityblock"] = "euclidean",
        random_state: int | None = None,
    ) -> None:
        r"""Initialize the K-Means clustering model.

        Parameters
        ----------
        k : int, default=3
            Number of clusters to partition the data into.
        max_iter : int, default=100
            Maximum number of Lloyd iterations to perform.
        tol : float, default=1e-3
            Convergence threshold for maximum centroid displacement.
        init : {'kmeans++', 'uniform'}, default='kmeans++'
            Strategy for initializing centroids before training.
        metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
            Distance metric for point-to-centroid computation.
        random_state : int or None, default=None
            Seed for random number generation. Controls centroid
            initialization reproducibility.

        Returns
        -------
        None
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.metric = metric
        self.random_state = random_state
        self.__fitted = False

    def initialize_centroids_(self, X: npt.NDArray[np.float64]) -> None:
        r"""Initialize cluster centroids using the chosen strategy.

        For ``'uniform'`` — selects :math:`k` random samples without
        replacement from the training data using ``numpy.random.default_rng``.

        For ``'kmeans++'`` — delegates to :class:`KmeansPP` for
        distance-weighted probabilistic initialization that spreads
        centroids across the data space.

        The initialized centroids are stored in ``self.centroids_``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data from which initial centroids are sampled.
            Each selected centroid is a row from this matrix.

        Returns
        -------
        None
        """
        if self.init == "uniform":
            rng = np.random.default_rng(self.random_state)
            ind = rng.choice(X.shape[0], size=self.k, replace=False)
            self.centroids_ = X[ind, :].copy()
        else:
            kmeanspp = KmeansPP(
                k=self.k, metric=self.metric, random_state=self.random_state
            )
            kmeanspp.initialize(X)
            self.centroids_ = kmeanspp.centroids_

    def fit(self, X: npt.NDArray[np.float64]) -> Self:
        r"""Fit the K-Means model to training data.

        Initializes centroids via :meth:`initialize_centroids_`, then
        alternates between two steps until convergence or ``max_iter``:

        1. **Assignment** — assign each sample to the nearest centroid
           via :meth:`cal_labels`.
        2. **Update** — recompute centroids as the mean of assigned
           points via :meth:`update_centroids_`.

        Convergence is checked after each update via :meth:`convergence`.
        Training stops when the maximum centroid displacement falls below
        ``tol``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster. Each row is one sample.

        Returns
        -------
        self : Kmeans
            The fitted model with ``centroids_`` attribute populated.
            Enables method chaining.

        Notes
        -----
        If an empty cluster is encountered during training, its centroid
        is reinitialized and a ``RuntimeWarning`` is raised. This does
        not interrupt training.
        """
        self.initialize_centroids_(X)
        for _ in range(self.max_iter):
            labels = self.cal_labels(X)
            C_old = self.centroids_.copy()
            self.update_centroids_(X, labels)
            if self.convergence(C_old):
                break
        self.__fitted = True
        return self

    def convergence(self, C_old: npt.NDArray[np.float64]) -> bool:
        r"""Check whether centroids have converged.

        Computes the displacement of each centroid from its position
        in the previous iteration using the chosen ``metric``. Returns
        ``True`` if all displacements are strictly below ``tol``:

        .. math::

            \forall j \in \{1, \ldots, k\}: \,
            d(\mu_j^{new}, \mu_j^{old}) < tol

        The distance computation depends on ``self.metric``:

        - ``'euclidean'`` — L2 norm: :math:`\| \Delta \mu \|_2`
        - ``'cityblock'`` — L1 norm: :math:`\| \Delta \mu \|_1`
        - ``'chebyshev'`` — L∞ norm: :math:`\max |\Delta \mu|`

        Parameters
        ----------
        C_old : np.ndarray of shape (k, n_features)
            Centroid positions from the previous iteration.

        Returns
        -------
        converged : bool
            ``True`` if all centroids moved less than ``tol``,
            ``False`` otherwise.
        """
        dif = self.centroids_ - C_old
        if self.metric == "euclidean":
            dist = np.linalg.norm(dif, ord=2, axis=1)
        elif self.metric == "chebyshev":
            dist = np.max(np.abs(dif), axis=1)
        else:
            dist = np.linalg.norm(dif, ord=1, axis=1)
        return bool(np.all(dist < self.tol))

    def cal_labels(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        r"""Assign each sample to the nearest centroid.

        Computes pairwise distances between all samples and all
        centroids using :func:`scipy.spatial.distance.cdist` with
        the configured ``metric``, then assigns each sample to the
        closest centroid index:

        .. math::

            \text{label}_i = \arg\min_{j} \, d(x_i, \mu_j)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to assign to clusters.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster index in :math:`\{0, 1, \ldots, k-1\}` for
            each sample.
        """
        centroids_ = (
            self.centroids_
            if self.centroids_.ndim == 2
            else np.array([self.centroids_])
        )
        distances = cdist(X, centroids_, metric=self.metric)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids_(
        self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.intp]
    ) -> None:
        r"""Update each centroid as the arithmetic mean of its assigned points.

        For each cluster :math:`j`, computes:

        .. math::

            \mu_j := \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i

        where :math:`C_j` is the set of samples assigned to cluster
        :math:`j` in the current iteration.

        **Empty cluster handling:**

        If a cluster :math:`j` receives no assigned points, its centroid
        is reinitialized to the training sample with the maximum minimum
        distance to any existing centroid:

        .. math::

            x^* = \arg\max_{x_i} \, \min_{l} \, d(x_i, \mu_l)

        A ``RuntimeWarning`` is emitted to notify the user.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full training data. Used for reinitializing empty clusters.
        labels : np.ndarray of shape (n_samples,)
            Current cluster assignment for each sample. Values must be
            in :math:`\{0, 1, \ldots, k-1\}`.

        Returns
        -------
        None
        """
        for i in range(self.k):
            neighbors = X[labels == i]
            if neighbors.shape[0] != 0:
                self.centroids_[i] = neighbors.mean(axis=0)
            else:
                dist = cdist(self.centroids_, X, metric=self.metric)
                self.centroids_[i] = X[np.argmax(np.min(dist, axis=0))].copy()
                warnings.warn(
                    f"Cluster {i} is empty. Reinitializing centroid.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        r"""Predict cluster labels for new data.

        Assigns each sample in ``X`` to the nearest centroid learned
        during fitting by calling :meth:`cal_labels`. The centroids
        must have been learned via :meth:`fit` before calling this
        method.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to assign to clusters. Must have the same
            number of features as the training data.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted cluster indices in :math:`\{0, 1, \ldots, k-1\}`.

        Raises
        ------
        NotFittedError
            If ``predict`` is called before ``fit``. The model must
            be trained before making predictions.
        """
        if not self.__fitted:
            raise NotFittedError(self)
        labels = self.cal_labels(X)
        return labels
