"""K-Means clustering with k-means++ initialization."""

import warnings

import numpy as np
from scipy.spatial.distance import cdist

from ..core.base import PredictableClusterer
from ..core.dtypes import ClusterLabels, FeatureMatrix
from ..core.exceptions import InvalidParameterError


class KMeans(PredictableClusterer):
    r"""K-Means clustering with k-means++ centroid initialization.

    Partitions samples into n_clusters groups by iteratively assigning
    each sample to its nearest centroid, then updating each centroid to
    the mean of the samples assigned to it. This minimizes the
    within-cluster sum of squares:

    .. math::
        J = \sum_{j=1}^{k} \sum_{x \in C_j} \|x - \mu_j\|^2

    where :math:`C_j` is the set of samples assigned to cluster j, and
    :math:`\mu_j` is that cluster's centroid.

    Centroids are initialized using k-means++: the first centroid is
    chosen uniformly at random, and each subsequent centroid is chosen
    with probability proportional to its squared distance from the
    nearest existing centroid — spreading centroids out and reducing
    the chance of converging to a poor local minimum compared to fully
    random initialization.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form. Must not exceed the number of
        training samples.
    max_iter : int, optional
        Maximum number of assignment/update iterations.
    tol : float, optional
        Minimum total squared centroid movement between consecutive
        iterations required to continue; if the movement falls below
        this, training stops early.
    random_state : int or None, optional
        Seed for the random number generator used in k-means++
        initialization, for reproducible results.

    Attributes
    ----------
    centroids_ : FeatureMatrix
        Learned cluster centroids, of shape (n_clusters, n_features).
    labels_ : ClusterLabels
        Cluster assignment for each training sample, of shape
        (n_samples,).
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        If a cluster becomes empty during training (no samples are
        closest to its centroid), its centroid is reinitialized to the
        training sample farthest from any existing centroid, and a
        RuntimeWarning is raised.

    .. note::
        K-Means always uses squared Euclidean distance internally, since
        the mean-based centroid update is only mathematically consistent
        with that metric. Unlike the neighbors models, there is no
        metric parameter here.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.cluster import KMeans

        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=(-3, -3), scale=1.0, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=1.0, size=(50, 2))
        X2 = rng.normal(loc=(0, 4), scale=1.0, size=(50, 2))
        X = np.vstack([X0, X1, X2])

        model = KMeans(n_clusters=3, random_state=42).fit(X)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", alpha=0.6)
        ax.scatter(
            model.centroids_[:, 0], model.centroids_[:, 1],
            marker="x", s=200, color="red", label="Centroids",
        )
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("KMeans clustering")
        ax.legend(loc="best")
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Initialize this clusterer with the given hyperparameters.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to form.
        max_iter : int, optional
            Maximum number of assignment/update iterations. Defaults to 300.
        tol : float, optional
            Minimum total squared centroid movement between consecutive
            iterations required to continue. Defaults to 1e-4.
        random_state : int or None, optional
            Seed for the random number generator used in initialization.
            Defaults to None.
        """
        super().__init__()
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.random_state: int | None = random_state

    def _initialize_centroids(self, X: FeatureMatrix, /) -> None:
        """Choose initial centroids using the k-means++ strategy.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        """
        rng = np.random.default_rng(self.random_state)
        self.centroids_ = np.zeros(shape=(self.n_clusters, X.shape[1]), dtype=X.dtype)
        first_idx = rng.integers(X.shape[0])
        self.centroids_[0] = X[first_idx].copy()
        for i in range(1, self.n_clusters):
            dists = cdist(self.centroids_[:i], X, "sqeuclidean")
            min_dists = np.min(dists, axis=0)
            prob = min_dists / (np.sum(min_dists) + 1e-10)
            idx = rng.choice(X.shape[0], p=prob)
            self.centroids_[i] = X[idx].copy()

    def _assign_clusters(self, X: FeatureMatrix, /) -> ClusterLabels:
        """Assign each sample to its nearest centroid.

        Parameters
        ----------
        X : FeatureMatrix
            Data to assign, of shape (n_samples, n_features).

        Returns
        -------
        ClusterLabels
            Index of the nearest centroid for each sample, of shape
            (n_samples,).
        """
        dists = cdist(X, self.centroids_, "sqeuclidean")
        labels = np.argmin(dists, axis=1)
        return labels

    def _get_labels(self) -> ClusterLabels:
        """Return the cluster labels computed during fit.

        Returns
        -------
        ClusterLabels
            Cluster index assigned to each sample seen during fit.
        """
        return self.labels_

    def _fit(self, X: FeatureMatrix, /) -> None:
        """Run the k-means assignment/update loop until convergence.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).

        Raises
        ------
        InvalidParameterError
            If n_clusters is not positive, or exceeds the number of
            training samples.
        """
        if self.n_clusters <= 0:
            raise InvalidParameterError(
                f"n_clusters must be a positive integer, got {self.n_clusters}"
            )
        if self.n_clusters > X.shape[0]:
            raise InvalidParameterError(
                f"n_clusters={self.n_clusters} cannot exceed the number of "
                f"training samples ({X.shape[0]})."
            )
        self._initialize_centroids(X)
        for i in range(self.max_iter):
            labels = self._assign_clusters(X)
            mu_old = self.centroids_.copy()
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if cluster_points.shape[0] > 0:
                    self.centroids_[j] = np.mean(cluster_points, axis=0)
                else:
                    dists = cdist(self.centroids_, X, "sqeuclidean")
                    farthest_idx = np.argmax(np.min(dists, axis=0))
                    self.centroids_[j] = X[farthest_idx].copy()
                    warnings.warn(
                        f"Cluster {j} is empty. Reinitializing centroid.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            c_shift = np.sum((self.centroids_ - mu_old) ** 2)
            if c_shift <= self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter
        self.labels_ = self._assign_clusters(X)

    def _predict(self, X: FeatureMatrix, /) -> ClusterLabels:
        """Assign new samples to their nearest learned centroid.

        Parameters
        ----------
        X : FeatureMatrix
            New data of shape (n_samples, n_features).

        Returns
        -------
        ClusterLabels
            Index of the nearest centroid for each sample, of shape
            (n_samples,).
        """
        return self._assign_clusters(X)
