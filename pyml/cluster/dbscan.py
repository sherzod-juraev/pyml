"""DBSCAN density-based clustering."""

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

from ..core.base import Clusterer
from ..core.dtypes import ClusterLabels, FeatureMatrix
from ..core.exceptions import InvalidParameterError


class DBSCAN(Clusterer):
    r"""Density-based clustering that groups closely packed points.

    Unlike KMeans, DBSCAN does not require specifying the number of
    clusters in advance, and can identify arbitrarily shaped clusters as
    well as noise points that don't belong to any cluster.

    For a point :math:`p`, its :math:`\varepsilon`-neighborhood is the
    set of points within distance ``eps``:

    .. math::
        N_\varepsilon(p) = \{q \in D : \text{dist}(p, q) \leq \varepsilon\}

    A point :math:`p` is a *core point* if its neighborhood contains at
    least ``min_samples`` points, including itself:

    .. math::
        p \text{ is a core point} \iff |N_\varepsilon(p)| \geq \text{min\_samples}

    A cluster is formed by taking a core point, finding all points
    *density-reachable* from it through a chain of overlapping
    core-point neighborhoods:

    .. math::
        q \text{ is reachable from } p \iff \exists\, p_1, \ldots, p_n,\
            p_1 = p,\ p_n = q,\ p_{i+1} \in N_\varepsilon(p_i)\
            \text{ and } p_i \text{ is a core point}

    and including any non-core points within :math:`\varepsilon` of those
    core points (border points). Points reachable from no core point are
    labeled as noise (``-1``).

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two points for one to be considered
        in the neighborhood of the other. Must be positive. Defaults to
        0.5.
    min_samples : int, optional
        The number of points (including the point itself) required
        within a distance of eps for a point to be considered a core
        point. Must not exceed the number of training samples. Defaults
        to 5.
    metric : {"euclidean", "cityblock", "chebyshev"}, optional
        Distance metric used to find neighbors, passed directly to
        :func:`scipy.spatial.distance.cdist`. Defaults to "euclidean".

    Attributes
    ----------
    labels_ : ClusterLabels
        Cluster assignment for each training sample, of shape
        (n_samples,). A value of -1 indicates noise.


    .. note::
        DBSCAN has no predict method: because cluster membership depends
        on the density structure of the entire training set, there is no
        well-defined way to assign a brand-new point to a cluster
        without recomputing that structure. Use fit_predict to obtain
        labels for the training data itself.

    .. note::
        Choosing eps and min_samples depends heavily on the scale and
        density of your data. Too small an eps produces mostly noise;
        too large an eps merges distinct clusters together.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.cluster import DBSCAN

        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=(-5, -5), scale=1.0, size=(50, 2))
        X1 = rng.normal(loc=(5, 5), scale=1.0, size=(50, 2))
        noise = rng.uniform(-15, 15, size=(10, 2))
        X = np.vstack([X0, X1, noise])

        model = DBSCAN(eps=1.5, min_samples=5)
        labels = model.fit_predict(X)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("DBSCAN clustering (noise shown in a distinct color)")
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: Literal["euclidean", "cityblock", "chebyshev"] = "euclidean",
    ) -> None:
        """Initialize this clusterer with the given hyperparameters.

        Parameters
        ----------
        eps : float, optional
            The maximum distance between two points for one to be
            considered in the neighborhood of the other. Defaults to 0.5.
        min_samples : int, optional
            The number of points required within a distance of eps for a
            point to be considered a core point. Defaults to 5.
        metric : {"euclidean", "cityblock", "chebyshev"}, optional
            Distance metric used to find neighbors. Defaults to "euclidean".
        """
        super().__init__()
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.metric: Literal["euclidean", "cityblock", "chebyshev"] = metric

    def _fit(self, X: FeatureMatrix, /) -> None:
        """Assign cluster labels using the DBSCAN density-reachability algorithm.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).

        Raises
        ------
        InvalidParameterError
            If eps is not positive, min_samples is not positive, or
            min_samples exceeds the number of training samples.
        """
        if self.eps <= 0:
            raise InvalidParameterError(f"eps must be a positive float, got {self.eps}")
        if self.min_samples <= 0:
            raise InvalidParameterError(
                f"min_samples must be a positive integer, got {self.min_samples}"
            )
        if self.min_samples > X.shape[0]:
            raise InvalidParameterError(
                f"min_samples={self.min_samples} cannot exceed the number of "
                f"training samples ({X.shape[0]})."
            )
        self.labels_ = np.full(X.shape[0], -2, dtype=np.int32)
        dists = cdist(X, X, self.metric)
        cluster_idx = 0
        for i in range(X.shape[0]):
            if self.labels_[i] >= 0:
                continue
            neigh_idx = np.where(dists[i] <= self.eps)[0]
            if neigh_idx.shape[0] >= self.min_samples:
                j = 0
                while j < neigh_idx.shape[0]:
                    core_idx = neigh_idx[j]
                    new_neigh_idx = np.where(dists[core_idx] <= self.eps)[0]
                    if new_neigh_idx.shape[0] >= self.min_samples:
                        mask = np.isin(new_neigh_idx, neigh_idx, invert=True)
                        if np.any(mask):
                            neigh_idx = np.concatenate([neigh_idx, new_neigh_idx[mask]])
                    j += 1
                mask = self.labels_[neigh_idx] < 0
                self.labels_[neigh_idx[mask]] = cluster_idx
                cluster_idx += 1
            elif self.labels_[i] == -2:
                self.labels_[i] = -1

    def _get_labels(self) -> ClusterLabels:
        """Return the cluster labels computed during fit.

        Returns
        -------
        ClusterLabels
            Cluster index assigned to each sample seen during fit. A value
            of -1 indicates noise.
        """
        return self.labels_
