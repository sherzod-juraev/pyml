from collections import deque
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Groups points into clusters based on local density, automatically
    determining the number of clusters and identifying outliers as noise.

    Unlike centroid-based methods such as KMeans, DBSCAN does not require
    specifying the number of clusters in advance and can discover clusters
    of arbitrary shape.

    A point :math:`p` is classified as one of three types:

    **Core point** — has at least ``MinPts`` neighbors within radius ``eps``:

    .. math::

        |N_{\\varepsilon}(p)| \\geq MinPts

    where the :math:`\\varepsilon`-neighborhood of :math:`p` is defined as:

    .. math::

        N_{\\varepsilon}(p) = \\{ q \\in D \\mid dist(p, q) \\leq \\varepsilon \\}
    **Border point** — not a core point itself, but lies within the
    :math:`\\varepsilon`-neighborhood of at least one core point.

    **Noise point** — neither a core point nor a border point. Assigned
    label :math:`-1`.

    Two points :math:`p` and :math:`q` are **directly density-reachable**
    if:

    .. math::

        q \\in N_{\\varepsilon}(p) \\quad \\text{and} \\quad
        |N_{\\varepsilon}(p)| \\geq MinPts

    A point :math:`q` is **density-reachable** from :math:`p` if there
    exists a chain:

    .. math::

        p = p_1, p_2, \\ldots, p_n = q

    such that each :math:`p_{i+1}` is directly density-reachable from
    :math:`p_i`.

    Two points are **density-connected** if there exists a point :math:`o`
    such that both :math:`p` and :math:`q` are density-reachable from
    :math:`o`.

    A cluster is defined as a maximal set of density-connected points.

    The algorithm proceeds as follows:

    1. Compute the full pairwise distance matrix :math:`D \\in \\mathbb{R}^{n \\times n}`:

    .. math::

        D_{ij} = dist(x_i, x_j)

    2. For each unvisited point :math:`p`:

       - Find :math:`N_{\\varepsilon}(p) = \\{ j \\mid D_{pj} \\leq \\varepsilon \\}`
       - If :math:`|N_{\\varepsilon}(p)| < MinPts` → mark as noise (:math:`-1`)
       - Else → create new cluster, expand via BFS queue

    3. BFS expansion: for each neighbor :math:`q` in queue:

       - If :math:`|N_{\\varepsilon}(q)| \\geq MinPts` → add its unvisited
         neighbors to queue and assign current cluster label

    Border points that fall within multiple clusters are assigned to the
    cluster whose core point discovers them first (BFS order).

    The algorithm is **deterministic** — given the same ``X``, ``eps``,
    and ``MinPts``, the output is always identical.

    Time complexity is :math:`O(n^2)` due to full pairwise distance
    computation via ``cdist``

    Parameters
    ----------
    eps : float
        The radius :math:`\\varepsilon` of the neighborhood around each point.
        Points within this distance are considered neighbors. Smaller values
        produce more, tighter clusters; larger values merge clusters.
    MinPts : int, default=5
        Minimum number of points required within ``eps`` radius for a point
        to be considered a core point. Higher values require denser regions
        to form clusters.
    metric : {'euclidean', 'cityblock', 'chebyshev'}, default='euclidean'
        Distance metric used to compute pairwise distances:

        - ``'euclidean'``: :math:`\\sqrt{\\sum_i (p_i - q_i)^2}` — standard
          geometric distance.
        - ``'cityblock'``: :math:`\\sum_i |p_i - q_i|` — Manhattan distance,
          robust to high-dimensional data.
        - ``'chebyshev'``: :math:`\\max_i |p_i - q_i|` — maximum coordinate
          difference.

    Notes
    -----
    Unlike KMeans, DBSCAN has no learnable parameters and therefore has
    no ``predict`` method. New points cannot be assigned to clusters after
    fitting without re-running the full algorithm. Use ``fit_predict``
    to obtain cluster labels in a single pass.

    Noise points are assigned label :math:`-1`. Cluster labels start
    from :math:`0` and increment by :math:`1` for each new cluster found.

    The choice of ``eps`` and ``MinPts`` significantly affects results.
    A common heuristic for ``MinPts`` is :math:`2 \\times n\\_features`.
    For ``eps``, a k-distance plot (sorted distances to the k-th nearest
    neighbor) can help identify a suitable value.

    Examples
    --------
    >>> model = DBSCAN(eps=0.5, MinPts=5, metric='euclidean')
    >>> labels = model.fit_predict(X)
    >>> n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    >>> n_noise = np.sum(labels == -1)
    """

    def __init__(
            self,
            eps: float,
            MinPts: int = 5,
            metric: Literal['euclidean', 'cityblock', 'chebyshev'] = 'euclidean'
    ):

        self.eps = eps
        self.MinPts = MinPts
        self.metric = metric

    def check_params(self) -> None:

        allowed_metrics = ['euclidean', 'cityblock', 'chebyshev']
        if self.metric not in allowed_metrics:
            raise ValueError(
                f"Unsupported metric '{self.metric}'. "
                f"Expected one of {allowed_metrics}."
            )

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        """
        Compute cluster labels for all points in ``X``.

        Runs the full DBSCAN algorithm in a single pass — computes
        pairwise distances, identifies core points, and expands clusters
        via Breadth-First Search (BFS).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data matrix. Each row is one sample.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster label for each point. Noise points are labeled
            :math:`-1`. Cluster labels start from :math:`0`.

        Notes
        -----
        Time complexity is :math:`O(n^2)` due to full pairwise distance
        matrix computation. For large datasets, consider approximate
        nearest neighbor methods.

        Border points that lie within ``eps`` of multiple clusters are
        assigned to whichever cluster's core point discovers them first
        in BFS traversal order.
        """

        self.check_params()
        distances = cdist(X, X, metric=self.metric)
        visited = np.full(X.shape[0], False, dtype=bool)
        labels = np.full(X.shape[0], -1, dtype=int)
        label_id = 0
        n = X.shape[0]
        for i in range(n):
            if visited[i]:
                continue
            neigh_ind = np.where(distances[i] <= self.eps)[0]
            nonvisited_ind = np.where(~visited[neigh_ind])[0]
            if neigh_ind.shape[0] >= self.MinPts:
                neigh_ind = neigh_ind[nonvisited_ind]
                labels[neigh_ind] = label_id
                queue: deque[np.intp] = deque()
                queue.extend(neigh_ind)
                while len(queue) > 0:
                    j = queue.popleft()
                    neigh_ind = np.where(distances[j] <= self.eps)[0]
                    nonvisited_ind = np.where(~visited[neigh_ind])[0]
                    if neigh_ind.shape[0] >= self.MinPts:
                        neigh_ind = neigh_ind[nonvisited_ind]
                        labels[neigh_ind] = label_id
                        queue.extend(neigh_ind)
                    visited[j] = True
                label_id += 1
            visited[i] = True
        return labels
