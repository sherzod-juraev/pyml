"""K-Nearest Neighbors regression."""

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

from ..core.base import Regressor
from ..core.dtypes import FeatureMatrix, RegressionTarget
from ..core.exceptions import InvalidParameterError


class KNNRegressor(Regressor):
    r"""K-Nearest Neighbors regression.

    Predicts a continuous target by finding the n_neighbors closest
    training samples (by the given distance metric) to each query point,
    and averaging their target values:

    .. math::
        \hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i

    where :math:`N_k(x)` is the set of the k training samples closest to
    the query point x. Unlike the gradient-descent-based models, KNN has
    no training phase beyond storing the data — all computation happens
    at prediction time.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neighbors to use. Must not exceed the number
        of training samples.
    metric : {"euclidean", "chebyshev", "cityblock"}, optional
        Distance metric. See :doc:`/api/pyml/core/distance_metrics` for
        formulas. Passed directly to :func:`scipy.spatial.distance.cdist`.
    weights : {"uniform", "distance"}, optional
        How neighbors are weighted when averaging their targets.
        "uniform" gives every neighbor equal weight; "distance" weights
        each neighbor by the inverse of its distance to the query point,
        so closer neighbors contribute more.

    Attributes
    ----------
    X_ : FeatureMatrix
        Training feature matrix, stored as-is for use at prediction time.
    y_ : RegressionTarget
        Training target values, stored as-is for use at prediction time.


    .. note::
        With weights="distance", a query point that exactly coincides
        with a training sample is handled by adding a small constant to
        every distance before inverting it, so a zero distance produces
        a very large but finite weight rather than a division-by-zero
        error.

    .. note::
        Because prediction requires computing distances to every stored
        training sample, KNN scales poorly to large datasets compared to
        parametric models like linear regression, whose prediction cost
        does not grow with the size of the training set.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.neighbors import KNNRegressor

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
        y = np.sin(X.ravel()) + rng.normal(0, 0.1, size=30)

        model = KNNRegressor(n_neighbors=5).fit(X, y)
        X_line = np.linspace(0, 10, 200).reshape(-1, 1)
        y_line = model.predict(X_line)

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.6, label="Training data")
        ax.plot(X_line, y_line, color="red", label="KNN prediction")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("KNNRegressor fit (n_neighbors=5)")
        ax.legend(loc="best")
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "chebyshev", "cityblock"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        """Initialize this regressor with the given neighbor-search hyperparameters.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of nearest neighbors to use. Defaults to 5.
        metric : {"euclidean", "chebyshev", "cityblock"}, optional
            Distance metric used to find neighbors. Defaults to "euclidean".
        weights : {"uniform", "distance"}, optional
            How neighbors are weighted when averaging their targets.
            Defaults to "uniform".
        """
        super().__init__()
        self.n_neighbors: int = n_neighbors
        self.metric: Literal["euclidean", "chebyshev", "cityblock"] = metric
        self.weights: Literal["uniform", "distance"] = weights

    def _fit(self, X: FeatureMatrix, y: RegressionTarget, /) -> None:
        """Validate n_neighbors and store the training data.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : RegressionTarget
            Continuous target values of shape (n_samples,).

        Raises
        ------
        InvalidParameterError
            If n_neighbors is not positive, or exceeds the number of
            training samples.
        """
        if self.n_neighbors <= 0:
            raise InvalidParameterError(
                f"n_neighbors must be a positive integer, got {self.n_neighbors}."
            )
        if self.n_neighbors > X.shape[0]:
            raise InvalidParameterError(
                f"n_neighbors={self.n_neighbors} cannot exceed the number of "
                f"training samples ({X.shape[0]})."
            )
        self.X_ = X.copy()
        self.y_ = y.copy()

    def _predict(self, X: FeatureMatrix, /) -> RegressionTarget:
        """Predict targets by averaging the n_neighbors nearest training targets.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        RegressionTarget
            Predicted continuous values of shape (n_samples,), computed as
            the (optionally distance-weighted) average of each query point's
            nearest neighbors' targets.
        """
        dists = cdist(X, self.X_, metric=self.metric)
        neighbor_ind = np.argpartition(dists, kth=self.n_neighbors - 1, axis=1)[
            :, : self.n_neighbors
        ]
        if self.weights == "uniform":
            return np.mean(self.y_[neighbor_ind], axis=1)
        neighbor_dist = np.take_along_axis(dists, neighbor_ind, axis=1)
        weights = 1 / (neighbor_dist + 1e-12)
        y_pred = np.average(self.y_[neighbor_ind], axis=1, weights=weights)
        return y_pred
