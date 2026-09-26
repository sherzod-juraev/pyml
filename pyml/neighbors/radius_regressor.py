"""Radius Neighbors regression."""

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

from ..core.base import Regressor
from ..core.dtypes import FeatureMatrix, RegressionTarget
from ..core.exceptions import InvalidParameterError, NoNeighborsError


class RadiusNeighborsRegressor(Regressor):
    r"""Radius Neighbors regression.

    Predicts a continuous target by finding all training samples within
    a fixed radius of each query point, and averaging their target
    values:

    .. math::
        \hat{y} = \frac{\sum_{i \in N_r(x)} w_i \cdot y_i}{\sum_{i \in N_r(x)} w_i}

    where :math:`N_r(x)` is the set of training samples within distance
    r (the radius) of the query point x, and w_i = 1 for uniform
    weighting or w_i = 1 / (d_i + \epsilon) for distance weighting.
    Unlike KNNRegressor, the number of neighbors considered varies per
    query point depending on how many training samples fall within the
    radius — it is not fixed.

    Parameters
    ----------
    radius : float, optional
        Radius within which training samples are considered neighbors.
        Must be positive.
    metric : {"euclidean", "chebyshev", "cityblock"}, optional
        Distance metric. See :doc:`/api/pyml/core/distance_metrics` for
        formulas. Passed directly to :func:`scipy.spatial.distance.cdist`.
    weights : {"uniform", "distance"}, optional
        How neighbors are weighted when averaging their targets.
        "uniform" gives every neighbor equal weight; "distance" weights
        each neighbor by the inverse of its distance to the query point,
        so closer neighbors contribute more.
    outlier_label : float or None, optional
        Value to assign to a query point with no training samples within
        the radius. If None, such a query raises NoNeighborsError
        instead.

    Attributes
    ----------
    X_ : FeatureMatrix
        Training feature matrix, stored as-is for use at prediction time.
    y_ : RegressionTarget
        Training target values, stored as-is for use at prediction time.


    .. note::
        Because the number of neighbors per query point is not fixed,
        prediction cannot be fully vectorized the way KNNRegressor's can
        — each query point is processed individually. This makes
        RadiusNeighborsRegressor slower than KNNRegressor for large
        numbers of queries.

    .. note::
        Choosing an appropriate radius depends heavily on the scale and
        density of the feature space. Too small a radius produces many
        outliers (no neighbors found); too large a radius includes
        distant, less relevant samples in every prediction.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.neighbors import RadiusNeighborsRegressor

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
        y = np.sin(X.ravel()) + rng.normal(0, 0.1, size=30)

        model = RadiusNeighborsRegressor(radius=1.5).fit(X, y)
        X_line = np.linspace(0, 10, 200).reshape(-1, 1)
        y_line = model.predict(X_line)

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.6, label="Training data")
        ax.plot(X_line, y_line, color="red", label="Radius Neighbors prediction")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("RadiusNeighborsRegressor fit (radius=1.5)")
        ax.legend(loc="best")
    """

    def __init__(
        self,
        radius: float = 1.0,
        metric: Literal["euclidean", "chebyshev", "cityblock"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        outlier_label: float | None = None,
    ) -> None:
        """Initialize this regressor with the given radius-search hyperparameters.

        Parameters
        ----------
        radius : float, optional
            Radius within which training samples are considered neighbors.
            Defaults to 1.0.
        metric : {"euclidean", "chebyshev", "cityblock"}, optional
            Distance metric used to find neighbors. Defaults to "euclidean".
        weights : {"uniform", "distance"}, optional
            How neighbors are weighted when averaging their targets.
            Defaults to "uniform".
        outlier_label : float or None, optional
            Value to assign when no neighbors are found within the radius.
            Defaults to None.
        """
        super().__init__()
        self.radius: float = radius
        self.metric: Literal["euclidean", "chebyshev", "cityblock"] = metric
        self.weights: Literal["uniform", "distance"] = weights
        self.outlier_label: float | None = outlier_label

    def _fit(self, X: FeatureMatrix, y: RegressionTarget, /) -> None:
        """Validate radius and store the training data.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : RegressionTarget
            Continuous target values of shape (n_samples,).

        Raises
        ------
        InvalidParameterError
            If radius is not positive.
        """
        if self.radius <= 0:
            raise InvalidParameterError(f"radius must be a positive float, got {self.radius}")
        self.X_ = X.copy()
        self.y_ = y.copy()

    def _predict(self, X: FeatureMatrix, /) -> RegressionTarget:
        """Predict targets by averaging training targets within the radius.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        RegressionTarget
            Predicted continuous values of shape (n_samples,), computed as
            the (optionally distance-weighted) average of each query point's
            targets within the radius.

        Raises
        ------
        NoNeighborsError
            If a query point has no training samples within the radius and
            outlier_label is None.
        """
        dists = cdist(X, self.X_, metric=self.metric)
        y_pred = np.empty(shape=X.shape[0], dtype=self.y_.dtype)
        for i in range(X.shape[0]):
            mask = dists[i] <= self.radius
            if not np.any(mask):
                if self.outlier_label is None:
                    raise NoNeighborsError(
                        "No neighbors found within the radius and outlier_label is None."
                    )
                y_pred[i] = self.outlier_label
                continue
            neighbor_targets = self.y_[mask]
            if self.weights == "uniform":
                y_pred[i] = np.mean(neighbor_targets)
            else:
                neighbor_dists = dists[i][mask]
                weights = 1 / (neighbor_dists + 1e-10)
                y_pred[i] = np.average(neighbor_targets, weights=weights)
        return y_pred
