"""Radius Neighbors classification."""

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode

from ..core.base import Classifier
from ..core.dtypes import ClassificationTarget, FeatureMatrix
from ..core.exceptions import InvalidParameterError, NoNeighborsError


class RadiusNeighborsClassifier(Classifier):
    r"""Radius Neighbors classification.

    Predicts a class label by finding all training samples within a
    fixed radius of each query point, and taking a (optionally
    distance-weighted) majority vote among their labels:

    .. math::
        \hat{y} = \arg\max_{c} \sum_{i \in N_r(x)} w_i \cdot \mathbb{1}(y_i = c)

    where :math:`N_r(x)` is the set of training samples within distance
    r (the radius) of the query point x, and w_i = 1 for uniform
    weighting or w_i = 1 / (d_i + \epsilon) for distance weighting.
    Unlike KNNClassifier, the number of neighbors considered varies per
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
        How neighbors are weighted when voting. "uniform" gives every
        neighbor one equal vote; "distance" weights each neighbor's vote
        by the inverse of its distance to the query point, so closer
        neighbors have more influence.
    outlier_label : int or None, optional
        Label to assign to a query point with no training samples within
        the radius. If None, such a query raises NoNeighborsError
        instead.

    Attributes
    ----------
    X_ : FeatureMatrix
        Training feature matrix, stored as-is for use at prediction time.
    y_ : ClassificationTarget
        Training class labels, stored as-is for use at prediction time.


    .. note::
        Because the number of neighbors per query point is not fixed,
        prediction cannot be fully vectorized the way KNNClassifier's
        can — each query point is processed individually. This makes
        RadiusNeighborsClassifier slower than KNNClassifier for large
        numbers of queries.

    .. note::
        Choosing an appropriate radius depends heavily on the scale and
        density of the feature space. Too small a radius produces many
        outliers (no neighbors found); too large a radius includes
        distant, less relevant samples in every vote.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.neighbors import RadiusNeighborsClassifier

        rng = np.random.default_rng(42)
        n = 100
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        model = RadiusNeighborsClassifier(radius=2.0, outlier_label=-1).fit(X, y)

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        preds = model.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, preds, levels=2, cmap="RdBu", alpha=0.5)
        ax.scatter(X0[:, 0], X0[:, 1], label="Class 0", edgecolor="k")
        ax.scatter(X1[:, 0], X1[:, 1], label="Class 1", edgecolor="k")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("RadiusNeighborsClassifier decision boundary (radius=2.0)")
        ax.legend(loc="best")
    """

    def __init__(
        self,
        radius: float = 1.0,
        metric: Literal["euclidean", "chebyshev", "cityblock"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        outlier_label: int | None = None,
    ) -> None:
        """Initialize this classifier with the given radius-search hyperparameters.

        Parameters
        ----------
        radius : float, optional
            Radius within which training samples are considered neighbors.
            Defaults to 1.0.
        metric : {"euclidean", "chebyshev", "cityblock"}, optional
            Distance metric used to find neighbors. Defaults to "euclidean".
        weights : {"uniform", "distance"}, optional
            How neighbors are weighted when voting. Defaults to "uniform".
        outlier_label : int or None, optional
            Label to assign when no neighbors are found within the radius.
            Defaults to None.
        """
        super().__init__()
        self.radius: float = radius
        self.metric: Literal["euclidean", "chebyshev", "cityblock"] = metric
        self.weights: Literal["uniform", "distance"] = weights
        self.outlier_label: int | None = outlier_label

    def _fit(self, X: FeatureMatrix, y: ClassificationTarget, /) -> None:
        """Validate radius and store the training data.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : ClassificationTarget
            Class labels of shape (n_samples,).

        Raises
        ------
        InvalidParameterError
            If radius is not positive.
        """
        if self.radius <= 0:
            raise InvalidParameterError(f"radius must be a positive float, got {self.radius}")
        self.X_ = X.copy()
        self.y_ = y.copy()

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        """Predict labels by a majority vote among training labels within the radius.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget
            Predicted labels of shape (n_samples,), computed as the
            (optionally distance-weighted) majority vote among each query
            point's neighbors within the radius.

        Raises
        ------
        NoNeighborsError
            If a query point has no training samples within the radius and
            outlier_label is None.
        """
        dists = cdist(X, self.X_, metric=self.metric)
        y_pred = np.empty(X.shape[0], dtype=self.y_.dtype)
        for i in range(X.shape[0]):
            mask = dists[i] <= self.radius
            if not np.any(mask):
                if self.outlier_label is None:
                    raise NoNeighborsError(
                        "No neighbors found within the radius and outlier_label is None."
                    )
                y_pred[i] = self.outlier_label
                continue
            neighbor_labels = self.y_[mask]
            if self.weights == "uniform":
                y_pred[i] = mode(neighbor_labels, keepdims=False).mode
            else:
                neighbor_dists = dists[i][mask]
                weights = 1 / (neighbor_dists + 1e-10)
                uniq_labels, inverse = np.unique(neighbor_labels, return_inverse=True)
                votes = np.bincount(inverse, weights=weights)
                y_pred[i] = uniq_labels[np.argmax(votes)]
        return y_pred
