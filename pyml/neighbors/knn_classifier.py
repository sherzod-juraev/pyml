"""K-Nearest Neighbors classification."""

from typing import Any, Literal, cast

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode

from ..core.base import Classifier
from ..core.dtypes import ClassificationTarget, FeatureMatrix
from ..core.exceptions import InvalidParameterError


class KNNClassifier(Classifier):
    r"""K-Nearest Neighbors classification.

    Predicts a class label by finding the n_neighbors closest training
    samples (by the given distance metric) to each query point, and
    taking a (optionally distance-weighted) majority vote among their
    labels:

    .. math::
        \hat{y} = \arg\max_{c} \sum_{i \in N_k(x)} w_i \cdot \mathbb{1}(y_i = c)

    where :math:`N_k(x)` is the set of the k training samples closest to
    the query point x, and w_i = 1 for uniform weighting or
    w_i = 1 / (d_i + \epsilon) for distance weighting. Unlike the
    gradient-descent-based models, KNN has no training phase beyond
    storing the data — all computation happens at prediction time.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neighbors to use. Must not exceed the number
        of training samples. Defaults to 5.
    metric : {"euclidean", "chebyshev", "cityblock"}, optional
        Distance metric used to find neighbors, passed directly to
        :func:`scipy.spatial.distance.cdist`. Defaults to "euclidean".
    weights : {"uniform", "distance"}, optional
        How neighbors are weighted when voting. "uniform" gives every
        neighbor one equal vote; "distance" weights each neighbor's vote
        by the inverse of its distance to the query point, so closer
        neighbors have more influence. Defaults to "uniform".

    Attributes
    ----------
    X_ : FeatureMatrix
        Training feature matrix, stored as-is for use at prediction time.
    y_ : ClassificationTarget
        Training class labels, stored as-is for use at prediction time.
    classes_ : ClassificationTarget
        Sorted array of the unique class labels seen during fit.


    .. note::
        With weights="distance", a query point that exactly coincides
        with a training sample is handled by adding a small constant to
        every distance before inverting it, so a zero distance produces
        a very large but finite weight rather than a division-by-zero
        error.

    .. note::
        Because prediction requires computing distances to every stored
        training sample, KNN scales poorly to large datasets compared to
        parametric models like logistic regression, whose prediction
        cost does not grow with the size of the training set.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.neighbors import KNNClassifier

        rng = np.random.default_rng(42)
        n = 100
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        model = KNNClassifier(n_neighbors=5).fit(X, y)

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        preds = model.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, preds, levels=1, cmap="RdBu", alpha=0.5)
        ax.scatter(X0[:, 0], X0[:, 1], label="Class 0", edgecolor="k")
        ax.scatter(X1[:, 0], X1[:, 1], label="Class 1", edgecolor="k")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("KNNClassifier decision boundary (n_neighbors=5)")
        ax.legend()
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "chebyshev", "cityblock"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        """Initialize this classifier with the given neighbor-search hyperparameters.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of nearest neighbors to use. Defaults to 5.
        metric : {"euclidean", "chebyshev", "cityblock"}, optional
            Distance metric used to find neighbors. Defaults to "euclidean".
        weights : {"uniform", "distance"}, optional
            How neighbors are weighted when voting. Defaults to "uniform".
        """
        super().__init__()
        self.n_neighbors: int = n_neighbors
        self.metric: Literal["euclidean", "chebyshev", "cityblock"] = metric
        self.weights: Literal["uniform", "distance"] = weights

    def _fit(self, X: FeatureMatrix, y: ClassificationTarget, /) -> None:
        """Validate n_neighbors and store the training data.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : ClassificationTarget
            Class labels of shape (n_samples,).

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
        self.classes_ = np.unique(y)

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        """Predict labels by a majority vote among the n_neighbors nearest training labels.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget
            Predicted labels of shape (n_samples,), computed as the
            (optionally distance-weighted) majority vote among each query
            point's nearest neighbors' labels.
        """
        dists = cdist(X, self.X_, metric=self.metric)
        neighbor_ind = np.argpartition(dists, kth=self.n_neighbors - 1, axis=1)[
            :, : self.n_neighbors
        ]
        if self.weights == "uniform":
            result = mode(self.y_[neighbor_ind], axis=1, keepdims=False)
            return cast(np.typing.NDArray[np.integer[Any]], result.mode)
        neighbor_dist = np.take_along_axis(dists, neighbor_ind, axis=1)
        neighbor_labels = self.y_[neighbor_ind]
        weights = 1 / (neighbor_dist + 1e-10)
        one_hot = neighbor_labels[:, :, np.newaxis] == self.classes_
        weighted_votes = one_hot * weights[:, :, np.newaxis]
        class_scores = np.sum(weighted_votes, axis=1)
        predicted_idx = np.argmax(class_scores, axis=1)
        return self.classes_[predicted_idx]
