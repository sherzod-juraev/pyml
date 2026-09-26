"""Decision tree classifier fit via recursive Gini-impurity splitting."""

from typing import cast

import numpy as np

from ..core.base import Classifier
from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ._base import _DecisionTreeBase


class DecisionTreeClassifier(_DecisionTreeBase, Classifier):
    r"""Decision tree classifier fit via recursive splitting.

    At each node, the split that most reduces Gini impurity is chosen.
    For a node with class proportions :math:`p_1, \dots, p_k`:

    .. math::
        Gini = 1 - \sum_{i=1}^{k} p_i^2

    Gini is 0 when a node is pure (all samples share one class) and
    approaches 1 as classes become more evenly mixed. Each leaf predicts
    the majority class among the samples that reach it.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree. Splitting stops once a node reaches
        this depth, regardless of impurity.
    min_samples : int, optional
        Minimum number of samples required to consider splitting a node.
        Nodes with fewer samples become leaves. Defaults to 2.
    min_impurity_decrease : float, optional
        Minimum impurity gain required to accept a split. A candidate
        split whose gain falls below this becomes a leaf instead.
        Defaults to 1e-7.

    Attributes
    ----------
    tree_ : _Node
        Root node of the fitted tree.


    .. note::
        Splitting is exhaustive: every unique value of every feature is
        tried as a candidate threshold. This scales as
        O(n_features x n_samples^2) per tree, so very large datasets or
        deep trees can be slow to fit.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.tree import DecisionTreeClassifier

        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(50, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = DecisionTreeClassifier(max_depth=3).fit(X, y)

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
        )
        grid_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, grid_pred.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("DecisionTreeClassifier decision boundary (max_depth=3)")
    """

    def _split_quality(self, y: ClassificationTarget | RegressionTarget, /) -> float:
        r"""Compute the Gini impurity of this node's class distribution.

        .. math::
            Gini = 1 - \sum_{i=1}^{k} p_i^2

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Targets at this node, of shape (n_samples,).

        Returns
        -------
        float
            The Gini impurity, in [0, 1). 0.0 for an empty or pure node.
        """
        if y.shape[0] == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.shape[0]
        return float(1.0 - np.sum(probabilities**2))

    def _leaf_node(self, y: ClassificationTarget | RegressionTarget, /) -> int | float:
        """Compute the majority class among the targets at a leaf.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Targets at this leaf, of shape (n_samples,).

        Returns
        -------
        int or float
            The most frequent class label in `y`.
        """
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        """Predict class labels by traversing the fitted tree.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget
            Predicted class label for each sample, of shape (n_samples,).
        """
        return cast(ClassificationTarget, super()._predict(X))
