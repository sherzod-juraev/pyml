"""Decision tree regressor fit via recursive variance-reduction splitting."""

from typing import cast

import numpy as np

from ..core.base import Regressor
from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ._base import _DecisionTreeBase


class DecisionTreeRegressor(_DecisionTreeBase, Regressor):
    r"""Decision tree regressor fit via recursive splitting.

    At each node, the split that most reduces target variance is chosen:

    .. math::
        Var(y) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2

    Each leaf predicts the mean target among the samples that reach it.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree. Splitting stops once a node reaches
        this depth, regardless of variance.
    min_samples : int, optional
        Minimum number of samples required to consider splitting a node.
        Nodes with fewer samples become leaves. Defaults to 2.
    min_impurity_decrease : float, optional
        Minimum variance reduction required to accept a split. A
        candidate split whose gain falls below this becomes a leaf
        instead. Defaults to 1e-7.

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
        from pyml.tree import DecisionTreeRegressor

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(60, 1)), axis=0)
        y = np.sin(X).ravel() + rng.normal(0, 0.1, size=60)

        model = DecisionTreeRegressor(max_depth=3).fit(X, y)
        X_line = np.linspace(0, 10, 200).reshape(-1, 1)

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.6, label="Training data")
        ax.plot(
            X_line,
            model.predict(X_line),
            color="red",
            label="DecisionTreeRegressor (max_depth=3)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("DecisionTreeRegressor fit")
        ax.legend(loc="best")
    """

    def _split_quality(self, y: ClassificationTarget | RegressionTarget, /) -> float:
        r"""Compute the variance of this node's targets.

        .. math::
            Var(y) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Targets at this node, of shape (n_samples,).

        Returns
        -------
        float
            The variance of `y`. 0.0 for an empty node.
        """
        return float(np.var(y)) if y.shape[0] > 0 else 0.0

    def _leaf_node(self, y: ClassificationTarget | RegressionTarget, /) -> int | float:
        """Compute the mean of the targets at a leaf.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Targets at this leaf, of shape (n_samples,).

        Returns
        -------
        int or float
            The mean of `y`. 0.0 for an empty leaf.
        """
        return float(np.mean(y)) if y.shape[0] > 0 else 0.0

    def _predict(self, X: FeatureMatrix, /) -> RegressionTarget:
        """Predict continuous values by traversing the fitted tree.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        RegressionTarget
            Predicted value for each sample, of shape (n_samples,).
        """
        return cast(RegressionTarget, super()._predict(X))
