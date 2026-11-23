"""Decision tree regressor fit via recursive variance-reduction splitting."""

from typing import cast

import numpy as np

from ..core.base import Regressor
from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ._base import _DecisionTreeBase, _FloatArray


class DecisionTreeRegressor(_DecisionTreeBase, Regressor):
    r"""Decision tree regressor fit via recursive splitting.

    At each node, the split that most reduces target variance is chosen:

    .. math::
        Var(y) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2

    Each leaf predicts the mean target among the samples that reach it.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree; must be non-negative. Splitting stops
        once a node reaches this depth, regardless of variance. A depth of 0
        gives a tree made of a single leaf.
    min_samples : int, optional
        Minimum number of samples required to consider splitting a node;
        must be at least 1. Nodes with fewer samples become leaves.
        Defaults to 2.
    min_impurity_decrease : float, optional
        Minimum variance reduction required to accept a split. A
        candidate split whose gain falls below this becomes a leaf
        instead. Must be non-negative. Defaults to 1e-7.
    max_features : int or None, optional
        Number of features drawn at random, without replacement, as split
        candidates at each node; must be at least 1 and at most the number
        of features seen in fit. None, or the number of features itself,
        uses every feature and makes the tree deterministic. Defaults to
        None.
    random_state : int or None, optional
        Seed for the generator that draws the candidate features; must be
        non-negative. Only used when max_features is smaller than the
        number of features. None draws a fresh seed on every fit. Defaults
        to None.

    Attributes
    ----------
    tree_ : _Node
        Root node of the fitted tree.


    .. note::
        Splitting is exhaustive: every unique value of every candidate
        feature is tried as a threshold. This scales as
        O(n_candidates x n_samples^2) per node, so very large datasets or
        deep trees can be slow to fit. A smaller max_features makes each
        node proportionally cheaper.

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
        raw_pred = super()._predict(X)
        return cast(RegressionTarget, raw_pred)

    def _weighted_child_impurity(
        self,
        y_sorted: ClassificationTarget | RegressionTarget,
        w_left: _FloatArray,
        w_right: _FloatArray,
        /,
    ) -> _FloatArray:
        """Compute the weighted variance of every split of one feature.

        The variance of a child is the mean of the squared targets minus the
        squared mean, taken from cumulative sums. The targets are first
        centred on their overall mean. Variance does not change under a
        shift, and centring keeps the two terms small, so the difference does
        not lose its digits when the targets are large compared with their
        spread.

        Parameters
        ----------
        y_sorted : ClassificationTarget or RegressionTarget
            Targets ordered by the feature's value, of shape (n_samples,).
        w_left : FloatArray
            Share of the samples that go left in each split, of shape
            (n_samples - 1,).
        w_right : FloatArray
            Share of the samples that go right in each split, of shape
            (n_samples - 1,).

        Returns
        -------
        FloatArray
            The weighted sum of the two child variances of each split, of
            shape (n_samples - 1,).
        """
        yc = y_sorted - y_sorted.mean()

        cum_sum = np.cumsum(yc)[:-1]
        cum_sq = np.cumsum(yc**2)[:-1]

        n_left = np.arange(1, y_sorted.shape[0], dtype=np.float64)
        n_right = y_sorted.shape[0] - n_left

        total_sum = yc.sum()
        total_sq = (yc**2).sum()

        var_left = cum_sq / n_left - (cum_sum / n_left) ** 2
        var_right = (total_sq - cum_sq) / n_right - ((total_sum - cum_sum) / n_right) ** 2

        impurity: _FloatArray = w_left * var_left + w_right * var_right
        return impurity
