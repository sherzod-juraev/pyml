"""Random forest regressor averaging the predictions of decision trees."""

import numpy as np

from ..core.base import Regressor
from ..core.dtypes import FeatureMatrix, RegressionTarget
from ..tree import DecisionTreeRegressor
from ._base import _ForestBase


class RandomForestRegressor(_ForestBase[DecisionTreeRegressor], Regressor):
    r"""Random forest regressor averaging the predictions of decision trees.

    .. versionadded:: 0.5.0

    The forest grows ``n_estimators`` regression trees. Each tree is fitted on
    its own bootstrap sample of the training rows, and at every node it
    considers only a random subset of ``max_features`` features as split
    candidates. These two sources of randomness make the trees differ from one
    another, so averaging them lowers the variance of a single deep tree while
    its bias stays nearly the same.

    A leaf of a tree predicts the mean target of the training samples that
    reach it, and the forest predicts the mean of the :math:`B` trees:

    .. math::
        \hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b(x)

    Every tree has the same weight, whatever the size of its leaves. The trees
    are fully reproducible from ``random_state``.

    Parameters
    ----------
    max_depth : int
        Maximum depth of every tree; must be non-negative. Passed to each tree
        unchanged.
    n_estimators : int, optional
        Number of trees in the forest; must be at least 1. A forest of a single
        tree is allowed. Defaults to 100.
    min_samples : int, optional
        Minimum number of samples required to consider splitting a node of a
        tree; must be at least 1. Defaults to 2.
    min_impurity_decrease : float, optional
        Minimum impurity gain required to accept a split in a tree; must be
        non-negative. Defaults to 1e-7.
    max_features : int, {"sqrt"} or None, optional
        Number of features drawn at random, without replacement, as split
        candidates at each node of every tree. An int must be at least 1 and at
        most the number of features seen in fit. "sqrt" uses the integer part
        of the square root of the number of features. None uses every feature,
        so only the bootstrap decorrelates the trees. Defaults to "sqrt".
    bootstrap : bool, optional
        If True, each tree is grown on a bootstrap sample: as many rows as the
        training set, drawn with replacement. If False, every tree sees the
        whole training set. Defaults to True.
    random_state : int or None, optional
        Seed that makes the forest reproducible; must be non-negative. It
        controls the bootstrap samples and the feature sampling of the trees.
        None draws fresh randomness on every fit. Defaults to None.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The fitted trees, in the order they were grown, of length
        ``n_estimators``.


    .. note::
        The default ``max_features="sqrt"`` is the usual choice for
        classification. For regression, a common choice is about a third of the
        features, which is set with an int, for example
        ``max_features=n_features // 3``, or ``max_features=None`` to let every
        node see all the features.

    .. note::
        The trees are fitted one after another, so the fit time grows linearly
        with ``n_estimators`` on top of the cost of a single tree, whose
        splitting is exhaustive. Prediction visits every tree as well. With
        ``n_estimators=1``, ``bootstrap=False`` and ``max_features=None`` the
        forest is a single ``DecisionTreeRegressor``.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.ensemble import RandomForestRegressor

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 6, size=(80, 1)), axis=0)
        y = np.sin(X[:, 0]) + rng.normal(scale=0.3, size=80)

        model = RandomForestRegressor(max_depth=4, n_estimators=25, random_state=42).fit(X, y)

        grid = np.linspace(0, 6, 200).reshape(-1, 1)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], y, color="gray", edgecolor="k", label="data")
        ax.plot(grid[:, 0], model.predict(grid), color="C3", label="forest (25 trees)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.set_title("RandomForestRegressor fit to a noisy sine curve")
        ax.legend()
    """

    def _make_tree(
        self, random_state: int | None, max_features: int | None, /
    ) -> DecisionTreeRegressor:
        """Build one unfitted regression tree for the forest.

        Parameters
        ----------
        random_state : int or None
            Seed of the tree, used to draw the candidate features at each node.
        max_features : int or None
            Number of candidate features per node, already resolved from the
            max_features of the forest; None uses every feature.

        Returns
        -------
        DecisionTreeRegressor
            An unfitted tree created with the max_depth, min_samples and
            min_impurity_decrease of the forest and the given max_features and
            random_state.
        """
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples=self.min_samples,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=max_features,
            random_state=random_state,
        )

    def _predict(self, X: FeatureMatrix, /) -> RegressionTarget:
        """Predict the mean of the predictions of the trees.

        Does no input validation: it is called by ``predict`` after the checks.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        RegressionTarget
            Predicted value for each sample, of shape (n_samples,).
        """
        n_samples = X.shape[0]
        total = np.zeros(n_samples)
        for tree in self.estimators_:
            total += tree._predict(X)
        return total / self.n_estimators
