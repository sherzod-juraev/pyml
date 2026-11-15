"""Decision tree classifier fit via recursive Gini-impurity splitting."""

from typing import cast

import numpy as np

from ..core.base import Classifier
from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ._base import _DecisionTreeBase, _FloatArray, _Node


class DecisionTreeClassifier(_DecisionTreeBase, Classifier):
    r"""Decision tree classifier fit via recursive splitting.

    .. versionchanged:: 0.4.6
        Added ``classes_`` and ``predict_proba``.

    At each node, the split that most reduces Gini impurity is chosen.
    For a node with class proportions :math:`p_1, \dots, p_k`:

    .. math::
        Gini = 1 - \sum_{i=1}^{k} p_i^2

    Gini is 0 when a node is pure (all samples share one class) and
    approaches 1 as classes become more evenly mixed. Each leaf predicts
    the majority class among the samples that reach it, and
    ``predict_proba`` returns the share of each class among those
    samples.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree; must be non-negative. Splitting stops
        once a node reaches this depth, regardless of impurity. A depth of 0
        gives a tree made of a single leaf.
    min_samples : int, optional
        Minimum number of samples required to consider splitting a node;
        must be at least 1. Nodes with fewer samples become leaves.
        Defaults to 2.
    min_impurity_decrease : float, optional
        Minimum impurity gain required to accept a split. A candidate
        split whose gain falls below this becomes a leaf instead. Must
        be non-negative. Defaults to 1e-7.
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
    classes_ : ClassificationTarget
        Sorted class labels seen in fit, of shape (n_classes,). The columns
        of ``predict_proba`` follow this order.


    .. note::
        Splitting is exhaustive: every unique value of every candidate
        feature is tried as a threshold. This scales as
        O(n_candidates x n_samples^2) per node, so very large datasets or
        deep trees can be slow to fit. A smaller max_features makes each
        node proportionally cheaper.

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

    def _before_build(self, y: ClassificationTarget | RegressionTarget, /) -> None:
        """Set ``classes_``, the order in which the leaves count the classes.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Class labels of shape (n_samples,).
        """
        self.classes_ = np.unique(y)

    def _make_leaf(self, y: ClassificationTarget | RegressionTarget, /) -> _Node:
        """Create a leaf that also counts the samples of each class.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Class labels at this leaf, of shape (n_samples,).

        Returns
        -------
        _Node
            A leaf holding the majority class and, in ``counts``, the number
            of samples of each class of ``classes_``.
        """
        leaf = super()._make_leaf(y)
        leaf.counts = np.bincount(
            np.searchsorted(self.classes_, y), minlength=self.classes_.shape[0]
        )
        return leaf

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

    def predict_proba(self, X: FeatureMatrix, /) -> _FloatArray:
        """Predict the class probabilities for each sample.

        The probabilities of a sample are the shares of the classes among the
        training samples in the leaf it reaches, so a pure leaf gives 0 and 1.
        The class with the largest share is the one ``predict`` returns, and
        on a tie both give the smallest label. Validates the input, then
        delegates to ``_predict_proba``.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Class probabilities of shape (n_samples, n_classes). The columns
            follow ``classes_`` and every row sums to 1.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If X is not 2D, or contains NaN or infinite values.
        TypeError
            If X's dtype is not floating-point.
        ShapeMismatchError
            If X has a different number of features than seen in fit.
        """
        self._check_is_fitted()
        self._validate_X(X)
        self._check_n_features(X)
        return self._predict_proba(X)

    def _predict_proba(self, X: FeatureMatrix, /) -> _FloatArray:
        """Compute the class probabilities of each sample from its leaf.

        Does no input validation: it is called by ``predict_proba`` after the
        checks.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Class probabilities of shape (n_samples, n_classes).
        """
        counts = np.array(
            [self._find_leaf(x, self.tree_).counts for x in X], dtype=np.float64
        ).reshape(X.shape[0], self.classes_.shape[0])
        probabilities: _FloatArray = counts / counts.sum(axis=1, keepdims=True)
        return probabilities
