"""Abstract base class for CART-style binary decision trees.

Centralizes the shared recursive tree-building algorithm (greedy,
exhaustive threshold search per feature) and prediction traversal for
decision tree models. Subclasses supply their own split-quality
measure and leaf-value rule, allowing classification (Gini/entropy)
and regression (variance) trees to reuse the same tree-construction
machinery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt

from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ..core.exceptions import InvalidParameterError
from ..core.param_checks import check_integer, check_non_negative

_FloatArray = npt.NDArray[np.floating[Any]]


@dataclass
class _Node:
    """A single node in a fitted decision tree.

    Parameters
    ----------
    is_leaf : bool
        Whether this node is a leaf (holds a prediction) or an internal
        split node (holds a feature/threshold and children).
    value : int, float, or None
        Prediction stored at a leaf (majority class or mean target).
        None for internal nodes.
    feature_idx : int or None, optional
        Index of the feature this node splits on. None for leaves.
    threshold : float or None, optional
        Split threshold: samples with feature value at or below this go
        left, samples above go right. None for leaves.
    left : _Node or None, optional
        Left child, containing samples where feature_idx <= threshold.
    right : _Node or None, optional
        Right child, containing samples where feature_idx > threshold.
    """

    is_leaf: bool
    value: int | float | None
    feature_idx: int | None = None
    threshold: float | None = None
    left: Self | None = None
    right: Self | None = None


class _DecisionTreeBase(ABC):
    r"""Abstract base class for binary decision trees fit via recursive splitting.

    All decision tree models share the same greedy, top-down
    construction: at each node, the candidate features are scanned for
    the threshold that best separates the samples, splitting is applied
    where it reduces impurity the most, and the process repeats on each
    child until a stopping condition is met.

    For a candidate split of a node's samples into a left and right
    subset, the quality of the split is measured by the reduction in
    impurity it produces:

    .. math::
        Gain = I(parent) - \left(
            \frac{n_{left}}{n} I(left) + \frac{n_{right}}{n} I(right)
        \right)

    where :math:`I` is the impurity measure (Gini impurity, entropy, or
    variance, depending on the subclass), :math:`n` is the number of
    samples at the parent node, and :math:`n_{left}`, :math:`n_{right}`
    are the sample counts in each child. The threshold and feature that
    maximize this gain are chosen at each node.

    Subclasses differ only in how impurity and leaf values are computed,
    which lets them add new splitting criteria without duplicating the
    tree-construction loop.

    By default every feature is a candidate at every node and the tree is
    fully deterministic. With ``max_features`` smaller than the number of
    features, each node draws its own random subset of ``max_features``
    candidates, without replacement, from a generator seeded by
    ``random_state``. This is the per-node feature sampling used by
    random forests to decorrelate their trees.

    Attributes
    ----------
    tree_ : _Node
        Root node of the fitted tree.


    .. note::
        Splitting is exhaustive: every unique value of every candidate
        feature is tried as a threshold. This is simple and exact, but
        scales as O(n_candidates x n_samples^2) per node, since each
        candidate split recomputes impurity over its full subset.
    """

    def __init__(
        self,
        max_depth: int,
        min_samples: int = 2,
        min_impurity_decrease: float = 1e-7,
        max_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize tree-growth stopping criteria.

        Parameters
        ----------
        max_depth : int
            Maximum depth of the tree; must be non-negative. Splitting
            stops once a node reaches this depth, regardless of impurity.
            A depth of 0 gives a tree made of a single leaf.
        min_samples : int, optional
            Minimum number of samples required to consider splitting a node;
            must be at least 1. Nodes with fewer samples become leaves.
            Defaults to 2.
        min_impurity_decrease : float, optional
            Minimum impurity gain required to accept a split. A candidate
            split whose gain falls below this becomes a leaf instead; must
            be non-negative. Defaults to 1e-7.
        max_features : int or None, optional
            Number of features drawn at random, without replacement, as
            split candidates at each node; must be at least 1 and at most
            the number of features seen in fit. None, or the number of
            features itself, uses every feature and makes the tree
            deterministic. Defaults to None.
        random_state : int or None, optional
            Seed for the generator that draws the candidate features; must
            be non-negative. Only used when max_features is smaller than
            the number of features. None draws a fresh seed on every fit.
            Defaults to None.
        """
        super().__init__()
        self.max_depth: int = max_depth
        self.min_samples: int = min_samples
        self.min_impurity_decrease: float = min_impurity_decrease
        self.max_features: int | None = max_features
        self.random_state: int | None = random_state

    def _validate_params(self) -> None:
        """Validate the tree-growth and feature-sampling parameters.

        Raises
        ------
        InvalidParameterError
            If max_depth is not an integer or is negative, min_samples is
            not an integer or is less than 1, min_impurity_decrease is
            negative or NaN, max_features is not None and is not an
            integer of at least 1, or random_state is not None and is not
            a non-negative integer.
        """
        # This class sits in front of the family base in the MRO, so super() reaches
        # DataValidatorMixin; mypy cannot see that from a plain ABC base.
        cast(Any, super())._validate_params()
        check_integer("max_depth", self.max_depth, minimum=0)
        check_integer("min_samples", self.min_samples, minimum=1)
        check_non_negative("min_impurity_decrease", self.min_impurity_decrease)
        if self.max_features is not None:
            check_integer("max_features", self.max_features, minimum=1)
        if self.random_state is not None:
            check_integer("random_state", self.random_state, minimum=0)

    def _fit(self, X: FeatureMatrix, y: ClassificationTarget | RegressionTarget, /) -> None:
        """Grow the tree recursively from the root.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : ClassificationTarget or RegressionTarget
            Target values of shape (n_samples,).

        Raises
        ------
        InvalidParameterError
            If max_features exceeds the number of features in X.
        """
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise InvalidParameterError(
                f"max_features={self.max_features} cannot exceed the number of "
                f"features ({X.shape[1]})."
            )
        rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build_tree(X, y, 0, rng)

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget | RegressionTarget:
        """Predict by traversing the fitted tree for each sample.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget or RegressionTarget
            Predicted values of shape (n_samples,), one leaf value per
            sample.
        """
        return cast(
            ClassificationTarget | RegressionTarget,
            np.array([self._traverse(x, self.tree_) for x in X]),
        )

    def _build_tree(
        self,
        X: FeatureMatrix,
        y: ClassificationTarget | RegressionTarget,
        depth: int,
        rng: np.random.Generator,
        /,
    ) -> _Node:
        """Recursively build a subtree for the given samples.

        A node becomes a leaf when max_depth is reached, too few samples
        remain, all targets are already identical, no split improves on the
        parent, or the best gain falls below min_impurity_decrease.
        Otherwise, the best split is applied and the process repeats on each
        child.

        Parameters
        ----------
        X : FeatureMatrix
            Samples at this node, of shape (n_samples, n_features).
        y : ClassificationTarget or RegressionTarget
            Targets at this node, of shape (n_samples,).
        depth : int
            Current depth of this node, with the root at depth 0.
        rng : numpy.random.Generator
            Generator shared by the whole tree, used to draw the candidate
            features at each node.

        Returns
        -------
        _Node
            The root of the constructed subtree.
        """
        n = X.shape[0]
        if depth >= self.max_depth or n < self.min_samples or (np.all(y == y[0])):
            return _Node(is_leaf=True, value=self._leaf_node(y))
        split_result = self._best_split(X, y, rng)
        if split_result is None:
            return _Node(is_leaf=True, value=self._leaf_node(y))
        feature_idx, threshold, quality_gain = split_result
        if quality_gain < self.min_impurity_decrease:
            return _Node(is_leaf=True, value=self._leaf_node(y))
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)

        return _Node(
            is_leaf=False,
            threshold=threshold,
            feature_idx=feature_idx,
            left=left_child,
            right=right_child,
            value=None,
        )

    def _best_split(
        self,
        X: FeatureMatrix,
        y: ClassificationTarget | RegressionTarget,
        rng: np.random.Generator,
        /,
    ) -> tuple[int, float, float] | None:
        """Search the candidate features and thresholds for the highest-gain split.

        The candidates are every feature when max_features is None or equal
        to the number of features, and otherwise a fresh random subset of
        max_features features drawn from rng. For each candidate, samples
        are sorted by that feature's value, and every point between two
        distinct consecutive values is tried as a threshold, using the
        midpoint between them. Ties in gain go to the candidate examined
        first.

        Parameters
        ----------
        X : FeatureMatrix
            Samples at this node, of shape (n_samples, n_features).
        y : ClassificationTarget or RegressionTarget
            Targets at this node, of shape (n_samples,).
        rng : numpy.random.Generator
            Generator used to draw the candidate features. Not used when
            every feature is a candidate.

        Returns
        -------
        tuple[int, float, float] or None
            The best (feature_idx, threshold, gain) found, or None if no
            valid split exists (e.g. every candidate feature is constant).
        """
        best_gain: float = -1.0
        best_feature: int | None = None
        best_threshold: float | None = None

        n_samples = X.shape[0]
        if self.max_features is None or self.max_features == X.shape[1]:
            features = np.arange(X.shape[1])
        else:
            features = rng.choice(X.shape[1], self.max_features, replace=False)
        current_impurity = self._split_quality(y)

        for feature_id in features:
            feature_col = X[:, feature_id]
            sort_idx = np.argsort(feature_col)
            X_sorted = feature_col[sort_idx]
            y_sorted = y[sort_idx]

            for i in range(1, n_samples):
                if X_sorted[i - 1] == X_sorted[i]:
                    continue

                y_left = y_sorted[:i]
                y_right = y_sorted[i:]

                n_left, n_right = len(y_left), len(y_right)
                impurity_left = self._split_quality(y_left)
                impurity_right = self._split_quality(y_right)

                gain = current_impurity - (
                    (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature_id)
                    best_threshold = float(X_sorted[i - 1] + X_sorted[i]) / 2.0

        if best_feature is None or best_threshold is None:
            return None

        return int(best_feature), float(best_threshold), best_gain

    def _traverse(self, x: _FloatArray, node: _Node, /) -> int | float:
        """Follow a single sample down the tree to its leaf value.

        Parameters
        ----------
        x : FloatArray
            A single sample's feature values, of shape (n_features,).
        node : _Node
            The node to evaluate, starting from the root.

        Returns
        -------
        int or float
            The value stored at the leaf reached by this sample.
        """
        if node.is_leaf:
            return cast(int | float, node.value)
        feature_idx = cast(int, node.feature_idx)
        threshold = cast(float, node.threshold)
        left, right = cast(_Node, node.left), cast(_Node, node.right)
        if x[feature_idx] <= threshold:
            return self._traverse(x, left)
        return self._traverse(x, right)

    @abstractmethod
    def _split_quality(self, y: ClassificationTarget | RegressionTarget, /) -> float:
        """Compute this node's impurity before splitting — lower is purer.

        Subclasses implement this with their specific measure (e.g.
        Gini impurity or entropy for classification, variance for
        regression).

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Targets at this node, of shape (n_samples,).

        Returns
        -------
        float
            The impurity value.
        """

    @abstractmethod
    def _leaf_node(self, y: ClassificationTarget | RegressionTarget, /) -> int | float:
        """Compute the value to store at a leaf.

        Subclasses implement this with their specific rule (e.g. the
        majority class for classification, the mean target for
        regression).

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Targets at this leaf, of shape (n_samples,).

        Returns
        -------
        int or float
            The value to predict for samples reaching this leaf.
        """
