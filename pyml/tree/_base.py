"""Abstract base class for CART-style binary decision trees.

Centralizes the shared recursive tree-building algorithm (greedy,
exhaustive threshold search per feature) and prediction traversal for
decision tree models. Subclasses supply their own split-quality
measure and leaf-value rule, allowing classification (Gini/entropy)
and regression (variance) trees to reuse the same tree-construction
machinery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    counts : numpy.ndarray or None, optional
        Number of training samples of each class that reached this leaf,
        of shape (n_classes,), in the order of the classes of the model.
        Only the leaves of a classification tree have it; it is None for
        internal nodes and for the leaves of a regression tree. It is left
        out of the equality of nodes.
    """

    is_leaf: bool
    value: int | float | None
    feature_idx: int | None = None
    threshold: float | None = None
    left: Self | None = None
    right: Self | None = None
    counts: npt.NDArray[np.intp] | None = field(default=None, compare=False)


@dataclass(eq=False)
class _FlatTree:
    """A fitted tree stored as flat arrays, one entry per node.

    Node ``i`` is the node at position ``i`` of a preorder walk, so the
    root is node 0 and the left child of an internal node comes right
    after it. Leaves have ``left == right == -1``. The arrays are what lets
    ``predict`` move every sample one level down with a few array lookups
    instead of walking ``_Node`` objects one sample at a time.

    Parameters
    ----------
    feature_idx : numpy.ndarray of int
        Feature each node splits on, of shape (n_nodes,). Unused (0) at leaves.
    threshold : numpy.ndarray of float
        Split threshold of each node, of shape (n_nodes,). Unused (0.0) at leaves.
    left, right : numpy.ndarray of int
        Index of the left and right child of each node, of shape (n_nodes,).
        -1 at leaves.
    value : numpy.ndarray
        Prediction stored at each node, of shape (n_nodes,). Only the entries of
        leaves are meaningful. The dtype is that of the leaf values (integer
        for classifiers, float for regressors).
    counts : numpy.ndarray of int or None
        Number of training samples of each class at each leaf, of shape
        (n_nodes, n_classes). None for a regression tree.
    """

    feature_idx: npt.NDArray[np.intp]
    threshold: npt.NDArray[np.float64]
    left: npt.NDArray[np.intp]
    right: npt.NDArray[np.intp]
    value: npt.NDArray[np.integer[Any] | np.floating[Any]]
    counts: npt.NDArray[np.intp] | None


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

    def _flatten_tree(self, root: _Node) -> _FlatTree:
        """Convert a fitted ``_Node`` tree into flat arrays.

        The nodes are numbered in preorder (a node, then its whole left
        subtree, then its right subtree), so the root is 0. The walk uses an
        explicit stack, so it makes one pass over the nodes and does not
        recurse.

        Parameters
        ----------
        root : _Node
            Root of the fitted tree.

        Returns
        -------
        _FlatTree
            The same tree as arrays, with ``n_nodes`` entries each.
        """
        # Preorder walk: pop a node, record it, push right first so left is popped first.
        nodes: list[_Node] = []
        stack = [root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            if not node.is_leaf:
                stack.append(cast(_Node, node.right))
                stack.append(cast(_Node, node.left))

        n_nodes = len(nodes)
        # The id of a node is its position in `nodes`; children are looked up by identity.
        node_id = {id(node): i for i, node in enumerate(nodes)}

        feature_idx = np.zeros(n_nodes, dtype=np.intp)
        threshold = np.zeros(n_nodes, dtype=np.float64)
        # -1 marks "no child", which is how predict recognizes a leaf.
        left = np.full(n_nodes, -1, dtype=np.intp)
        right = np.full(n_nodes, -1, dtype=np.intp)

        # The root can be an internal node, whose value and counts are None, so the dtype
        # of `value` and the number of classes are read from a leaf (the first one in preorder).
        first_leaf = next(node for node in nodes if node.is_leaf)
        is_int = isinstance(first_leaf.value, (int, np.integer))
        value = np.zeros(n_nodes, dtype=np.intp if is_int else np.float64)
        counts = (
            None
            if first_leaf.counts is None
            else np.zeros((n_nodes, first_leaf.counts.shape[0]), dtype=np.intp)
        )

        for i, node in enumerate(nodes):
            if node.is_leaf:
                value[i] = cast(float, node.value)
                if counts is not None and node.counts is not None:
                    counts[i] = node.counts
            else:
                feature_idx[i] = cast(int, node.feature_idx)
                threshold[i] = cast(float, node.threshold)
                left[i] = node_id[id(node.left)]
                right[i] = node_id[id(node.right)]

        return _FlatTree(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left,
            right=right,
            value=cast(npt.NDArray[np.integer[Any] | np.floating[Any]], value),
            counts=counts,
        )

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
        self._before_build(y)
        rng = np.random.default_rng(self.random_state)
        new_tree = self._build_tree(X, y, 0, rng)
        new_flat = self._flatten_tree(new_tree)

        # Both are set together, only after both are built: if the build or the flatten
        # raises, a previously fitted model is left intact and tree_ always matches _flat.
        self.tree_ = new_tree
        self._flat = new_flat

    def _before_build(self, y: ClassificationTarget | RegressionTarget, /) -> None:  # noqa: B027
        """Prepare the model for growing the tree.

        Called once per fit, after the checks that need the data and before
        the first node is built, so what a subclass sets here is not changed
        by a fit that those checks reject. A subclass overrides it to set
        what its leaves need, such as the classes of a classifier. The
        default does nothing on purpose, so it is not abstract.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Target values of shape (n_samples,).
        """

    def _apply(self, X: FeatureMatrix, /) -> npt.NDArray[np.intp]:
        """Find the leaf that each sample reaches.

        Uses the flattened arrays (``self._flat``) to move all samples through
        the tree at once, one level per iteration, instead of walking each
        sample individually. The value of a sample is compared with the
        threshold of its node in float64, and a value equal to the threshold
        goes left, as in fit.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray of int
            Index of the leaf reached by each sample, of shape (n_samples,).
            The index is the position of the leaf in the arrays of
            ``self._flat``.
        """
        n_samples = X.shape[0]
        # current_nodes[i] is the node sample i is at now (every sample starts at the root, 0).
        current_nodes = np.zeros(n_samples, dtype=np.intp)
        # active[i] is True while sample i has not reached a leaf and must still go down.
        active = np.ones(n_samples, dtype=bool)

        # One iteration moves every active sample down one level, so the number of
        # iterations is the height of the tree, not the number of samples.
        while np.any(active):
            nodes = current_nodes[active]
            # left == -1 only at leaves: that is how _flatten_tree marks them.
            is_leaf = self._flat.left[nodes] == -1
            active_indices = np.where(active)[0]

            # Samples that reached a leaf stop; current_nodes[i] stays at that leaf.
            active[active_indices[is_leaf]] = False

            internal_indices = active_indices[~is_leaf]
            if len(internal_indices) == 0:
                break

            # Feature and threshold of the node each remaining sample is at, for all
            # samples at once (fancy indexing).
            internal_nodes = current_nodes[internal_indices]
            features = self._flat.feature_idx[internal_nodes]
            thresholds = self._flat.threshold[internal_nodes]

            # Each sample reads the feature of its own node: X[i, features[i]].
            feature_values = X[internal_indices, features]
            # Same rule as in fit: <= goes left, > goes right.
            go_left = feature_values <= thresholds

            next_nodes = np.where(
                go_left, self._flat.left[internal_nodes], self._flat.right[internal_nodes]
            )
            current_nodes[internal_indices] = next_nodes

        return current_nodes

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget | RegressionTarget:
        """Predict by reading the value of the leaf each sample reaches.

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
        leaves = self._apply(X)
        return cast(ClassificationTarget | RegressionTarget, self._flat.value[leaves])

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
        # Stopping rules: max depth reached, too few samples, or all targets equal (pure node).
        if depth >= self.max_depth or n < self.min_samples or (np.all(y == y[0])):
            return self._make_leaf(y)
        split_result = self._best_split(X, y, rng)
        if split_result is None:
            return self._make_leaf(y)
        feature_idx, threshold, quality_gain = split_result
        if quality_gain < self.min_impurity_decrease:
            return self._make_leaf(y)
        # predict uses this same `<=` rule, so the split by mask here and the later
        # traversal send every sample the same way.
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

    def _make_leaf(self, y: ClassificationTarget | RegressionTarget, /) -> _Node:
        """Create the leaf for the samples that reach a node that stops growing.

        The leaf stores the value from ``_leaf_node``. A subclass extends it
        when its leaves keep more than one number, as the class counts of a
        classifier.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Targets at this leaf, of shape (n_samples,).

        Returns
        -------
        _Node
            A leaf node holding the value of the targets `y`.
        """
        return _Node(is_leaf=True, value=self._leaf_node(y))

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
        first. The impurity of every threshold of one feature is computed
        at once by ``_weighted_child_impurity``, so the search has no Python
        loop over the samples.

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
        n_samples = X.shape[0]
        # rng is used only when max_features < n_features; otherwise the tree is
        # deterministic and the state of rng does not change.
        if self.max_features is None or self.max_features == X.shape[1]:
            features = np.arange(X.shape[1])
        else:
            features = rng.choice(X.shape[1], self.max_features, replace=False)

        current_impurity = self._split_quality(y)
        # Split i sends the first i samples left (i = 1..n-1). The weights do not depend
        # on the feature, so they are computed once, before the loop.
        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left
        w_left = n_left / n_samples
        w_right = n_right / n_samples

        best_gain: float = -1.0
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature_id in features:
            feature_col = X[:, feature_id]
            sort_idx = np.argsort(feature_col, kind="stable")
            X_sorted = feature_col[sort_idx]
            y_sorted = y[sort_idx]

            # A threshold cannot go between two equal values, so only the places where the
            # value changes are candidates. A constant feature has none and is skipped.
            distinct = X_sorted[1:] != X_sorted[:-1]
            if not np.any(distinct):
                continue
            child_impurity = self._weighted_child_impurity(y_sorted, w_left, w_right)

            gains = current_impurity - child_impurity
            # -inf so that an invalid split can never win the argmax.
            gains = np.where(distinct, gains, -np.inf)

            local_idx = int(np.argmax(gains))
            local_gain = float(gains[local_idx])
            # Strict `>`: on equal gain the candidate examined first is kept (tie rule).
            if local_gain > best_gain:
                best_gain = local_gain
                best_feature = int(feature_id)
                # The threshold is the midpoint of the two neighbouring values.
                best_threshold = float(X_sorted[local_idx] + X_sorted[local_idx + 1]) / 2.0

        if best_feature is None or best_threshold is None:
            return None

        return int(best_feature), float(best_threshold), best_gain

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

    @abstractmethod
    def _weighted_child_impurity(
        self,
        y_sorted: ClassificationTarget | RegressionTarget,
        w_left: _FloatArray,
        w_right: _FloatArray,
        /,
    ) -> _FloatArray:
        """Compute the weighted child impurity of every split of one feature.

        The targets are ordered by the value of one feature. Split ``i``
        (``i = 1, ..., n_samples - 1``) sends the first ``i`` targets to the
        left child and the rest to the right child. Subclasses implement
        this with cumulative sums over the sorted targets, so the targets of
        each split are not visited again.

        Parameters
        ----------
        y_sorted : ClassificationTarget or RegressionTarget
            Targets ordered by the feature's value, of shape (n_samples,).
        w_left : FloatArray
            Share of the samples that go left in each split, ``i / n_samples``,
            of shape (n_samples - 1,).
        w_right : FloatArray
            Share of the samples that go right in each split,
            ``(n_samples - i) / n_samples``, of shape (n_samples - 1,).

        Returns
        -------
        FloatArray
            For each split, the impurity of the left child times `w_left`
            plus the impurity of the right child times `w_right`, of shape
            (n_samples - 1,). The gain of a split is the impurity of the
            node minus this value.
        """
