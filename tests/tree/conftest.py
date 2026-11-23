"""Shared fixtures for the decision tree tests.

Provide a slow, exhaustive reference for the split search: it tries every
threshold of every feature and measures each candidate with the model's own
``_split_quality``, so it shares no code with the cumulative-sum search it
checks. Likewise, the prediction is checked against a walk of the ``_Node``
tree, one sample at a time, which shares no code with the array traversal
over the flat tree.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pyml.tree._base import _DecisionTreeBase, _Node

SplitGain = Callable[[_DecisionTreeBase, np.ndarray, np.ndarray, int, float], float]
ExhaustiveSearch = Callable[[_DecisionTreeBase, np.ndarray, np.ndarray], tuple[int, float, float]]
PreorderNodes = Callable[[_Node], list[_Node]]
LeafOfEachSample = Callable[[Any, np.ndarray], list[_Node]]


@pytest.fixture
def split_gain() -> SplitGain:
    """Return a function that measures the gain of one given split.

    Returns
    -------
    callable
        ``gain(model, X, y, feature, threshold)`` returning the impurity of
        the node minus the weighted impurity of the two children made by
        sending ``X[:, feature] <= threshold`` to the left.
    """

    def gain(
        model: _DecisionTreeBase,
        X: np.ndarray,
        y: np.ndarray,
        feature: int,
        threshold: float,
    ) -> float:
        left = X[:, feature] <= threshold
        n = y.shape[0]
        children = (left.sum() / n) * model._split_quality(y[left]) + (
            (~left).sum() / n
        ) * model._split_quality(y[~left])
        return float(model._split_quality(y) - children)

    return gain


@pytest.fixture
def exhaustive_best_split(split_gain: SplitGain) -> ExhaustiveSearch:
    """Return a slow search that tries every threshold of every feature.

    Returns
    -------
    callable
        ``search(model, X, y)`` returning ``(feature, threshold, gain)`` of
        the best split. Thresholds are the midpoints between distinct
        consecutive values, as in the model.
    """

    def search(model: _DecisionTreeBase, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        best: tuple[int, float, float] | None = None
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for low, high in zip(values[:-1], values[1:], strict=True):
                threshold = float(low + high) / 2.0
                gain = split_gain(model, X, y, feature, threshold)
                if best is None or gain > best[2]:
                    best = (feature, threshold, gain)
        assert best is not None
        return best

    return search


@pytest.fixture
def preorder_nodes() -> PreorderNodes:
    """Return a function that lists the nodes of a tree in preorder.

    Returns
    -------
    callable
        ``nodes(root)`` returning the root, then all the nodes of its left
        subtree, then all the nodes of its right subtree.
    """

    def nodes(root: _Node) -> list[_Node]:
        if root.is_leaf:
            return [root]
        assert root.left is not None
        assert root.right is not None
        return [root, *nodes(root.left), *nodes(root.right)]

    return nodes


@pytest.fixture
def leaf_of_each_sample() -> LeafOfEachSample:
    """Return a function that walks the fitted tree, one sample at a time.

    Returns
    -------
    callable
        ``leaves(model, X)`` returning, for each row of X, the leaf of
        ``model.tree_`` that the row reaches by the rule ``<=`` goes left.
        It reads only the ``_Node`` tree, never the flat arrays.
    """

    def leaves(model: Any, X: np.ndarray) -> list[_Node]:
        reached = []
        for x in X:
            node: _Node = model.tree_
            while not node.is_leaf:
                assert node.feature_idx is not None
                assert node.threshold is not None
                assert node.left is not None
                assert node.right is not None
                node = node.left if x[node.feature_idx] <= node.threshold else node.right
            reached.append(node)
        return reached

    return leaves
