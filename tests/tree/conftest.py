"""Shared fixtures for the decision tree tests.

Provide a slow, exhaustive reference for the split search: it tries every
threshold of every feature and measures each candidate with the model's own
``_split_quality``, so it shares no code with the cumulative-sum search it
checks.
"""

from collections.abc import Callable

import numpy as np
import pytest

from pyml.tree._base import _DecisionTreeBase

SplitGain = Callable[[_DecisionTreeBase, np.ndarray, np.ndarray, int, float], float]
ExhaustiveSearch = Callable[[_DecisionTreeBase, np.ndarray, np.ndarray], tuple[int, float, float]]


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
