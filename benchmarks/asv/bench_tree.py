"""Benchmarks of the decision trees: the time of ``fit`` and of ``predict``.

``DecisionTreeRegressorBench`` is the first benchmark, a shallow tree (depth 6). The ``*Predict``
classes use deep trees: the cost of ``predict`` grows with the depth, and the forests of the
examples use trees of depth 8 and 20.
"""

from typing import ClassVar

from pyml.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ._data import (
    N_SMALL,
    N_TREE_LARGE,
    classification_data,
    nonlinear_regression_data,
)

_N_FEATURES = 6
# The depths of the trees in the examples of the forests.
_DEPTHS = [8, 20]


class DecisionTreeRegressorBench:
    """Time of ``DecisionTreeRegressor.fit`` and ``predict`` for a growing number of samples."""

    params: ClassVar[list[list[int]]] = [[N_SMALL, N_TREE_LARGE]]
    param_names: ClassVar[list[str]] = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        """Create a non-linear data set and fit a tree for ``predict``. This is not timed."""
        self.X, self.y = nonlinear_regression_data(n_samples, _N_FEATURES)
        self.tree = DecisionTreeRegressor(max_depth=6).fit(self.X, self.y)

    def time_fit(self, n_samples: int) -> None:
        """Fit a tree of depth 6."""
        DecisionTreeRegressor(max_depth=6).fit(self.X, self.y)

    def time_predict(self, n_samples: int) -> None:
        """Predict the training samples with the fitted tree."""
        self.tree.predict(self.X)


class DecisionTreeRegressorPredict:
    """Time of ``DecisionTreeRegressor.predict`` of a deep tree.

    The tree is fitted in ``setup`` on noisy data, so it grows to ``max_depth``.
    """

    params: ClassVar[list[list[int]]] = [[N_SMALL, N_TREE_LARGE], _DEPTHS]
    param_names: ClassVar[list[str]] = ["n_samples", "max_depth"]

    def setup(self, n_samples: int, max_depth: int) -> None:
        """Create a non-linear data set and fit the tree. This is not timed."""
        self.X, y = nonlinear_regression_data(n_samples, _N_FEATURES)
        self.tree = DecisionTreeRegressor(max_depth=max_depth).fit(self.X, y)

    def time_predict(self, n_samples: int, max_depth: int) -> None:
        """Predict the training samples with the fitted tree."""
        self.tree.predict(self.X)


class DecisionTreeClassifierPredict:
    """Time of ``DecisionTreeClassifier.predict`` and ``predict_proba`` of a deep tree.

    The classes overlap, so the tree grows to ``max_depth`` or close to it.
    """

    params: ClassVar[list[list[int]]] = [[N_SMALL, N_TREE_LARGE], _DEPTHS]
    param_names: ClassVar[list[str]] = ["n_samples", "max_depth"]

    def setup(self, n_samples: int, max_depth: int) -> None:
        """Create two overlapping classes and fit the tree. This is not timed."""
        self.X, y = classification_data(n_samples, _N_FEATURES)
        self.tree = DecisionTreeClassifier(max_depth=max_depth).fit(self.X, y)

    def time_predict(self, n_samples: int, max_depth: int) -> None:
        """Predict the labels of the training samples."""
        self.tree.predict(self.X)

    def time_predict_proba(self, n_samples: int, max_depth: int) -> None:
        """Predict the class probabilities of the training samples."""
        self.tree.predict_proba(self.X)
