"""Benchmarks of the decision tree: the time of ``fit`` and of ``predict``."""

from typing import ClassVar

from pyml.tree import DecisionTreeRegressor

from ._data import N_SMALL, N_TREE_LARGE, nonlinear_regression_data

_N_FEATURES = 6


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
