"""Benchmarks of the decision tree: the time of ``fit`` and of ``predict``."""

from typing import ClassVar

import numpy as np

from pyml.tree import DecisionTreeRegressor

_N_FEATURES = 6


class DecisionTreeRegressorBench:
    """Time of ``DecisionTreeRegressor.fit`` and ``predict`` for a growing number of samples."""

    params: ClassVar[list[list[int]]] = [[500, 4000]]
    param_names: ClassVar[list[str]] = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        """Create a non-linear data set and fit a tree for ``predict``. This is not timed."""
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(n_samples, _N_FEATURES))
        self.y = self.X[:, 0] ** 2 + self.X[:, 1] + rng.normal(0.0, 0.1, size=n_samples)
        self.tree = DecisionTreeRegressor(max_depth=6).fit(self.X, self.y)

    def time_fit(self, n_samples: int) -> None:
        """Fit a tree of depth 6."""
        DecisionTreeRegressor(max_depth=6).fit(self.X, self.y)

    def time_predict(self, n_samples: int) -> None:
        """Predict the training samples with the fitted tree."""
        self.tree.predict(self.X)
