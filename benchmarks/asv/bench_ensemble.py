"""Benchmarks of the random forests: the time of ``predict``.

A forest asks every one of its trees for the leaf of every sample, so its ``predict`` is the
cost of ``predict`` of a tree times ``n_estimators``. The sizes follow the examples (trees of
depth 8) but with fewer trees, so that ``setup`` stays short.
"""

from typing import ClassVar

from pyml.ensemble import RandomForestClassifier, RandomForestRegressor

from ._data import N_SMALL, classification_data, nonlinear_regression_data

_N_FEATURES = 6
_N_ESTIMATORS = 25
_MAX_DEPTH = 8


class RandomForestRegressorPredict:
    """Time of ``RandomForestRegressor.predict``."""

    params: ClassVar[list[list[int]]] = [[N_SMALL]]
    param_names: ClassVar[list[str]] = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        """Create a non-linear data set and fit the forest. This is not timed."""
        self.X, y = nonlinear_regression_data(n_samples, _N_FEATURES)
        self.forest = RandomForestRegressor(
            max_depth=_MAX_DEPTH, n_estimators=_N_ESTIMATORS, random_state=0
        ).fit(self.X, y)

    def time_predict(self, n_samples: int) -> None:
        """Predict the training samples with the fitted forest."""
        self.forest.predict(self.X)


class RandomForestClassifierPredict:
    """Time of ``RandomForestClassifier.predict`` and ``predict_proba``."""

    params: ClassVar[list[list[int]]] = [[N_SMALL]]
    param_names: ClassVar[list[str]] = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        """Create two overlapping classes and fit the forest. This is not timed."""
        self.X, y = classification_data(n_samples, _N_FEATURES)
        self.forest = RandomForestClassifier(
            max_depth=_MAX_DEPTH, n_estimators=_N_ESTIMATORS, random_state=0
        ).fit(self.X, y)

    def time_predict(self, n_samples: int) -> None:
        """Predict the labels of the training samples."""
        self.forest.predict(self.X)

    def time_predict_proba(self, n_samples: int) -> None:
        """Predict the class probabilities of the training samples."""
        self.forest.predict_proba(self.X)
