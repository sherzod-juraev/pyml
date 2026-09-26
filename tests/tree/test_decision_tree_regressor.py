"""Tests for DecisionTreeRegressor."""

import numpy as np

from pyml.tree import DecisionTreeRegressor


class TestFit:
    def test_constant_target_becomes_a_single_leaf(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = np.full(20, 5.0)

        model = DecisionTreeRegressor(max_depth=5).fit(X, y)

        assert model.tree_.is_leaf

    def test_respects_max_depth(self, rng):
        X = rng.uniform(0, 10, size=(50, 2))
        y = rng.uniform(0, 10, size=50)

        model = DecisionTreeRegressor(max_depth=1).fit(X, y)

        assert model.tree_.is_leaf or (model.tree_.left.is_leaf and model.tree_.right.is_leaf)  # type: ignore[union-attr]

    def test_stops_early_when_samples_below_min_samples(self, rng):
        X = rng.uniform(0, 10, size=(3, 2))
        y = np.array([1.0, 2.0, 3.0])

        model = DecisionTreeRegressor(max_depth=10, min_samples=4).fit(X, y)

        assert model.tree_.is_leaf


class TestPredict:
    def test_fits_step_function_closely(self, rng):
        X = np.sort(rng.uniform(0, 10, size=(60, 1)), axis=0)
        y = np.where(X.ravel() < 5, 0.0, 10.0)

        model = DecisionTreeRegressor(max_depth=3).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_exact_match_returns_exact_value(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = rng.uniform(0, 10, size=20)

        model = DecisionTreeRegressor(max_depth=20, min_samples=1).fit(X, y)
        prediction = model.predict(X[:1])

        assert np.isclose(prediction[0], y[0])

    def test_leaf_predicts_mean_target(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([1.0, 2.0, 3.0])

        model = DecisionTreeRegressor(max_depth=0).fit(X, y)
        prediction = model.predict(np.array([[1.0]]))

        assert np.isclose(prediction[0], 2.0)
