"""Tests for DecisionTreeRegressor."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
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

    def test_constant_features_with_varying_target_become_a_leaf(self):
        X = np.ones((4, 2))
        y = np.array([1.0, 2.0, 3.0, 6.0])

        model = DecisionTreeRegressor(max_depth=5).fit(X, y)

        assert model.tree_.is_leaf
        assert model.predict(X[:1])[0] == pytest.approx(3.0)

    def test_repeated_feature_values_split_between_distinct_values(self):
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        y = np.array([1.0, 1.0, 5.0, 5.0])

        model = DecisionTreeRegressor(max_depth=3).fit(X, y)

        assert model.tree_.threshold == pytest.approx(0.5)
        assert np.allclose(model.predict(X), y)


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


class TestValidateParams:
    @pytest.mark.parametrize(
        ("name", "value"),
        [("max_depth", -1), ("min_samples", 0), ("min_impurity_decrease", -1.0)],
    )
    def test_error_message_names_the_parameter(self, linear_data, name, value):
        X, y = linear_data(n_samples=40)

        with pytest.raises(InvalidParameterError, match=name):
            DecisionTreeRegressor(**{"max_depth": 3, name: value}).fit(X, y)

    def test_zero_max_depth_gives_a_single_leaf(self, linear_data):
        X, y = linear_data(n_samples=40)

        model = DecisionTreeRegressor(max_depth=0).fit(X, y)

        assert model.tree_.is_leaf

    @pytest.mark.parametrize("params", [{"min_samples": 1}, {"min_impurity_decrease": 0.0}])
    def test_accepts_boundary_values(self, linear_data, params):
        X, y = linear_data(n_samples=40)

        model = DecisionTreeRegressor(**{"max_depth": 3, **params}).fit(X, y)

        assert model.is_fitted_ is True
