"""Tests for DecisionTreeRegressor."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.tree import DecisionTreeRegressor
from pyml.tree._base import _Node


def _split_features(node: _Node | None) -> set[int | None]:
    if node is None or node.is_leaf:
        return set()
    return {node.feature_idx} | _split_features(node.left) | _split_features(node.right)


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

    def test_all_features_as_max_features_matches_the_default_tree(self, rng):
        X = rng.uniform(0, 10, size=(60, 4))
        y = rng.uniform(0, 10, size=60)

        default = DecisionTreeRegressor(max_depth=4).fit(X, y)
        all_features = DecisionTreeRegressor(max_depth=4, max_features=4, random_state=7).fit(X, y)

        assert np.array_equal(all_features.predict(X), default.predict(X))
        assert all_features.tree_.feature_idx == default.tree_.feature_idx
        assert all_features.tree_.threshold == default.tree_.threshold

    def test_same_random_state_gives_the_same_tree(self, rng):
        X = rng.uniform(0, 10, size=(60, 6))
        y = rng.uniform(0, 10, size=60)

        first = DecisionTreeRegressor(max_depth=4, max_features=2, random_state=3).fit(X, y)
        second = DecisionTreeRegressor(max_depth=4, max_features=2, random_state=3).fit(X, y)

        assert np.array_equal(first.predict(X), second.predict(X))

    def test_max_features_restricts_the_split_candidates(self, rng):
        X = rng.uniform(0, 10, size=(60, 4))
        y = np.where(X[:, 0] > 5, 10.0, 0.0)

        roots = {
            DecisionTreeRegressor(max_depth=1, max_features=1, random_state=seed)
            .fit(X, y)
            .tree_.feature_idx
            for seed in range(20)
        }
        full = DecisionTreeRegressor(max_depth=1).fit(X, y)

        assert full.tree_.feature_idx == 0
        assert roots - {0}

    def test_each_node_draws_its_own_candidates(self, rng):
        X = rng.uniform(0, 10, size=(200, 4))
        y = X.sum(axis=1)

        model = DecisionTreeRegressor(max_depth=4, max_features=1, random_state=0).fit(X, y)

        assert len(_split_features(model.tree_)) > 1

    def test_max_features_above_the_feature_count_is_rejected(self, linear_data):
        X, y = linear_data(n_samples=40)

        with pytest.raises(InvalidParameterError, match="max_features"):
            DecisionTreeRegressor(max_depth=3, max_features=X.shape[1] + 1).fit(X, y)

    def test_leaves_keep_no_class_counts(self, rng):
        X = rng.uniform(0, 10, size=(30, 2))
        y = rng.uniform(0, 10, size=30)

        model = DecisionTreeRegressor(max_depth=1).fit(X, y)

        assert model.tree_.left.is_leaf  # type: ignore[union-attr]
        assert model.tree_.left.counts is None  # type: ignore[union-attr]
        assert model.tree_.right.counts is None  # type: ignore[union-attr]


class TestBestSplit:
    @pytest.mark.parametrize("shift", [0.0, 1e6, 1e8], ids=["no-shift", "shift-1e6", "shift-1e8"])
    def test_weighted_child_variance_matches_the_variance_of_each_part(self, rng, shift):
        y = rng.normal(size=40) + shift
        model = DecisionTreeRegressor(max_depth=3)
        n = y.shape[0]
        n_left = np.arange(1, n)

        result = model._weighted_child_impurity(y, n_left / n, (n - n_left) / n)

        expected = [(i / n) * np.var(y[:i]) + ((n - i) / n) * np.var(y[i:]) for i in range(1, n)]
        assert np.allclose(result, expected, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("shift", [0.0, 1e6], ids=["no-shift", "shift-1e6"])
    def test_matches_the_exhaustive_search(self, rng, exhaustive_best_split, split_gain, shift):
        X = rng.normal(size=(60, 3))
        y = X[:, 0] * 2 + rng.normal(scale=0.5, size=60) + shift
        model = DecisionTreeRegressor(max_depth=3)

        result = model._best_split(X, y, rng)

        _, _, best_gain = exhaustive_best_split(model, X, y)
        assert result is not None
        feature, threshold, gain = result
        assert gain == pytest.approx(best_gain, abs=1e-9)
        assert split_gain(model, X, y, feature, threshold) == pytest.approx(best_gain, abs=1e-9)

    def test_matches_the_exhaustive_search_with_repeated_values(
        self, rng, exhaustive_best_split, split_gain
    ):
        X = np.round(rng.normal(size=(80, 3)) * 2) / 2
        y = X[:, 1] + rng.normal(scale=0.5, size=80)
        model = DecisionTreeRegressor(max_depth=3)

        result = model._best_split(X, y, rng)

        _, _, best_gain = exhaustive_best_split(model, X, y)
        assert result is not None
        feature, threshold, gain = result
        assert gain == pytest.approx(best_gain, abs=1e-9)
        assert split_gain(model, X, y, feature, threshold) == pytest.approx(best_gain, abs=1e-9)

    def test_integer_targets_give_the_same_split_as_float_targets(self, rng):
        X = rng.normal(size=(50, 3))
        y = rng.integers(0, 10, size=50)
        model = DecisionTreeRegressor(max_depth=3)

        with_integers = model._best_split(X, y, rng)
        with_floats = model._best_split(X, y.astype(np.float64), rng)

        assert with_integers == with_floats

    def test_equal_gains_go_to_the_first_threshold(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 1.0, 1.0, 0.0])
        model = DecisionTreeRegressor(max_depth=3)

        result = model._best_split(X, y, np.random.default_rng(0))

        assert result is not None
        assert result[1] == pytest.approx(0.5)

    def test_equal_gains_go_to_the_first_feature(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y = np.array([0.0, 0.0, 5.0, 5.0])
        model = DecisionTreeRegressor(max_depth=3)

        result = model._best_split(X, y, np.random.default_rng(0))

        assert result is not None
        assert result[0] == 0

    def test_a_constant_feature_is_never_chosen(self):
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = np.array([0.0, 0.0, 5.0, 5.0])
        model = DecisionTreeRegressor(max_depth=3)

        result = model._best_split(X, y, np.random.default_rng(0))

        assert result is not None
        assert result[0] == 1
        assert result[1] == pytest.approx(1.5)

    def test_returns_none_when_every_feature_is_constant(self):
        X = np.ones((4, 2))
        y = np.array([1.0, 2.0, 3.0, 4.0])
        model = DecisionTreeRegressor(max_depth=3)

        result = model._best_split(X, y, np.random.default_rng(0))

        assert result is None


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
        [
            ("max_depth", -1),
            ("max_depth", 2.5),
            ("min_samples", 0),
            ("min_impurity_decrease", -1.0),
            ("max_features", 0),
            ("random_state", -1),
        ],
    )
    def test_error_message_names_the_parameter(self, linear_data, name, value):
        X, y = linear_data(n_samples=40)

        with pytest.raises(InvalidParameterError, match=name):
            DecisionTreeRegressor(**{"max_depth": 3, name: value}).fit(X, y)

    def test_zero_max_depth_gives_a_single_leaf(self, linear_data):
        X, y = linear_data(n_samples=40)

        model = DecisionTreeRegressor(max_depth=0).fit(X, y)

        assert model.tree_.is_leaf

    @pytest.mark.parametrize(
        "params",
        [
            {"min_samples": 1},
            {"min_impurity_decrease": 0.0},
            {"max_features": 1},
            {"max_features": 2},
            {"random_state": 0},
            {"max_depth": np.int64(3)},
        ],
    )
    def test_accepts_boundary_values(self, linear_data, params):
        X, y = linear_data(n_samples=40)

        model = DecisionTreeRegressor(**{"max_depth": 3, **params}).fit(X, y)

        assert model.is_fitted_ is True
