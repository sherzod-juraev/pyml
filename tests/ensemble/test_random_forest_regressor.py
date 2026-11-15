"""Tests for RandomForestRegressor."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.ensemble import RandomForestRegressor
from pyml.tree import DecisionTreeRegressor


def _noisy_data(
    rng: np.random.Generator, n_samples: int = 40, n_features: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(0, 10, size=(n_samples, n_features))
    y = 2.0 * X[:, 0] + rng.normal(size=n_samples)
    return X, y


class TestFit:
    def test_grows_n_estimators_regression_trees(self, rng):
        X, y = _noisy_data(rng)

        model = RandomForestRegressor(max_depth=3, n_estimators=7, random_state=0).fit(X, y)

        assert len(model.estimators_) == 7
        assert all(isinstance(tree, DecisionTreeRegressor) for tree in model.estimators_)

    def test_trees_get_the_tree_parameters_of_the_forest(self, rng):
        X, y = _noisy_data(rng)

        model = RandomForestRegressor(
            max_depth=4, n_estimators=2, min_samples=5, min_impurity_decrease=0.01, random_state=0
        ).fit(X, y)

        for tree in model.estimators_:
            assert tree.max_depth == 4
            assert tree.min_samples == 5
            assert tree.min_impurity_decrease == 0.01

    def test_trees_get_the_resolved_max_features(self, rng):
        X, y = _noisy_data(rng, n_samples=30, n_features=9)

        default = RandomForestRegressor(max_depth=2, n_estimators=2, random_state=0).fit(X, y)
        every = RandomForestRegressor(
            max_depth=2, n_estimators=2, max_features=None, random_state=0
        ).fit(X, y)

        assert all(tree.max_features == 3 for tree in default.estimators_)
        assert all(tree.max_features is None for tree in every.estimators_)

    def test_every_tree_gets_its_own_seed(self, rng):
        X, y = _noisy_data(rng)

        model = RandomForestRegressor(max_depth=2, n_estimators=10, random_state=0).fit(X, y)
        seeds = [tree.random_state for tree in model.estimators_]

        assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)
        assert len(set(seeds)) == 10

    def test_max_features_above_the_feature_count_is_rejected(self, rng):
        X, y = _noisy_data(rng, n_samples=20)

        with pytest.raises(InvalidParameterError, match="max_features"):
            RandomForestRegressor(max_depth=3, n_estimators=2, max_features=X.shape[1] + 1).fit(
                X, y
            )


class TestReproducibility:
    def test_same_random_state_gives_the_same_forest(self, rng):
        X, y = _noisy_data(rng)

        first = RandomForestRegressor(max_depth=3, n_estimators=6, random_state=7).fit(X, y)
        second = RandomForestRegressor(max_depth=3, n_estimators=6, random_state=7).fit(X, y)

        assert np.array_equal(first.predict(X), second.predict(X))

    def test_different_random_states_give_different_forests(self, rng):
        X, y = _noisy_data(rng)

        first = RandomForestRegressor(max_depth=3, n_estimators=6, random_state=0).fit(X, y)
        second = RandomForestRegressor(max_depth=3, n_estimators=6, random_state=1).fit(X, y)

        assert not np.array_equal(first.predict(X), second.predict(X))


class TestPredict:
    def test_is_the_mean_of_the_predictions_of_the_trees(self, rng):
        X, y = _noisy_data(rng)
        model = RandomForestRegressor(max_depth=3, n_estimators=6, random_state=0).fit(X, y)

        expected = np.mean([tree.predict(X) for tree in model.estimators_], axis=0)

        assert np.allclose(model.predict(X), expected)

    def test_single_tree_without_randomness_matches_a_decision_tree(self, rng):
        X, y = _noisy_data(rng)
        forest = RandomForestRegressor(
            max_depth=4, n_estimators=1, bootstrap=False, max_features=None
        ).fit(X, y)
        tree = DecisionTreeRegressor(max_depth=4).fit(X, y)

        assert np.array_equal(forest.predict(X), tree.predict(X))

    def test_gives_one_float_per_sample(self, rng):
        X, y = _noisy_data(rng)
        model = RandomForestRegressor(max_depth=3, n_estimators=4, random_state=0).fit(X, y)

        prediction = model.predict(X[:7])

        assert prediction.shape == (7,)
        assert prediction.dtype == np.float64

    def test_stays_within_the_range_of_the_targets(self, rng):
        X, y = _noisy_data(rng)
        model = RandomForestRegressor(max_depth=3, n_estimators=6, random_state=0).fit(X, y)

        prediction = model.predict(rng.uniform(0, 10, size=(30, 3)))

        assert prediction.min() >= y.min()
        assert prediction.max() <= y.max()

    def test_fits_a_linear_signal(self, linear_data):
        X, y = linear_data(n_samples=60)

        model = RandomForestRegressor(max_depth=5, n_estimators=10, random_state=0).fit(X, y)

        assert model.score(X, y) > 0.9
