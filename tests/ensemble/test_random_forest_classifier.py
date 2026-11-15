"""Tests for RandomForestClassifier."""

import numpy as np
import pytest

from pyml.core.dtypes import ClassificationTarget, FeatureMatrix
from pyml.core.exceptions import InvalidParameterError
from pyml.ensemble import RandomForestClassifier
from pyml.tree import DecisionTreeClassifier


def _noisy_data(
    rng: np.random.Generator, n_samples: int = 40, n_features: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(0, 10, size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    return X, y


def _record_fit_inputs(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    seen: list[np.ndarray] = []
    original = DecisionTreeClassifier.fit

    def spy(
        self: DecisionTreeClassifier, X: FeatureMatrix, y: ClassificationTarget, /
    ) -> DecisionTreeClassifier:
        seen.append(X.copy())
        return original(self, X, y)

    monkeypatch.setattr(DecisionTreeClassifier, "fit", spy)
    return seen


class TestFit:
    def test_grows_n_estimators_decision_trees(self, rng):
        X, y = _noisy_data(rng)

        model = RandomForestClassifier(max_depth=3, n_estimators=7, random_state=0).fit(X, y)

        assert len(model.estimators_) == 7
        assert all(isinstance(tree, DecisionTreeClassifier) for tree in model.estimators_)

    def test_classes_are_the_sorted_labels_seen_in_fit(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.array([9, 2, 5, 2, 9, 5])

        model = RandomForestClassifier(max_depth=2, n_estimators=3, random_state=0).fit(X, y)

        assert np.array_equal(model.classes_, [2, 5, 9])

    def test_each_tree_sees_a_bootstrap_sample_of_the_training_size(self, rng, monkeypatch):
        X, y = _noisy_data(rng)
        seen = _record_fit_inputs(monkeypatch)
        rows = set(map(tuple, X.tolist()))

        RandomForestClassifier(max_depth=2, n_estimators=6, random_state=0).fit(X, y)

        assert len(seen) == 6
        assert all(sample.shape == X.shape for sample in seen)
        assert all(set(map(tuple, sample.tolist())) <= rows for sample in seen)
        assert any(len(np.unique(sample, axis=0)) < len(X) for sample in seen)

    def test_without_bootstrap_each_tree_sees_the_whole_training_set(self, rng, monkeypatch):
        X, y = _noisy_data(rng)
        seen = _record_fit_inputs(monkeypatch)

        RandomForestClassifier(max_depth=2, n_estimators=4, bootstrap=False, random_state=0).fit(
            X, y
        )

        assert len(seen) == 4
        assert all(np.array_equal(sample, X) for sample in seen)

    def test_every_tree_gets_its_own_seed(self, rng):
        X, y = _noisy_data(rng)

        model = RandomForestClassifier(max_depth=2, n_estimators=10, random_state=0).fit(X, y)
        seeds = [tree.random_state for tree in model.estimators_]

        assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)
        assert len(set(seeds)) == 10

    @pytest.mark.parametrize(
        ("max_features", "n_features", "expected"),
        [
            ("sqrt", 1, 1),
            ("sqrt", 3, 1),
            ("sqrt", 9, 3),
            ("sqrt", 16, 4),
            (None, 5, None),
            (2, 5, 2),
        ],
    )
    def test_trees_get_the_resolved_max_features(self, rng, max_features, n_features, expected):
        X, y = _noisy_data(rng, n_samples=30, n_features=n_features)

        model = RandomForestClassifier(
            max_depth=2, n_estimators=2, max_features=max_features, random_state=0
        ).fit(X, y)

        assert all(tree.max_features == expected for tree in model.estimators_)

    def test_default_max_features_is_the_square_root(self, rng):
        X, y = _noisy_data(rng, n_samples=30, n_features=9)

        model = RandomForestClassifier(max_depth=2, n_estimators=2, random_state=0).fit(X, y)

        assert all(tree.max_features == 3 for tree in model.estimators_)

    def test_trees_get_the_tree_parameters_of_the_forest(self, rng):
        X, y = _noisy_data(rng)

        model = RandomForestClassifier(
            max_depth=4, n_estimators=2, min_samples=5, min_impurity_decrease=0.01, random_state=0
        ).fit(X, y)

        for tree in model.estimators_:
            assert tree.max_depth == 4
            assert tree.min_samples == 5
            assert tree.min_impurity_decrease == 0.01

    def test_max_features_above_the_feature_count_is_rejected(self, two_blobs):
        X, y = two_blobs(n_per_class=10)

        with pytest.raises(InvalidParameterError, match="max_features"):
            RandomForestClassifier(max_depth=3, n_estimators=2, max_features=X.shape[1] + 1).fit(
                X, y
            )

    def test_rejected_max_features_keeps_the_fitted_forest(self, rng):
        X = rng.uniform(0, 10, size=(30, 3))
        model = RandomForestClassifier(max_depth=3, n_estimators=3, random_state=0)
        model.fit(X, np.array([0, 1] * 15))
        trees = model.estimators_
        model.set_params(max_features=2)

        with pytest.raises(InvalidParameterError, match="max_features"):
            model.fit(X[:, :1], np.array([5, 6, 7] * 10))

        assert np.array_equal(model.classes_, [0, 1])
        assert model.estimators_ is trees


class TestReproducibility:
    def test_same_random_state_gives_the_same_forest(self, rng):
        X, y = _noisy_data(rng)

        first = RandomForestClassifier(max_depth=3, n_estimators=6, random_state=7).fit(X, y)
        second = RandomForestClassifier(max_depth=3, n_estimators=6, random_state=7).fit(X, y)

        assert np.array_equal(first.predict_proba(X), second.predict_proba(X))
        assert [t.random_state for t in first.estimators_] == [
            t.random_state for t in second.estimators_
        ]

    def test_different_random_states_give_different_forests(self, rng):
        X, y = _noisy_data(rng)

        first = RandomForestClassifier(max_depth=3, n_estimators=6, random_state=0).fit(X, y)
        second = RandomForestClassifier(max_depth=3, n_estimators=6, random_state=1).fit(X, y)

        assert not np.array_equal(first.predict_proba(X), second.predict_proba(X))


class TestPredictProba:
    def test_rows_sum_to_one_with_one_column_per_class(self, rng):
        X, _ = _noisy_data(rng)
        y = rng.integers(0, 3, size=len(X))
        model = RandomForestClassifier(max_depth=3, n_estimators=5, random_state=0).fit(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert proba.min() >= 0.0

    def test_single_tree_without_randomness_matches_a_decision_tree(self, rng):
        X, y = _noisy_data(rng)
        forest = RandomForestClassifier(
            max_depth=4, n_estimators=1, bootstrap=False, max_features=None
        ).fit(X, y)
        tree = DecisionTreeClassifier(max_depth=4).fit(X, y)

        assert np.array_equal(forest.predict_proba(X), tree.predict_proba(X))

    def test_is_the_average_of_the_probabilities_of_the_trees(self, rng):
        X, y = _noisy_data(rng)
        model = RandomForestClassifier(
            max_depth=3, n_estimators=6, bootstrap=False, max_features=1, random_state=0
        ).fit(X, y)

        expected = np.mean([tree.predict_proba(X) for tree in model.estimators_], axis=0)

        assert np.allclose(model.predict_proba(X), expected)

    def test_columns_follow_the_classes_when_labels_have_gaps(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.array([2, 5, 5, 9, 9, 9])
        model = RandomForestClassifier(
            max_depth=0, n_estimators=2, bootstrap=False, max_features=None
        ).fit(X, y)

        proba = model.predict_proba(X[:1])

        assert np.array_equal(model.classes_, [2, 5, 9])
        assert np.allclose(proba, [[1 / 6, 2 / 6, 3 / 6]])

    def test_a_tree_that_misses_a_class_adds_nothing_to_its_column(self, rng):
        X = rng.uniform(0, 10, size=(40, 3))
        y = np.where(X[:, 0] > 5, 9, 0)
        y[0] = 3
        model = RandomForestClassifier(max_depth=3, n_estimators=20, random_state=0).fit(X, y)
        n_know = sum(3 in tree.classes_ for tree in model.estimators_)

        expected = np.zeros((len(X), 3))
        for tree in model.estimators_:
            tree_proba = tree.predict_proba(X)
            for column, label in enumerate(tree.classes_):
                expected[:, list(model.classes_).index(label)] += tree_proba[:, column]
        expected /= 20

        assert 0 < n_know < 20
        assert np.array_equal(model.classes_, [0, 3, 9])
        assert np.allclose(model.predict_proba(X), expected)


class TestPredict:
    def test_separates_linearly_separable_classes(self, two_blobs):
        X, y = two_blobs(n_per_class=20)

        model = RandomForestClassifier(max_depth=3, n_estimators=8, random_state=0).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_returns_the_labels_not_their_indices(self, rng):
        X = rng.uniform(0, 10, size=(40, 3))
        y = np.where(X[:, 0] > 6, 9, np.where(X[:, 0] < 3, 3, 7))
        model = RandomForestClassifier(max_depth=3, n_estimators=6, random_state=0).fit(X, y)

        prediction = model.predict(X)

        assert set(prediction.tolist()) <= {3, 7, 9}
        assert np.array_equal(prediction, model.classes_[model.predict_proba(X).argmax(axis=1)])

    def test_single_tree_without_randomness_matches_a_decision_tree(self, rng):
        X, y = _noisy_data(rng)
        forest = RandomForestClassifier(
            max_depth=4, n_estimators=1, bootstrap=False, max_features=None
        ).fit(X, y)
        tree = DecisionTreeClassifier(max_depth=4).fit(X, y)

        assert np.array_equal(forest.predict(X), tree.predict(X))

    def test_a_tie_gives_the_smallest_label(self):
        X = np.zeros((4, 1))
        y = np.array([5, 5, 2, 2])
        model = RandomForestClassifier(
            max_depth=0, n_estimators=1, bootstrap=False, max_features=None
        ).fit(X, y)

        prediction = model.predict(X[:1])

        assert prediction[0] == 2


class TestValidateParams:
    @pytest.mark.parametrize(
        ("params", "name"),
        [
            ({"n_estimators": 0}, "n_estimators"),
            ({"bootstrap": 1}, "bootstrap"),
            ({"max_features": "log2"}, "max_features"),
            ({"max_depth": -1}, "max_depth"),
            ({"random_state": -1}, "random_state"),
        ],
    )
    def test_message_names_the_parameter(self, rng, params, name):
        X, y = _noisy_data(rng, n_samples=20)

        with pytest.raises(InvalidParameterError, match=name):
            RandomForestClassifier(**{"max_depth": 3, **params}).fit(X, y)

    @pytest.mark.parametrize("n_estimators", [1, np.int64(3)])
    def test_accepts_a_single_tree_and_numpy_integers(self, rng, n_estimators):
        X, y = _noisy_data(rng, n_samples=20)

        model = RandomForestClassifier(max_depth=2, n_estimators=n_estimators).fit(X, y)

        assert len(model.estimators_) == n_estimators

    @pytest.mark.parametrize("bootstrap", [True, False, np.True_, np.False_])
    def test_accepts_python_and_numpy_booleans(self, rng, bootstrap):
        X, y = _noisy_data(rng, n_samples=20)

        model = RandomForestClassifier(max_depth=2, n_estimators=2, bootstrap=bootstrap).fit(X, y)

        assert len(model.estimators_) == 2

    @pytest.mark.parametrize("max_features", ["sqrt", None, 1, 3, np.int64(2)])
    def test_accepts_every_form_of_max_features(self, rng, max_features):
        X, y = _noisy_data(rng, n_samples=20)

        model = RandomForestClassifier(max_depth=2, n_estimators=2, max_features=max_features).fit(
            X, y
        )

        assert len(model.estimators_) == 2
