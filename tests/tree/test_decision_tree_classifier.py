"""Tests for DecisionTreeClassifier."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.tree import DecisionTreeClassifier
from pyml.tree._base import _Node


def _split_features(node: _Node | None) -> set[int | None]:
    if node is None or node.is_leaf:
        return set()
    return {node.feature_idx} | _split_features(node.left) | _split_features(node.right)


class TestFit:
    def test_pure_node_becomes_a_single_leaf(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = np.zeros(20, dtype=int)

        model = DecisionTreeClassifier(max_depth=5).fit(X, y)

        assert model.tree_.is_leaf

    def test_respects_max_depth(self, rng):
        X = rng.uniform(0, 10, size=(50, 2))
        y = rng.integers(0, 2, size=50)

        model = DecisionTreeClassifier(max_depth=1).fit(X, y)

        assert model.tree_.is_leaf or (model.tree_.left.is_leaf and model.tree_.right.is_leaf)  # type: ignore[union-attr]

    def test_stops_early_when_samples_below_min_samples(self, rng):
        X = rng.uniform(0, 10, size=(3, 2))
        y = np.array([0, 1, 0])

        model = DecisionTreeClassifier(max_depth=10, min_samples=4).fit(X, y)

        assert model.tree_.is_leaf

    def test_constant_features_with_mixed_labels_become_a_leaf(self):
        X = np.ones((6, 2))
        y = np.array([0, 1, 1, 1, 0, 1])

        model = DecisionTreeClassifier(max_depth=5).fit(X, y)

        assert model.tree_.is_leaf
        assert model.predict(X[:1])[0] == 1

    def test_gain_below_min_impurity_decrease_becomes_a_leaf(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=5, min_impurity_decrease=10.0).fit(X, y)

        assert model.tree_.is_leaf

    def test_repeated_feature_values_split_between_distinct_values(self):
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3).fit(X, y)

        assert model.tree_.threshold == pytest.approx(0.5)
        assert np.array_equal(model.predict(X), y)

    def test_all_features_as_max_features_matches_the_default_tree(self, rng):
        X = rng.uniform(0, 10, size=(60, 4))
        y = rng.integers(0, 2, size=60)

        default = DecisionTreeClassifier(max_depth=4).fit(X, y)
        all_features = DecisionTreeClassifier(max_depth=4, max_features=4, random_state=7).fit(X, y)

        assert np.array_equal(all_features.predict(X), default.predict(X))
        assert all_features.tree_.feature_idx == default.tree_.feature_idx
        assert all_features.tree_.threshold == default.tree_.threshold

    def test_same_random_state_gives_the_same_tree(self, rng):
        X = rng.uniform(0, 10, size=(60, 6))
        y = rng.integers(0, 2, size=60)

        first = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=3).fit(X, y)
        second = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=3).fit(X, y)

        assert np.array_equal(first.predict(X), second.predict(X))

    def test_max_features_restricts_the_split_candidates(self, rng):
        X = rng.uniform(0, 10, size=(60, 4))
        y = (X[:, 0] > 5).astype(int)

        roots = {
            DecisionTreeClassifier(max_depth=1, max_features=1, random_state=seed)
            .fit(X, y)
            .tree_.feature_idx
            for seed in range(20)
        }
        full = DecisionTreeClassifier(max_depth=1).fit(X, y)

        assert full.tree_.feature_idx == 0
        assert roots - {0}

    def test_each_node_draws_its_own_candidates(self, rng):
        X = rng.uniform(0, 10, size=(200, 4))
        y = (X.sum(axis=1) > 20).astype(int)

        model = DecisionTreeClassifier(max_depth=4, max_features=1, random_state=0).fit(X, y)

        assert len(_split_features(model.tree_)) > 1

    def test_max_features_above_the_feature_count_is_rejected(self, two_blobs):
        X, y = two_blobs(n_per_class=20)

        with pytest.raises(InvalidParameterError, match="max_features"):
            DecisionTreeClassifier(max_depth=3, max_features=X.shape[1] + 1).fit(X, y)

    def test_classes_are_the_sorted_labels_seen_in_fit(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.array([9, 2, 5, 2, 9, 5])

        model = DecisionTreeClassifier(max_depth=2).fit(X, y)

        assert np.array_equal(model.classes_, [2, 5, 9])

    def test_leaves_count_the_samples_of_each_class(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([0, 0, 1, 0, 1, 1, 1, 1])

        model = DecisionTreeClassifier(max_depth=1).fit(X, y)

        assert model.tree_.counts is None
        assert np.array_equal(model.tree_.left.counts, [3, 1])  # type: ignore[union-attr, arg-type]
        assert np.array_equal(model.tree_.right.counts, [0, 4])  # type: ignore[union-attr, arg-type]

    def test_rejected_max_features_keeps_the_fitted_classes(self, rng):
        X = rng.uniform(0, 10, size=(30, 3))
        model = DecisionTreeClassifier(max_depth=3, max_features=2).fit(X, np.array([0, 1] * 15))

        with pytest.raises(InvalidParameterError, match="max_features"):
            model.fit(X[:, :1], np.array([5, 6, 7] * 10))

        assert np.array_equal(model.classes_, [0, 1])


class TestPredict:
    def test_separates_linearly_separable_classes(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = DecisionTreeClassifier(max_depth=3).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_exact_match_returns_exact_label(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = rng.integers(0, 2, size=20)

        model = DecisionTreeClassifier(max_depth=20, min_samples=1).fit(X, y)
        prediction = model.predict(X[:1])

        assert prediction[0] == y[0]

    def test_leaf_predicts_majority_class(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 0, 1])

        model = DecisionTreeClassifier(max_depth=0).fit(X, y)
        prediction = model.predict(np.array([[0.5]]))

        assert prediction[0] == 0


class TestPredictProba:
    def test_single_leaf_gives_the_class_frequencies(self):
        X = np.arange(4, dtype=float).reshape(-1, 1)
        y = np.array([0, 0, 0, 1])

        proba = DecisionTreeClassifier(max_depth=0).fit(X, y).predict_proba(X)

        assert np.allclose(proba, [[0.75, 0.25]] * 4)

    def test_impure_leaf_gives_its_class_shares(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([0, 0, 1, 0, 1, 1, 1, 1])
        model = DecisionTreeClassifier(max_depth=1).fit(X, y)

        proba = model.predict_proba(np.array([[1.0], [6.0]]))

        assert np.allclose(proba, [[0.75, 0.25], [0.0, 1.0]])

    def test_pure_leaves_give_one_hot_rows(self, two_blobs):
        X, y = two_blobs(n_per_class=20)
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)

        proba = model.predict_proba(X)

        assert np.array_equal(proba, np.eye(2)[y])

    def test_columns_follow_the_classes_when_labels_have_gaps(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.array([2, 5, 5, 9, 9, 9])
        model = DecisionTreeClassifier(max_depth=0).fit(X, y)

        proba = model.predict_proba(X[:1])

        assert np.array_equal(model.classes_, [2, 5, 9])
        assert np.allclose(proba, [[1 / 6, 2 / 6, 3 / 6]])

    def test_pure_leaf_fills_the_column_of_its_class(self):
        X = np.arange(4, dtype=float).reshape(-1, 1)
        y = np.array([9, 9, 2, 2])
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)

        proba = model.predict_proba(np.array([[0.0], [3.0]]))

        assert np.array_equal(model.classes_, [2, 9])
        assert np.array_equal(proba, [[0.0, 1.0], [1.0, 0.0]])

    def test_tie_goes_to_the_smallest_label_as_in_predict(self):
        X = np.zeros((4, 1))
        y = np.array([7, 3, 7, 3])
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)

        proba = model.predict_proba(X[:1])

        assert np.allclose(proba, [[0.5, 0.5]])
        assert model.predict(X[:1])[0] == 3
        assert model.classes_[np.argmax(proba, axis=1)][0] == 3

    def test_rows_sum_to_one_and_the_largest_share_is_the_prediction(self, rng):
        X = rng.uniform(0, 10, size=(120, 3))
        y = rng.integers(0, 4, size=120)
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (120, 4)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.array_equal(model.classes_[np.argmax(proba, axis=1)], model.predict(X))

    def test_rows_follow_the_order_of_the_samples(self, rng):
        X = rng.uniform(0, 10, size=(40, 2))
        y = rng.integers(0, 3, size=40)
        model = DecisionTreeClassifier(max_depth=2).fit(X, y)

        assert np.array_equal(model.predict_proba(X[::-1]), model.predict_proba(X)[::-1])


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
    def test_error_message_names_the_parameter(self, two_blobs, name, value):
        X, y = two_blobs(n_per_class=20)

        with pytest.raises(InvalidParameterError, match=name):
            DecisionTreeClassifier(**{"max_depth": 3, name: value}).fit(X, y)

    def test_zero_max_depth_gives_a_single_leaf(self, two_blobs):
        X, y = two_blobs(n_per_class=20)

        model = DecisionTreeClassifier(max_depth=0).fit(X, y)

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
    def test_accepts_boundary_values(self, two_blobs, params):
        X, y = two_blobs(n_per_class=20)

        model = DecisionTreeClassifier(**{"max_depth": 3, **params}).fit(X, y)

        assert model.is_fitted_ is True
