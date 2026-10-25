"""Tests for DecisionTreeClassifier."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.tree import DecisionTreeClassifier


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


class TestValidateParams:
    @pytest.mark.parametrize(
        ("name", "value"),
        [("max_depth", -1), ("min_samples", 0), ("min_impurity_decrease", -1.0)],
    )
    def test_error_message_names_the_parameter(self, two_blobs, name, value):
        X, y = two_blobs(n_per_class=20)

        with pytest.raises(InvalidParameterError, match=name):
            DecisionTreeClassifier(**{"max_depth": 3, name: value}).fit(X, y)

    def test_zero_max_depth_gives_a_single_leaf(self, two_blobs):
        X, y = two_blobs(n_per_class=20)

        model = DecisionTreeClassifier(max_depth=0).fit(X, y)

        assert model.tree_.is_leaf

    @pytest.mark.parametrize("params", [{"min_samples": 1}, {"min_impurity_decrease": 0.0}])
    def test_accepts_boundary_values(self, two_blobs, params):
        X, y = two_blobs(n_per_class=20)

        model = DecisionTreeClassifier(**{"max_depth": 3, **params}).fit(X, y)

        assert model.is_fitted_ is True
