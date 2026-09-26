"""Tests for DecisionTreeClassifier."""

import numpy as np

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
