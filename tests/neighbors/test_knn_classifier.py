"""Tests for KNNClassifier."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.neighbors import KNNClassifier


class TestFit:
    def test_rejects_zero_or_negative_n_neighbors(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = np.array([0, 1] * 5)

        with pytest.raises(InvalidParameterError):
            KNNClassifier(n_neighbors=0).fit(X, y)

    def test_rejects_n_neighbors_exceeding_sample_count(self, rng):
        X = rng.uniform(0, 10, size=(5, 2))
        y = np.array([0, 1, 0, 1, 0])

        with pytest.raises(InvalidParameterError):
            KNNClassifier(n_neighbors=10).fit(X, y)

    def test_sets_classes(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = np.array([0, 1, 2] * 3 + [0])

        model = KNNClassifier(n_neighbors=3).fit(X, y)

        assert np.array_equal(model.classes_, np.array([0, 1, 2]))


class TestPredict:
    def test_exact_match_returns_exact_label(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = np.array([0, 1] * 10)

        model = KNNClassifier(n_neighbors=1).fit(X, y)
        prediction = model.predict(X[:1])

        assert prediction[0] == y[0]

    def test_separates_linearly_separable_classes(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = KNNClassifier(n_neighbors=5).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_uniform_weighting_matches_manual_majority_vote(self, rng):
        X = rng.uniform(0, 10, size=(30, 2))
        y = rng.integers(0, 3, size=30)
        query = rng.uniform(0, 10, size=(1, 2))

        model = KNNClassifier(n_neighbors=5, weights="uniform").fit(X, y)
        prediction = model.predict(query)

        distances = np.linalg.norm(X - query, axis=1)
        nearest_idx = np.argsort(distances)[:5]
        values, counts = np.unique(y[nearest_idx], return_counts=True)
        expected = values[np.argmax(counts)]

        assert prediction[0] == expected

    def test_distance_weighting_favors_closer_neighbor(self):
        # one neighbor very close with label 1, several farther with label 0
        X = np.array([[0.0, 0.0], [10.0, 10.0], [10.1, 10.1], [10.2, 10.2]])
        y = np.array([1, 0, 0, 0])
        query = np.array([[0.1, 0.1]])

        model = KNNClassifier(n_neighbors=4, weights="distance").fit(X, y)
        prediction = model.predict(query)

        assert prediction[0] == 1
