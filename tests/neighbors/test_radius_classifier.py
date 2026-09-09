"""Tests for RadiusNeighborsClassifier."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError, NoNeighborsError
from pyml.neighbors import RadiusNeighborsClassifier


class TestFit:
    def test_rejects_zero_or_negative_radius(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = np.array([0, 1] * 5)

        with pytest.raises(InvalidParameterError):
            RadiusNeighborsClassifier(radius=0.0).fit(X, y)


class TestPredict:
    def test_exact_match_returns_exact_label(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = np.array([0, 1] * 10)

        model = RadiusNeighborsClassifier(radius=1.0).fit(X, y)
        prediction = model.predict(X[:1])

        assert prediction[0] == y[0]

    def test_separates_linearly_separable_classes(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = RadiusNeighborsClassifier(radius=2.0).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_raises_when_no_neighbors_and_no_outlier_label(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        query = np.array([[100.0, 100.0]])

        model = RadiusNeighborsClassifier(radius=1.0).fit(X, y)

        with pytest.raises(NoNeighborsError):
            model.predict(query)

    def test_returns_outlier_label_when_configured(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        query = np.array([[100.0, 100.0]])

        model = RadiusNeighborsClassifier(radius=1.0, outlier_label=-1).fit(X, y)
        prediction = model.predict(query)

        assert prediction[0] == -1

    def test_distance_weighting_favors_closer_neighbor(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [1.1, 1.1]])
        y = np.array([1, 0, 0])
        query = np.array([[0.1, 0.1]])

        model = RadiusNeighborsClassifier(radius=5.0, weights="distance").fit(X, y)
        prediction = model.predict(query)

        assert prediction[0] == 1
