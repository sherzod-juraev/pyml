"""Tests for RadiusNeighborsRegressor."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError, NoNeighborsError
from pyml.neighbors import RadiusNeighborsRegressor


class TestFit:
    def test_rejects_zero_or_negative_radius(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        with pytest.raises(InvalidParameterError):
            RadiusNeighborsRegressor(radius=0.0).fit(X, y)


class TestPredict:
    def test_exact_match_returns_exact_target(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        y = np.array([5.0, 50.0, 500.0])

        model = RadiusNeighborsRegressor(radius=1.0).fit(X, y)
        prediction = model.predict(X[:1])

        assert prediction[0] == pytest.approx(y[0])

    def test_uniform_weighting_matches_manual_mean(self):
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [10.0, 10.0]])
        y = np.array([1.0, 2.0, 3.0, 100.0])
        query = np.array([[0.0, 0.0]])

        model = RadiusNeighborsRegressor(radius=2.0, weights="uniform").fit(X, y)
        prediction = model.predict(query)

        # points at (0,0), (0.5,0.5), (1,1) are within radius 2.0 of (0,0);
        # (10,10) is not
        expected = np.mean([1.0, 2.0, 3.0])

        assert prediction[0] == pytest.approx(expected)

    def test_raises_when_no_neighbors_and_no_outlier_label(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([1.0, 2.0])
        query = np.array([[100.0, 100.0]])

        model = RadiusNeighborsRegressor(radius=1.0).fit(X, y)

        with pytest.raises(NoNeighborsError):
            model.predict(query)

    def test_returns_outlier_label_when_configured(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([1.0, 2.0])
        query = np.array([[100.0, 100.0]])

        model = RadiusNeighborsRegressor(radius=1.0, outlier_label=-1.0).fit(X, y)
        prediction = model.predict(query)

        assert prediction[0] == pytest.approx(-1.0)

    def test_distance_weighting_favors_closer_neighbor(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [1.1, 1.1]])
        y = np.array([100.0, 0.0, 0.0])
        query = np.array([[0.1, 0.1]])

        model = RadiusNeighborsRegressor(radius=5.0, weights="distance").fit(X, y)
        uniform_model = RadiusNeighborsRegressor(radius=5.0, weights="uniform").fit(X, y)

        weighted_pred = model.predict(query)[0]
        uniform_pred = uniform_model.predict(query)[0]

        assert weighted_pred > uniform_pred
