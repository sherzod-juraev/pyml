"""Tests for KNNRegressor."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.neighbors import KNNRegressor


class TestFit:
    def test_rejects_zero_or_negative_n_neighbors(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        with pytest.raises(InvalidParameterError):
            KNNRegressor(n_neighbors=0).fit(X, y)

    def test_rejects_n_neighbors_exceeding_sample_count(self, rng):
        X = rng.uniform(0, 10, size=(5, 2))
        y = rng.uniform(0, 10, size=5)

        with pytest.raises(InvalidParameterError):
            KNNRegressor(n_neighbors=10).fit(X, y)


class TestPredict:
    def test_exact_match_returns_exact_target(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = rng.uniform(0, 10, size=20)

        model = KNNRegressor(n_neighbors=1).fit(X, y)
        prediction = model.predict(X[:1])

        assert prediction[0] == pytest.approx(y[0])

    def test_uniform_weighting_matches_manual_mean(self, rng):
        X = rng.uniform(0, 10, size=(20, 2))
        y = rng.uniform(0, 10, size=20)
        query = rng.uniform(0, 10, size=(1, 2))

        model = KNNRegressor(n_neighbors=3, weights="uniform").fit(X, y)
        prediction = model.predict(query)

        distances = np.linalg.norm(X - query, axis=1)
        nearest_idx = np.argsort(distances)[:3]
        expected = np.mean(y[nearest_idx])

        assert prediction[0] == pytest.approx(expected)

    def test_distance_weighting_favors_closer_neighbors(self, rng):
        # one neighbor very close with a distinct value, others far with
        # a different value cluster
        X = np.array([[0.0, 0.0], [10.0, 10.0], [10.1, 10.1], [10.2, 10.2]])
        y = np.array([100.0, 0.0, 0.0, 0.0])
        query = np.array([[0.1, 0.1]])

        model = KNNRegressor(n_neighbors=4, weights="distance").fit(X, y)
        uniform_model = KNNRegressor(n_neighbors=4, weights="uniform").fit(X, y)

        weighted_pred = model.predict(query)[0]
        uniform_pred = uniform_model.predict(query)[0]

        # distance weighting should pull the prediction much closer to the
        # nearby point's value (100.0) than uniform averaging would
        assert weighted_pred > uniform_pred

    def test_score_reasonable_on_noise_free_data(self, rng):
        X = np.sort(rng.uniform(0, 10, size=(100, 1)), axis=0)
        y = np.sin(X.ravel())

        model = KNNRegressor(n_neighbors=3).fit(X, y)

        assert model.score(X, y) > 0.95
