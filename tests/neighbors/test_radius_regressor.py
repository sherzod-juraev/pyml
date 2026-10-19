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


class TestBlocking:
    @pytest.mark.parametrize("weights", ["uniform", "distance"])
    @pytest.mark.parametrize("rows_per_block", [1, 3, 7, 100])
    def test_blocked_predict_matches_single_block(self, rng, monkeypatch, weights, rows_per_block):
        X = rng.uniform(0, 10, size=(40, 2))
        y = rng.uniform(0, 10, size=40)
        # some queries lie far outside the data and fall back to outlier_label
        queries = np.vstack([rng.uniform(0, 10, size=(14, 2)), np.full((3, 2), 100.0)])
        model = RadiusNeighborsRegressor(radius=2.0, weights=weights, outlier_label=-1.0).fit(X, y)
        expected = model.predict(queries)

        monkeypatch.setattr(
            "pyml.neighbors.radius_regressor._MAX_BLOCK_ELEMENTS", rows_per_block * X.shape[0]
        )

        assert np.allclose(model.predict(queries), expected)

    def test_budget_smaller_than_one_row_still_predicts(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(40, 2))
        y = rng.uniform(0, 10, size=40)
        queries = rng.uniform(0, 10, size=(6, 2))
        model = RadiusNeighborsRegressor(radius=3.0, outlier_label=-1.0).fit(X, y)
        expected = model.predict(queries)

        monkeypatch.setattr("pyml.neighbors.radius_regressor._MAX_BLOCK_ELEMENTS", 1)

        assert np.allclose(model.predict(queries), expected)

    def test_no_neighbors_error_propagates_from_a_later_block(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(20, 2))
        y = rng.uniform(0, 10, size=20)
        queries = np.vstack([X[:5], [[100.0, 100.0]]])
        model = RadiusNeighborsRegressor(radius=1.0).fit(X, y)

        monkeypatch.setattr("pyml.neighbors.radius_regressor._MAX_BLOCK_ELEMENTS", 2 * X.shape[0])

        with pytest.raises(NoNeighborsError):
            model.predict(queries)

    def test_blocked_predict_keeps_target_dtype(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(30, 2))
        y = rng.uniform(0, 10, size=30).astype(np.float32)
        queries = rng.uniform(0, 10, size=(10, 2))
        model = RadiusNeighborsRegressor(radius=3.0, outlier_label=-1.0).fit(X, y)
        expected_dtype = model.predict(queries).dtype

        monkeypatch.setattr("pyml.neighbors.radius_regressor._MAX_BLOCK_ELEMENTS", 4 * X.shape[0])

        assert model.predict(queries).dtype == expected_dtype

    def test_queries_are_processed_in_blocks_of_the_budgeted_size(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(40, 2))
        y = rng.uniform(0, 10, size=40)
        queries = rng.uniform(0, 10, size=(17, 2))
        model = RadiusNeighborsRegressor(radius=3.0, outlier_label=-1.0).fit(X, y)
        block_sizes = []
        original = RadiusNeighborsRegressor._predict_block

        def spy(self, X_block):
            block_sizes.append(X_block.shape[0])
            return original(self, X_block)

        monkeypatch.setattr(RadiusNeighborsRegressor, "_predict_block", spy)
        monkeypatch.setattr("pyml.neighbors.radius_regressor._MAX_BLOCK_ELEMENTS", 5 * X.shape[0])

        model.predict(queries)

        assert block_sizes == [5, 5, 5, 2]
