"""Tests for KNNRegressor."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.neighbors import KNNRegressor


class TestFit:
    def test_rejects_n_neighbors_exceeding_sample_count(self, rng):
        X = rng.uniform(0, 10, size=(5, 2))
        y = rng.uniform(0, 10, size=5)

        with pytest.raises(InvalidParameterError):
            KNNRegressor(n_neighbors=10).fit(X, y)


class TestValidateParams:
    def test_error_message_names_the_parameter(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        with pytest.raises(InvalidParameterError, match="n_neighbors"):
            KNNRegressor(n_neighbors=0).fit(X, y)

    def test_unknown_weights_message_lists_the_options(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        with pytest.raises(InvalidParameterError, match="uniform.*distance"):
            KNNRegressor(weights="inverse").fit(X, y)  # type: ignore[arg-type]

    def test_unknown_metric_message_lists_the_supported_metrics(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        with pytest.raises(InvalidParameterError, match="euclidean.*cityblock.*chebyshev"):
            KNNRegressor(metric="euclidian").fit(X, y)  # type: ignore[arg-type]

    def test_accepts_boundary_values(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        model = KNNRegressor(n_neighbors=1).fit(X, y)

        assert model.is_fitted_ is True


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


class TestBlocking:
    @pytest.mark.parametrize("weights", ["uniform", "distance"])
    @pytest.mark.parametrize("rows_per_block", [1, 3, 7, 100])
    def test_blocked_predict_matches_single_block(self, rng, monkeypatch, weights, rows_per_block):
        X = rng.uniform(0, 10, size=(40, 3))
        y = rng.uniform(0, 10, size=40)
        queries = rng.uniform(0, 10, size=(17, 3))
        model = KNNRegressor(n_neighbors=5, weights=weights).fit(X, y)
        expected = model.predict(queries)

        monkeypatch.setattr(
            "pyml.neighbors.knn_regressor._MAX_BLOCK_ELEMENTS", rows_per_block * X.shape[0]
        )

        assert np.allclose(model.predict(queries), expected)

    def test_budget_smaller_than_one_row_still_predicts(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(40, 3))
        y = rng.uniform(0, 10, size=40)
        queries = rng.uniform(0, 10, size=(6, 3))
        model = KNNRegressor(n_neighbors=5).fit(X, y)
        expected = model.predict(queries)

        monkeypatch.setattr("pyml.neighbors.knn_regressor._MAX_BLOCK_ELEMENTS", 1)

        assert np.allclose(model.predict(queries), expected)

    @pytest.mark.parametrize("weights", ["uniform", "distance"])
    def test_blocked_predict_keeps_target_dtype(self, rng, monkeypatch, weights):
        X = rng.uniform(0, 10, size=(30, 2))
        y = rng.uniform(0, 10, size=30).astype(np.float32)
        queries = rng.uniform(0, 10, size=(10, 2))
        model = KNNRegressor(n_neighbors=3, weights=weights).fit(X, y)
        expected_dtype = model.predict(queries).dtype

        monkeypatch.setattr("pyml.neighbors.knn_regressor._MAX_BLOCK_ELEMENTS", 4 * X.shape[0])

        assert model.predict(queries).dtype == expected_dtype

    def test_queries_are_processed_in_blocks_of_the_budgeted_size(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(40, 3))
        y = rng.uniform(0, 10, size=40)
        queries = rng.uniform(0, 10, size=(17, 3))
        model = KNNRegressor(n_neighbors=5).fit(X, y)
        block_sizes = []
        original = KNNRegressor._predict_block

        def spy(self, X_block):
            block_sizes.append(X_block.shape[0])
            return original(self, X_block)

        monkeypatch.setattr(KNNRegressor, "_predict_block", spy)
        monkeypatch.setattr("pyml.neighbors.knn_regressor._MAX_BLOCK_ELEMENTS", 5 * X.shape[0])

        model.predict(queries)

        assert block_sizes == [5, 5, 5, 2]
