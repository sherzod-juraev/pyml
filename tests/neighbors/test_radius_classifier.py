"""Tests for RadiusNeighborsClassifier."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError, NoNeighborsError
from pyml.neighbors import RadiusNeighborsClassifier


class TestValidateParams:
    def test_error_message_names_the_parameter(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = np.array([0, 1] * 5)

        with pytest.raises(InvalidParameterError, match="radius"):
            RadiusNeighborsClassifier(radius=-1.0).fit(X, y)

    def test_unknown_weights_message_lists_the_options(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = np.array([0, 1] * 5)

        with pytest.raises(InvalidParameterError, match="uniform.*distance"):
            RadiusNeighborsClassifier(radius=1.0, weights="inverse").fit(X, y)  # type: ignore[arg-type]

    def test_unknown_metric_message_lists_the_supported_metrics(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = np.array([0, 1] * 5)

        with pytest.raises(InvalidParameterError, match="euclidean.*cityblock.*chebyshev"):
            RadiusNeighborsClassifier(radius=1.0, metric="euclidian").fit(X, y)  # type: ignore[arg-type]

    def test_accepts_boundary_values(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = np.array([0, 1] * 5)

        model = RadiusNeighborsClassifier(radius=1e-9).fit(X, y)

        assert model.is_fitted_ is True


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


class TestBlocking:
    @pytest.mark.parametrize("weights", ["uniform", "distance"])
    @pytest.mark.parametrize("rows_per_block", [1, 3, 7, 100])
    def test_blocked_predict_matches_single_block(self, rng, monkeypatch, weights, rows_per_block):
        X = rng.uniform(0, 10, size=(40, 2))
        y = rng.integers(0, 3, size=40)
        # some queries lie far outside the data and fall back to outlier_label
        queries = np.vstack([rng.uniform(0, 10, size=(14, 2)), np.full((3, 2), 100.0)])
        model = RadiusNeighborsClassifier(radius=2.0, weights=weights, outlier_label=-1).fit(X, y)
        expected = model.predict(queries)

        monkeypatch.setattr(
            "pyml.neighbors.radius_classifier._MAX_BLOCK_ELEMENTS", rows_per_block * X.shape[0]
        )

        assert np.array_equal(model.predict(queries), expected)

    def test_budget_smaller_than_one_row_still_predicts(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(40, 2))
        y = rng.integers(0, 3, size=40)
        queries = rng.uniform(0, 10, size=(6, 2))
        model = RadiusNeighborsClassifier(radius=3.0, outlier_label=-1).fit(X, y)
        expected = model.predict(queries)

        monkeypatch.setattr("pyml.neighbors.radius_classifier._MAX_BLOCK_ELEMENTS", 1)

        assert np.array_equal(model.predict(queries), expected)

    def test_no_neighbors_error_propagates_from_a_later_block(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(20, 2))
        y = rng.integers(0, 2, size=20)
        queries = np.vstack([X[:5], [[100.0, 100.0]]])
        model = RadiusNeighborsClassifier(radius=1.0).fit(X, y)

        monkeypatch.setattr("pyml.neighbors.radius_classifier._MAX_BLOCK_ELEMENTS", 2 * X.shape[0])

        with pytest.raises(NoNeighborsError):
            model.predict(queries)

    def test_blocked_predict_keeps_label_dtype(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(30, 2))
        y = rng.integers(0, 3, size=30).astype(np.int32)
        queries = rng.uniform(0, 10, size=(10, 2))
        model = RadiusNeighborsClassifier(radius=3.0, outlier_label=-1).fit(X, y)

        monkeypatch.setattr("pyml.neighbors.radius_classifier._MAX_BLOCK_ELEMENTS", 4 * X.shape[0])

        assert model.predict(queries).dtype == y.dtype

    def test_queries_are_processed_in_blocks_of_the_budgeted_size(self, rng, monkeypatch):
        X = rng.uniform(0, 10, size=(40, 2))
        y = rng.integers(0, 3, size=40)
        queries = rng.uniform(0, 10, size=(17, 2))
        model = RadiusNeighborsClassifier(radius=3.0, outlier_label=-1).fit(X, y)
        block_sizes = []
        original = RadiusNeighborsClassifier._predict_block

        def spy(self, X_block):
            block_sizes.append(X_block.shape[0])
            return original(self, X_block)

        monkeypatch.setattr(RadiusNeighborsClassifier, "_predict_block", spy)
        monkeypatch.setattr("pyml.neighbors.radius_classifier._MAX_BLOCK_ELEMENTS", 5 * X.shape[0])

        model.predict(queries)

        assert block_sizes == [5, 5, 5, 2]
