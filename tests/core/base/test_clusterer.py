"""Tests for the Clusterer and PredictableClusterer abstract base classes."""

import numpy as np
import pytest

from pyml.core.base import Clusterer, PredictableClusterer
from pyml.core.exceptions import NotFittedError


class _DummyClusterer(Clusterer):
    """Assigns every sample to cluster 0."""

    def _fit(self, X):
        self._labels = np.zeros(X.shape[0], dtype=np.int64)

    def _get_labels(self):
        return self._labels


class _DummyPredictableClusterer(PredictableClusterer):
    """Assigns every sample to cluster 0, including new samples."""

    def _fit(self, X):
        self._labels = np.zeros(X.shape[0], dtype=np.int64)

    def _get_labels(self):
        return self._labels

    def _predict(self, X):
        return np.zeros(X.shape[0], dtype=np.int64)


@pytest.fixture
def clusterer():
    return _DummyClusterer()


@pytest.fixture
def predictable_clusterer():
    return _DummyPredictableClusterer()


class TestAbstractness:
    def test_cannot_instantiate_clusterer_directly(self):
        with pytest.raises(TypeError):
            Clusterer()  # type: ignore[abstract]

    def test_cannot_instantiate_predictable_clusterer_directly(self):
        with pytest.raises(TypeError):
            PredictableClusterer()  # type: ignore[abstract]


class TestFit:
    def test_sets_is_fitted_true(self, clusterer, rng):
        X = rng.random((10, 3))
        clusterer.fit(X)
        assert clusterer.is_fitted_ is True

    def test_returns_self(self, clusterer, rng):
        X = rng.random((10, 3))
        result = clusterer.fit(X)
        assert result is clusterer

    def test_rejects_invalid_x(self, clusterer, rng):
        X = rng.random(10)  # 1D, invalid
        with pytest.raises(ValueError, match="2D"):
            clusterer.fit(X)


class TestFitPredict:
    def test_returns_labels_for_each_sample(self, clusterer, rng):
        X = rng.random((10, 3))
        labels = clusterer.fit_predict(X)
        assert labels.shape == (10,)
        assert np.all(labels == 0)

    def test_sets_is_fitted_true(self, clusterer, rng):
        X = rng.random((10, 3))
        clusterer.fit_predict(X)
        assert clusterer.is_fitted_ is True


class TestPredictableClustererPredict:
    def test_raises_not_fitted_before_fit(self, predictable_clusterer, rng):
        X = rng.random((5, 3))
        with pytest.raises(NotFittedError):
            predictable_clusterer.predict(X)

    def test_returns_labels_after_fit(self, predictable_clusterer, rng):
        X_train = rng.random((10, 3))
        predictable_clusterer.fit(X_train)

        X_test = rng.random((5, 3))
        labels = predictable_clusterer.predict(X_test)
        assert labels.shape == (5,)
        assert np.all(labels == 0)

    def test_rejects_invalid_x(self, predictable_clusterer, rng):
        X_train = rng.random((10, 3))
        predictable_clusterer.fit(X_train)

        X_test = rng.random(5)  # 1D, invalid
        with pytest.raises(ValueError, match="2D"):
            predictable_clusterer.predict(X_test)
