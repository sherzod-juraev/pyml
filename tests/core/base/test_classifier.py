"""Tests for the Classifier abstract base class."""

import numpy as np
import pytest

from pyml.core.base import Classifier
from pyml.core.exceptions import NotFittedError, ShapeMismatchError
from pyml.metrics import accuracy_score


class _DummyClassifier(Classifier):
    """Always predicts the most frequent label seen during fit."""

    def _fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self._majority_label = int(values[np.argmax(counts)])

    def _predict(self, X):
        return np.full(X.shape[0], self._majority_label, dtype=np.int64)


@pytest.fixture
def classifier():
    return _DummyClassifier()


class TestAbstractness:
    def test_cannot_instantiate_classifier_directly(self):
        with pytest.raises(TypeError):
            Classifier()  # type: ignore[abstract]


class TestFit:
    def test_sets_is_fitted_true(self, classifier, rng):
        X = rng.random((10, 3))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        classifier.fit(X, y)
        assert classifier.is_fitted_ is True

    def test_returns_self(self, classifier, rng):
        X = rng.random((10, 3))
        y = np.zeros(10, dtype=np.int64)
        result = classifier.fit(X, y)
        assert result is classifier

    def test_calls_underlying_fit_logic(self, classifier, rng):
        X = rng.random((5, 3))
        y = np.array([0, 1, 1, 1, 0], dtype=np.int64)
        classifier.fit(X, y)
        assert classifier._majority_label == 1

    def test_rejects_invalid_x(self, classifier, rng):
        X = rng.random(10)  # 1D, invalid
        y = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="2D"):
            classifier.fit(X, y)

    def test_rejects_invalid_y(self, classifier, rng):
        X = rng.random((10, 3))
        y = rng.random(10)  # floating, invalid for labels
        with pytest.raises(TypeError, match="integers"):
            classifier.fit(X, y)

    def test_rejects_mismatched_samples(self, classifier, rng):
        X = rng.random((10, 3))
        y = np.zeros(5, dtype=np.int64)
        with pytest.raises(ShapeMismatchError):
            classifier.fit(X, y)


class TestPredict:
    def test_raises_not_fitted_before_fit(self, classifier, rng):
        X = rng.random((5, 3))
        with pytest.raises(NotFittedError):
            classifier.predict(X)

    def test_returns_predictions_after_fit(self, classifier, rng):
        X_train = rng.random((10, 3))
        y_train = np.ones(10, dtype=np.int64)
        classifier.fit(X_train, y_train)

        X_test = rng.random((5, 3))
        predictions = classifier.predict(X_test)
        assert predictions.shape == (5,)
        assert np.all(predictions == 1)

    def test_rejects_invalid_x(self, classifier, rng):
        X_train = rng.random((10, 3))
        y_train = np.zeros(10, dtype=np.int64)
        classifier.fit(X_train, y_train)

        X_test = rng.random(5)  # 1D, invalid
        with pytest.raises(ValueError, match="2D"):
            classifier.predict(X_test)


class TestScore:
    def test_matches_accuracy_score(self, classifier, rng):
        X_train = rng.random((10, 3))
        y_train = np.array([0, 1] * 5, dtype=np.int64)
        classifier.fit(X_train, y_train)

        X_test = rng.random((5, 3))
        y_test = np.array([0, 1, 0, 1, 1], dtype=np.int64)
        expected = accuracy_score(y_test, classifier.predict(X_test))
        assert classifier.score(X_test, y_test) == pytest.approx(expected)
