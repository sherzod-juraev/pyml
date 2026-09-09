"""Tests for the Regressor abstract base class."""

import numpy as np
import pytest

from pyml.core.base import Regressor
from pyml.core.exceptions import NotFittedError, ShapeMismatchError
from pyml.metrics import r2_score


class _DummyRegressor(Regressor):
    """Always predicts the mean of y seen during fit."""

    def _fit(self, X, y):
        self._mean = float(np.mean(y))

    def _predict(self, X):
        return np.full(X.shape[0], self._mean)


@pytest.fixture
def regressor():
    return _DummyRegressor()


class TestAbstractness:
    def test_cannot_instantiate_regressor_directly(self):
        with pytest.raises(TypeError):
            Regressor()  # type: ignore[abstract]


class TestFit:
    def test_sets_is_fitted_true(self, regressor, rng):
        X = rng.random((10, 3))
        y = rng.random(10)
        regressor.fit(X, y)
        assert regressor.is_fitted_ is True

    def test_returns_self(self, regressor, rng):
        X = rng.random((10, 3))
        y = rng.random(10)
        result = regressor.fit(X, y)
        assert result is regressor

    def test_calls_underlying_fit_logic(self, regressor, rng):
        X = rng.random((10, 3))
        y = np.full(10, 5.0)
        regressor.fit(X, y)
        assert regressor._mean == pytest.approx(5.0)

    def test_rejects_invalid_x(self, regressor, rng):
        X = rng.random(10)  # 1D, invalid
        y = rng.random(10)
        with pytest.raises(ValueError, match="2D"):
            regressor.fit(X, y)

    def test_rejects_invalid_y(self, regressor, rng):
        X = rng.random((10, 3))
        y = np.array([1, 2, 3], dtype=np.int64)  # wrong dtype
        with pytest.raises(TypeError, match="floating-point"):
            regressor.fit(X, y)

    def test_rejects_mismatched_samples(self, regressor, rng):
        X = rng.random((10, 3))
        y = rng.random(5)
        with pytest.raises(ShapeMismatchError):
            regressor.fit(X, y)


class TestPredict:
    def test_raises_not_fitted_before_fit(self, regressor, rng):
        X = rng.random((5, 3))
        with pytest.raises(NotFittedError):
            regressor.predict(X)

    def test_returns_predictions_after_fit(self, regressor, rng):
        X_train = rng.random((10, 3))
        y_train = np.full(10, 3.0)
        regressor.fit(X_train, y_train)

        X_test = rng.random((5, 3))
        predictions = regressor.predict(X_test)
        assert predictions.shape == (5,)
        assert np.allclose(predictions, 3.0)

    def test_rejects_invalid_x(self, regressor, rng):
        X_train = rng.random((10, 3))
        y_train = rng.random(10)
        regressor.fit(X_train, y_train)

        X_test = rng.random(5)  # 1D, invalid
        with pytest.raises(ValueError, match="2D"):
            regressor.predict(X_test)


class TestScore:
    def test_matches_r2_score(self, regressor, rng):
        X_train = rng.random((10, 3))
        y_train = rng.random(10)
        regressor.fit(X_train, y_train)

        X_test = rng.random((5, 3))
        y_test = rng.random(5)
        expected = r2_score(y_test, regressor.predict(X_test))
        assert regressor.score(X_test, y_test) == pytest.approx(expected)
