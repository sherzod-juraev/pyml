"""Tests for the Transformer abstract base class."""

import numpy as np
import pytest

from pyml.core.base import Transformer
from pyml.core.exceptions import InvalidParameterError, NotFittedError


class _DummyScaler(Transformer):
    """Subtracts the mean seen during fit; inverse adds it back."""

    def _fit(self, X):
        self._mean = np.mean(X, axis=0)

    def _transform(self, X):
        return X - self._mean

    def _inverse_transform(self, X):
        return X + self._mean


class _BadParamsScaler(_DummyScaler):
    """Rejects its hyper-parameters on every fit."""

    def _validate_params(self):
        raise InvalidParameterError("bad hyper-parameter")


@pytest.fixture
def transformer():
    return _DummyScaler()


class TestAbstractness:
    def test_cannot_instantiate_transformer_directly(self):
        with pytest.raises(TypeError):
            Transformer()  # type: ignore[abstract]


class TestFit:
    def test_sets_is_fitted_true(self, transformer, rng):
        X = rng.random((10, 3))
        transformer.fit(X)
        assert transformer.is_fitted_ is True

    def test_returns_self(self, transformer, rng):
        X = rng.random((10, 3))
        result = transformer.fit(X)
        assert result is transformer

    def test_calls_underlying_fit_logic(self, transformer):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        transformer.fit(X)
        assert np.allclose(transformer._mean, [2.0, 3.0])

    def test_rejects_invalid_x(self, transformer, rng):
        X = rng.random(10)  # 1D, invalid
        with pytest.raises(ValueError, match="2D"):
            transformer.fit(X)


class TestValidateParams:
    def test_rejects_invalid_hyperparameters(self, rng):
        X = rng.random((10, 3))

        with pytest.raises(InvalidParameterError):
            _BadParamsScaler().fit(X)

    def test_is_checked_before_the_data(self, rng):
        X = rng.random(10)  # 1D, invalid

        with pytest.raises(InvalidParameterError):
            _BadParamsScaler().fit(X)

    def test_failed_fit_leaves_the_estimator_untouched(self, rng):
        X = rng.random((10, 3))
        model = _BadParamsScaler()

        with pytest.raises(InvalidParameterError):
            model.fit(X)

        assert model.is_fitted_ is False
        assert not hasattr(model, "n_features_in_")
        assert not hasattr(model, "_mean")

    def test_runs_once_per_fit(self, transformer, rng, monkeypatch):
        X = rng.random((10, 3))
        calls = []
        monkeypatch.setattr(transformer, "_validate_params", lambda: calls.append(1))

        transformer.fit(X)

        assert calls == [1]

    def test_fit_transform_checks_the_hyperparameters_too(self, rng):
        X = rng.random((10, 3))

        with pytest.raises(InvalidParameterError):
            _BadParamsScaler().fit_transform(X)


class TestTransform:
    def test_raises_not_fitted_before_fit(self, transformer, rng):
        X = rng.random((5, 3))
        with pytest.raises(NotFittedError):
            transformer.transform(X)

    def test_returns_transformed_data_after_fit(self, transformer):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        transformer.fit(X_train)

        X_test = np.array([[5.0, 6.0]])
        result = transformer.transform(X_test)
        assert np.allclose(result, [[3.0, 3.0]])

    def test_rejects_invalid_x(self, transformer, rng):
        X_train = rng.random((10, 3))
        transformer.fit(X_train)

        X_test = rng.random(5)  # 1D, invalid
        with pytest.raises(ValueError, match="2D"):
            transformer.transform(X_test)


class TestInverseTransform:
    def test_raises_not_fitted_before_fit(self, transformer, rng):
        X = rng.random((5, 3))
        with pytest.raises(NotFittedError):
            transformer.inverse_transform(X)

    def test_reverses_transform(self, transformer, rng):
        X_train = rng.random((10, 3))
        transformer.fit(X_train)

        X_test = rng.random((5, 3))
        transformed = transformer.transform(X_test)
        restored = transformer.inverse_transform(transformed)
        assert np.allclose(restored, X_test)

    def test_rejects_invalid_x(self, transformer, rng):
        X_train = rng.random((10, 3))
        transformer.fit(X_train)

        X_test = rng.random(5)  # 1D, invalid
        with pytest.raises(ValueError, match="2D"):
            transformer.inverse_transform(X_test)


class TestFitTransform:
    def test_matches_fit_then_transform(self, transformer, rng):
        X = rng.random((10, 3))

        combined_result = _DummyScaler().fit_transform(X)
        separate_result = transformer.fit(X).transform(X)

        assert np.allclose(combined_result, separate_result)
