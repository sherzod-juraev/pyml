"""Tests for BaseEstimator's parameter management and fit-state tracking."""

import numpy as np
import pytest

from pyml.core.base.estimator import BaseEstimator
from pyml.core.exceptions import InvalidParameterError, NotFittedError, ShapeMismatchError


class _DummyEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0, beta: int = 2) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta


class TestInit:
    def test_is_fitted_starts_false(self):
        estimator = _DummyEstimator()
        assert estimator.is_fitted_ is False


class TestCheckIsFitted:
    def test_raises_not_fitted_error_when_not_fitted(self):
        estimator = _DummyEstimator()
        with pytest.raises(NotFittedError):
            estimator._check_is_fitted()

    def test_does_not_raise_when_fitted(self):
        estimator = _DummyEstimator()
        estimator.is_fitted_ = True
        estimator._check_is_fitted()

    def test_not_fitted_error_message_mentions_class_name(self):
        estimator = _DummyEstimator()
        with pytest.raises(NotFittedError, match="_DummyEstimator"):
            estimator._check_is_fitted()


class TestGetParams:
    def test_returns_constructor_params_with_defaults(self):
        estimator = _DummyEstimator()
        assert estimator.get_params() == {"alpha": 1.0, "beta": 2}

    def test_returns_constructor_params_with_custom_values(self):
        estimator = _DummyEstimator(alpha=0.5, beta=10)
        assert estimator.get_params() == {"alpha": 0.5, "beta": 10}

    def test_reflects_attribute_mutation_after_construction(self):
        estimator = _DummyEstimator()
        estimator.alpha = 99.0
        assert estimator.get_params()["alpha"] == 99.0


class TestSetParams:
    def test_updates_existing_attribute(self):
        estimator = _DummyEstimator()
        estimator.set_params(alpha=7.0)
        assert estimator.alpha == 7.0

    def test_updates_multiple_attributes(self):
        estimator = _DummyEstimator()
        estimator.set_params(alpha=7.0, beta=42)
        assert estimator.alpha == 7.0
        assert estimator.beta == 42

    def test_returns_self_for_chaining(self):
        estimator = _DummyEstimator()
        result = estimator.set_params(alpha=7.0)
        assert result is estimator

    def test_no_arguments_returns_self_unchanged(self):
        estimator = _DummyEstimator()
        result = estimator.set_params()
        assert result is estimator
        assert estimator.get_params() == {"alpha": 1.0, "beta": 2}

    def test_raises_invalid_parameter_error_for_unknown_param(self):
        estimator = _DummyEstimator()
        with pytest.raises(InvalidParameterError):
            estimator.set_params(gamma=1.0)

    def test_invalid_parameter_error_lists_valid_params(self):
        estimator = _DummyEstimator()
        with pytest.raises(InvalidParameterError, match="alpha"):
            estimator.set_params(gamma=1.0)


class TestSetNFeatures:
    def test_records_number_of_features(self):
        estimator = _DummyEstimator()
        X = np.zeros((10, 4))
        estimator._set_n_features(X)
        assert estimator.n_features_in_ == 4

    def test_overwrites_previous_value_on_refit(self):
        estimator = _DummyEstimator()
        estimator._set_n_features(np.zeros((10, 4)))
        estimator._set_n_features(np.zeros((5, 7)))
        assert estimator.n_features_in_ == 7


class TestCheckNFeatures:
    def test_does_not_raise_when_feature_count_matches(self):
        estimator = _DummyEstimator()
        estimator._set_n_features(np.zeros((10, 4)))
        estimator._check_n_features(np.zeros((3, 4)))

    def test_raises_shape_mismatch_error_when_feature_count_differs(self):
        estimator = _DummyEstimator()
        estimator._set_n_features(np.zeros((10, 4)))
        with pytest.raises(ShapeMismatchError):
            estimator._check_n_features(np.zeros((3, 5)))

    def test_shape_mismatch_error_message_mentions_class_name(self):
        estimator = _DummyEstimator()
        estimator._set_n_features(np.zeros((10, 4)))
        with pytest.raises(ShapeMismatchError, match="_DummyEstimator"):
            estimator._check_n_features(np.zeros((3, 5)))
