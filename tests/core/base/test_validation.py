"""Tests for DataValidatorMixin's input-validation checks."""

import numpy as np
import pytest

from pyml.core.base.validation import DataValidatorMixin
from pyml.core.exceptions import ShapeMismatchError


class _Validator(DataValidatorMixin):
    pass


@pytest.fixture
def validator():
    return _Validator()


class TestValidateX:
    def test_accepts_valid_2d_float_array(self, validator, rng):
        X = rng.random((10, 3))
        validator._validate_X(X)

    def test_rejects_1d_array(self, validator, rng):
        X = rng.random(10)
        with pytest.raises(ValueError, match="2D"):
            validator._validate_X(X)

    def test_rejects_3d_array(self, validator, rng):
        X = rng.random((10, 3, 2))
        with pytest.raises(ValueError, match="2D"):
            validator._validate_X(X)

    def test_rejects_integer_dtype(self, validator):
        X = np.array([[1, 2], [3, 4]], dtype=np.int64)
        with pytest.raises(TypeError, match="floating-point"):
            validator._validate_X(X)

    def test_rejects_nan(self, validator, rng):
        X = rng.random((5, 2))
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validator._validate_X(X)

    def test_rejects_infinity(self, validator, rng):
        X = rng.random((5, 2))
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinity"):
            validator._validate_X(X)


class TestValidateYNumeric:
    def test_accepts_valid_1d_float_array(self, validator, rng):
        y = rng.random(10)
        validator._validate_y_numeric(y)

    def test_rejects_2d_array(self, validator, rng):
        y = rng.random((10, 1))
        with pytest.raises(ValueError, match="1D"):
            validator._validate_y_numeric(y)

    def test_rejects_integer_dtype(self, validator):
        y = np.array([1, 2, 3], dtype=np.int64)
        with pytest.raises(TypeError, match="floating-point"):
            validator._validate_y_numeric(y)

    def test_rejects_nan(self, validator, rng):
        y = rng.random(5)
        y[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validator._validate_y_numeric(y)

    def test_rejects_infinity(self, validator, rng):
        y = rng.random(5)
        y[0] = np.inf
        with pytest.raises(ValueError, match="infinity"):
            validator._validate_y_numeric(y)


class TestValidateYLabel:
    def test_accepts_valid_1d_integer_array(self, validator):
        y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        validator._validate_y_label(y)

    def test_rejects_2d_array(self, validator):
        y = np.array([[0, 1], [1, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="1D"):
            validator._validate_y_label(y)

    def test_rejects_float_dtype(self, validator, rng):
        y = rng.random(5)
        with pytest.raises(TypeError, match="integers"):
            validator._validate_y_label(y)


class TestValidateXYSamples:
    def test_accepts_matching_sample_counts(self, validator, rng):
        X = rng.random((10, 3))
        y = rng.random(10)
        validator._validate_X_y_samples(X, y)

    def test_rejects_mismatched_sample_counts(self, validator, rng):
        X = rng.random((10, 3))
        y = rng.random(5)
        with pytest.raises(ShapeMismatchError, match=r"\[10, 5\]"):
            validator._validate_X_y_samples(X, y)
