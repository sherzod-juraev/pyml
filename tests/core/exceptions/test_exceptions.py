"""Tests for the pyml exception hierarchy.

Verifies inheritance relationships, that subclasses are catchable via
their parent classes, and that exception messages are preserved.
"""

import pytest

from pyml.core.exceptions import (
    FittingError,
    InvalidParameterError,
    NotFittedError,
    PymlError,
    ShapeMismatchError,
    ValidationError,
)


class TestPymlError:
    def test_pyml_error_is_exception(self):
        assert issubclass(PymlError, Exception)

    def test_pyml_error_message_preserved(self):
        with pytest.raises(PymlError, match="something went wrong"):
            raise PymlError("something went wrong")


class TestFittingError:
    def test_fitting_error_is_pyml_error(self):
        assert issubclass(FittingError, PymlError)

    def test_fitting_error_caught_by_pyml_error(self):
        with pytest.raises(PymlError):
            raise FittingError("fitting failed")

    def test_fitting_error_message_preserved(self):
        with pytest.raises(FittingError, match="fitting failed"):
            raise FittingError("fitting failed")


class TestNotFittedError:
    def test_not_fitted_error_is_fitting_error(self):
        assert issubclass(NotFittedError, FittingError)

    def test_not_fitted_error_is_pyml_error(self):
        assert issubclass(NotFittedError, PymlError)

    def test_not_fitted_error_caught_by_fitting_error(self):
        with pytest.raises(FittingError):
            raise NotFittedError("not fitted")

    def test_not_fitted_error_caught_by_pyml_error(self):
        with pytest.raises(PymlError):
            raise NotFittedError("not fitted")

    def test_not_fitted_error_message_preserved(self):
        with pytest.raises(NotFittedError, match="not fitted"):
            raise NotFittedError("not fitted")


class TestValidationError:
    def test_validation_error_is_pyml_error(self):
        assert issubclass(ValidationError, PymlError)

    def test_validation_error_caught_by_pyml_error(self):
        with pytest.raises(PymlError):
            raise ValidationError("invalid input")

    def test_validation_error_message_preserved(self):
        with pytest.raises(ValidationError, match="invalid input"):
            raise ValidationError("invalid input")


class TestInvalidParameterError:
    def test_invalid_parameter_error_is_validation_error(self):
        assert issubclass(InvalidParameterError, ValidationError)

    def test_invalid_parameter_error_is_pyml_error(self):
        assert issubclass(InvalidParameterError, PymlError)

    def test_invalid_parameter_error_caught_by_validation_error(self):
        with pytest.raises(ValidationError):
            raise InvalidParameterError("bad parameter")

    def test_invalid_parameter_error_caught_by_pyml_error(self):
        with pytest.raises(PymlError):
            raise InvalidParameterError("bad parameter")

    def test_invalid_parameter_error_message_preserved(self):
        with pytest.raises(InvalidParameterError, match="bad parameter"):
            raise InvalidParameterError("bad parameter")


class TestShapeMismatchError:
    def test_shape_mismatch_error_is_validation_error(self):
        assert issubclass(ShapeMismatchError, ValidationError)

    def test_shape_mismatch_error_is_pyml_error(self):
        assert issubclass(ShapeMismatchError, PymlError)

    def test_shape_mismatch_error_caught_by_validation_error(self):
        with pytest.raises(ValidationError):
            raise ShapeMismatchError("shape mismatch")

    def test_shape_mismatch_error_caught_by_pyml_error(self):
        with pytest.raises(PymlError):
            raise ShapeMismatchError("shape mismatch")

    def test_shape_mismatch_error_message_preserved(self):
        with pytest.raises(ShapeMismatchError, match="shape mismatch"):
            raise ShapeMismatchError("shape mismatch")
