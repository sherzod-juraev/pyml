"""Exceptions related to invalid input or configuration.

This module defines errors raised when data, parameters, or other
user-provided values do not satisfy the requirements of an estimator.
"""

from .base import PymlError


class ValidationError(PymlError):
    """Base class for invalid input or configuration errors.

    Subclasses indicate that a value provided by the user (data,
    parameters, etc.) does not satisfy the requirements of an
    estimator.
    """


class InvalidParameterError(ValidationError):
    """Raised when set_params receives an unknown parameter name.

    The name does not match any of the estimator's constructor
    arguments.
    """


class ShapeMismatchError(ValidationError):
    """Raised when X and y have a mismatched number of samples.

    This occurs when the number of rows in the feature matrix does not
    match the length of the target array.
    """
