"""Hyper-parameter checks shared by all pyml estimators.

Small, reusable functions that estimators call from their
``_validate_params`` hook. Every check raises InvalidParameterError and
is written so that NaN is rejected as well, because any comparison
with NaN is false and would otherwise slip through silently.
"""

import math
from collections.abc import Collection

from .exceptions import InvalidParameterError

DISTANCE_METRICS: tuple[str, ...] = ("euclidean", "cityblock", "chebyshev")
"""Distance metrics supported by the neighbors models and DBSCAN."""


def check_positive(name: str, value: float, /, *, finite: bool = False) -> None:
    """Check that a parameter is strictly greater than zero.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : float
        Parameter value to check.
    finite : bool, optional
        If True, infinite values are rejected as well. Defaults to False.

    Raises
    ------
    InvalidParameterError
        If ``value`` is zero, negative or NaN, or infinite when
        ``finite`` is True.
    """
    if not value > 0 or (finite and not math.isfinite(value)):
        qualifier = "positive finite" if finite else "positive"
        raise InvalidParameterError(f"{name} must be a {qualifier} number, got {value!r}.")


def check_non_negative(name: str, value: float, /, *, finite: bool = False) -> None:
    """Check that a parameter is greater than or equal to zero.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : float
        Parameter value to check.
    finite : bool, optional
        If True, infinite values are rejected as well. Defaults to False.

    Raises
    ------
    InvalidParameterError
        If ``value`` is negative or NaN, or infinite when ``finite`` is
        True.
    """
    if not value >= 0 or (finite and not math.isfinite(value)):
        qualifier = "non-negative finite" if finite else "non-negative"
        raise InvalidParameterError(f"{name} must be a {qualifier} number, got {value!r}.")


def check_at_least(name: str, value: float, minimum: float, /) -> None:
    """Check that a parameter is greater than or equal to a lower bound.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : float
        Parameter value to check.
    minimum : float
        Smallest accepted value.

    Raises
    ------
    InvalidParameterError
        If ``value`` is smaller than ``minimum`` or NaN.
    """
    if not value >= minimum:
        raise InvalidParameterError(f"{name} must be at least {minimum}, got {value!r}.")


def check_in_range(name: str, value: float, low: float, high: float, /) -> None:
    """Check that a parameter lies in the closed interval [low, high].

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : float
        Parameter value to check.
    low : float
        Smallest accepted value.
    high : float
        Largest accepted value.

    Raises
    ------
    InvalidParameterError
        If ``value`` is outside [low, high] or NaN.
    """
    if not low <= value <= high:
        raise InvalidParameterError(f"{name} must be in [{low}, {high}], got {value!r}.")


def check_choice(name: str, value: object, choices: Collection[object], /) -> None:
    """Check that a parameter is one of the allowed values.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : object
        Parameter value to check.
    choices : Collection
        The allowed values.

    Raises
    ------
    InvalidParameterError
        If ``value`` is not in ``choices``.
    """
    if value not in choices:
        options = ", ".join(repr(choice) for choice in choices)
        raise InvalidParameterError(f"{name} must be one of {options}, got {value!r}.")
