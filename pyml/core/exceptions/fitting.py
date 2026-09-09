"""Exceptions related to the fit/predict lifecycle of an estimator.

This module defines errors raised when an estimator's fitted state
does not satisfy the requirements of the operation being performed,
such as calling ``predict`` before ``fit``.
"""

from .base import PymlError


class FittingError(PymlError):
    """Base class for fit/predict lifecycle errors.

    Subclasses indicate problems with the state of an estimator,
    such as being used before it has been fitted.
    """


class NotFittedError(FittingError):
    """Raised when a fitted-estimator method is called before fit.

    This typically occurs when calling ``predict`` on an estimator
    whose ``is_fitted_`` attribute is still ``False``.
    """


class NoNeighborsError(FittingError):
    """Raised when no neighbors are found within a given radius or threshold.

    Typically raised by radius-based estimators (e.g.
    RadiusNeighborsClassifier) when a query point has no training
    samples within the configured radius, and no fallback value
    (such as outlier_label) has been provided.
    """
