"""Shared input-validation helpers for fit/predict pipelines.

Provides DataValidatorMixin, a set of reusable checks for feature
matrices and targets (shape, dtype, finiteness) used by Regressor,
Classifier, and other estimator base classes before delegating to
their model-specific _fit/_predict implementations.
"""

import numpy as np

from ..dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ..exceptions import ShapeMismatchError


class DataValidatorMixin:
    """Mixin providing reusable input-validation checks.

    Intended to be combined with BaseEstimator subclasses. Each method
    validates one aspect of the input (X's shape/dtype, y's shape/dtype,
    or X/y sample-count consistency) and raises on the first violation
    found.
    """

    def _validate_X(self, X: FeatureMatrix, /) -> None:  # noqa: N802
        """Validate that X is a well-formed 2D floating-point feature matrix.

        Parameters
        ----------
        X : FeatureMatrix
            Feature matrix to validate.

        Raises
        ------
        ValueError
            If X is not 2D, or contains NaN/infinite/out-of-range values.
        TypeError
            If X's dtype is not floating-point.
        """
        if X.ndim != 2:
            raise ValueError(f"X was expected 2D array, got {X.ndim}D array.")
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError(f"X must be of floating-point type, got {X.dtype.name}")
        if np.any(~np.isfinite(X)):
            raise ValueError(
                f"Input X contains NaN, infinity or a value too large for dtype('{X.dtype.name}')"
            )

    def _validate_y_numeric(self, y: RegressionTarget, /) -> None:
        """Validate that y is a well-formed 1D floating-point target array.

        Parameters
        ----------
        y : RegressionTarget
            Target array to validate.

        Raises
        ------
        ValueError
            If y is not 1D, or contains NaN/infinite/out-of-range values.
        TypeError
            If y's dtype is not floating-point.
        """
        if y.ndim != 1:
            raise ValueError(f"y was expected 1D array, got {y.ndim}D array.")
        if not np.issubdtype(y.dtype, np.floating):
            raise TypeError(f"y must be floating-point numbers, got {y.dtype.name}")
        if np.any(~np.isfinite(y)):
            raise ValueError(
                f"Input y contains NaN, infinity or a value too large for dtype('{y.dtype.name}')"
            )

    def _validate_y_label(self, y: ClassificationTarget, /) -> None:
        """Validate that y is a well-formed 1D integer label array.

        Parameters
        ----------
        y : ClassificationTarget
            Label array to validate.

        Raises
        ------
        ValueError
            If y is not 1D.
        TypeError
            If y's dtype is not integer.
        """
        if y.ndim != 1:
            raise ValueError(f"y labels expected 1D array, got {y.ndim}D array.")
        if not np.issubdtype(y.dtype, np.integer):
            raise TypeError(f"y labels must be integers, got {y.dtype.name}")

    def _validate_X_y_samples(  # noqa: N802
        self, X: FeatureMatrix, y: ClassificationTarget | RegressionTarget, /
    ) -> None:
        """Validate that X and y have a matching number of samples.

        Parameters
        ----------
        X : FeatureMatrix
            Feature matrix.
        y : ClassificationTarget or RegressionTarget
            Target array.

        Raises
        ------
        ShapeMismatchError
            If X and y have a different number of samples along axis 0.
        """
        if X.shape[0] != y.shape[0]:
            raise ShapeMismatchError(
                "Found input variables with inconsistent numbers of samples: "
                f"[{X.shape[0]}, {y.shape[0]}]"
            )
