"""Regression evaluation metrics.

Provides standard error and goodness-of-fit measures for comparing
predicted continuous values against ground truth, used by Regressor
subclasses' score method and by test suites for verifying model
correctness.
"""

import numpy as np

from ..core.dtypes import RegressionTarget
from ..core.exceptions import ShapeMismatchError


def _validate_regression_inputs(y_true: RegressionTarget, y_pred: RegressionTarget, /) -> None:
    """Validate regression targets for shape, dtype, and finiteness.

    Parameters
    ----------
    y_true : RegressionTarget
        Ground truth target values.
    y_pred : RegressionTarget
        Predicted target values.

    Raises
    ------
    ValueError
        If y_true or y_pred is not 1D, or contains NaN/infinite values.
    TypeError
        If y_true or y_pred does not have a floating-point dtype.
    ShapeMismatchError
        If y_true and y_pred have a different number of samples.
    """
    if y_true.ndim != 1:
        raise ValueError(f"y_true was expected 1D array, got {y_true.ndim}D array.")
    if not np.issubdtype(y_true.dtype, np.floating):
        raise TypeError(f"y_true must be floating-point numbers, got {y_true.dtype.name}")
    if np.any(~np.isfinite(y_true)):
        raise ValueError(
            "Input y_true contains NaN, infinity or a value too large for"
            f" dtype('{y_true.dtype.name}')"
        )
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred was expected 1D array, got {y_pred.ndim}D array.")
    if not np.issubdtype(y_pred.dtype, np.floating):
        raise TypeError(f"y_pred must be floating-point numbers, got {y_pred.dtype.name}")
    if np.any(~np.isfinite(y_pred)):
        raise ValueError(
            "Input y_pred contains NaN, infinity or a value too large for"
            f" dtype('{y_pred.dtype.name}')"
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ShapeMismatchError(
            "Found input variables with inconsistent numbers of samples: "
            f"[{y_true.shape[0]}, {y_pred.shape[0]}]"
        )


def mean_squared_error(y_true: RegressionTarget, y_pred: RegressionTarget, /) -> float:
    r"""Compute the mean squared error between true and predicted values.

    .. math::
        MSE = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2

    Parameters
    ----------
    y_true : RegressionTarget
        Ground truth target values.
    y_pred : RegressionTarget
        Predicted target values.

    Returns
    -------
    float
        The mean squared error. Always non-negative; 0 indicates a
        perfect match.
    """
    _validate_regression_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: RegressionTarget, y_pred: RegressionTarget, /) -> float:
    r"""Compute the root mean squared error between true and predicted values.

    .. math::
        RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}

    Parameters
    ----------
    y_true : RegressionTarget
        Ground truth target values.
    y_pred : RegressionTarget
        Predicted target values.

    Returns
    -------
    float
        The root mean squared error, in the same units as the target.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: RegressionTarget, y_pred: RegressionTarget, /) -> float:
    r"""Compute the mean absolute error between true and predicted values.

    .. math::
        MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|

    Parameters
    ----------
    y_true : RegressionTarget
        Ground truth target values.
    y_pred : RegressionTarget
        Predicted target values.

    Returns
    -------
    float
        The mean absolute error. Less sensitive to outliers than MSE.
    """
    _validate_regression_inputs(y_true, y_pred)
    return float(np.mean(np.absolute(y_true - y_pred)))


def r2_score(y_true: RegressionTarget, y_pred: RegressionTarget, /) -> float:
    r"""Compute the coefficient of determination (R^2).

    .. math::
        R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

        SS_{res} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2

        SS_{tot} = \sum_{i=1}^{n} \left( y_i - \bar{y} \right)^2

    Parameters
    ----------
    y_true : RegressionTarget
        Ground truth target values.
    y_pred : RegressionTarget
        Predicted target values.

    Returns
    -------
    float
        R^2 score. 1.0 indicates a perfect fit; can be negative for
        predictions worse than always predicting the mean. Falls back to
        1.0 or 0.0 when y_true has zero variance, since the ratio is
        undefined in that case.
    """
    _validate_regression_inputs(y_true, y_pred)
    SS_res = np.sum((y_true - y_pred) ** 2)
    y_mean = np.mean(y_true)
    SS_tot = np.sum((y_true - y_mean) ** 2)
    if np.isclose(SS_tot, 0):
        return 1.0 if np.isclose(SS_res, 0) else 0.0
    return float(1.0 - (SS_res / SS_tot))
