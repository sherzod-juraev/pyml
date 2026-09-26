"""Utilities for splitting data into training and test sets."""

import numpy as np

from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ..core.exceptions import ShapeMismatchError


def _validate_inputs(
    X: FeatureMatrix, y: ClassificationTarget | RegressionTarget, test_size: float, /
) -> None:
    """Validate X, y, and test_size for train_test_split.

    Parameters
    ----------
    X : FeatureMatrix
        Feature matrix to split.
    y : ClassificationTarget or RegressionTarget
        Target values to split.
    test_size : float
        Proportion of the data to allocate to the test set.

    Raises
    ------
    ValueError
        If X or y is not the expected number of dimensions, X contains
        NaN/infinite values, test_size is not strictly between 0 and 1,
        or the resulting test set would be empty.
    TypeError
        If X is not floating-point, or y is neither floating-point nor
        integer.
    ShapeMismatchError
        If X and y have a different number of samples.
    """
    if X.ndim != 2:
        raise ValueError(f"X was expected 2D array, got {X.ndim}D array.")
    if not np.issubdtype(X.dtype, np.floating):
        raise TypeError(f"X must be of floating-point type, got {X.dtype.name}")
    if np.any(~np.isfinite(X)):
        raise ValueError(
            f"Input X contains NaN, infinity or a value too large for dtype('{X.dtype.name}')"
        )
    if not np.issubdtype(y.dtype, np.integer) and not np.issubdtype(y.dtype, np.floating):
        raise TypeError(f"y must be floating-points numbers or integers, got {y.dtype.name}")
    if y.ndim != 1:
        raise ValueError(f"y expected 1D array, got {y.ndim}D array.")
    if X.shape[0] != y.shape[0]:
        raise ShapeMismatchError(
            "Found input variables with inconsistent numbers of samples: "
            f"[{X.shape[0]}, {y.shape[0]}]"
        )
    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            f"test_size must be between 0.0 and 1.0 (exclusive). Got {test_size} instead."
        )
    n_test = int(X.shape[0] * test_size)
    if n_test == 0:
        raise ValueError(
            f"With n_samples={X.shape[0]} and test_size={test_size}, "
            f"the resulting test set will be empty. Adjust test_size or provide more data."
        )


def train_test_split(
    X: FeatureMatrix,
    y: ClassificationTarget | RegressionTarget,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[
    FeatureMatrix,
    FeatureMatrix,
    ClassificationTarget | RegressionTarget,
    ClassificationTarget | RegressionTarget,
]:
    """Split feature and target arrays into random train and test subsets.

    Parameters
    ----------
    X : FeatureMatrix
        Feature matrix of shape (n_samples, n_features).
    y : ClassificationTarget or RegressionTarget
        Target values of shape (n_samples,).
    test_size : float, optional
        Proportion of the data to allocate to the test set, strictly
        between 0 and 1.
    random_state : int or None, optional
        Seed for the random number generator used to shuffle the data,
        for reproducible splits. If None, the split is not reproducible
        across calls.
    shuffle : bool, optional
        Whether to shuffle the data before splitting. If False, the
        first samples become the training set and the last become the
        test set, in their original order.

    Returns
    -------
    tuple[
        FeatureMatrix,
        FeatureMatrix,
        ClassificationTarget | RegressionTarget,
        ClassificationTarget | RegressionTarget,
    ]
        X_train, X_test, y_train, y_test, in that order.

    Raises
    ------
    ValueError
        If X or y fail shape/finiteness validation, test_size is not
        strictly between 0 and 1, or the resulting test set would be
        empty.
    TypeError
        If X or y have an invalid dtype.
    ShapeMismatchError
        If X and y have a different number of samples.
    """
    _validate_inputs(X, y, test_size)
    n_test = int(X.shape[0] * test_size)
    indices = np.arange(X.shape[0])
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
