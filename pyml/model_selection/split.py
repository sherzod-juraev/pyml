"""Utilities for splitting data into training, validation and test sets."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ..core.exceptions import InvalidParameterError, ShapeMismatchError
from ..core.param_checks import check_bool, check_integer


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


class KFold:
    """Split samples into consecutive train and validation folds.

    .. versionadded:: 0.8.0

    The samples are divided into ``n_splits`` groups, the folds. ``split`` yields
    one pair of index arrays per fold: the fold is the validation set and the
    other folds together are the training set. Every sample is in a validation
    set exactly once and in a training set ``n_splits - 1`` times, and the
    validation sets do not overlap.

    ``KFold`` only gives the indices. Fitting a model on them and scoring it is
    what ``cross_val_score`` does.

    Parameters
    ----------
    n_splits : int, optional
        Number of folds, at least 2. It must not be greater than the number of
        samples, which ``split`` checks. Defaults to 5.
    shuffle : bool, optional
        Whether to shuffle the samples before they are divided into folds. If
        False, the folds are consecutive blocks of the samples in their original
        order. Defaults to False.
    random_state : int or None, optional
        Non-negative seed of the generator that shuffles the samples. It has an
        effect only when ``shuffle`` is True. If None, every call of ``split``
        shuffles differently. Defaults to None.

    Attributes
    ----------
    n_splits : int
        The number of folds, as given.
    shuffle : bool
        Whether the samples are shuffled, as given.
    random_state : int or None
        The seed of the shuffling, as given.

    Notes
    -----
    The folds are not stratified: a fold does not keep the proportions of the
    classes. When the labels are sorted by class, use ``shuffle=True``, because
    otherwise a training set can miss a whole class. The first
    ``n_samples % n_splits`` folds have one sample more than the others.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Create a K-fold splitter.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds, at least 2. Defaults to 5.
        shuffle : bool, optional
            Whether to shuffle the samples before dividing them into folds.
            Defaults to False.
        random_state : int or None, optional
            Non-negative seed of the shuffling. Defaults to None.

        Raises
        ------
        InvalidParameterError
            If n_splits is not an integer of at least 2, shuffle is not a bool,
            or random_state is not None and is not a non-negative integer.
        """
        check_integer("n_splits", n_splits, minimum=2)
        check_bool("shuffle", shuffle)
        if random_state is not None:
            check_integer("random_state", random_state, minimum=0)

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: FeatureMatrix, y: npt.NDArray[Any] | None = None
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        """Yield the training and validation indices of each fold.

        Parameters
        ----------
        X : FeatureMatrix
            Feature matrix of shape (n_samples, n_features). Only its number of
            samples is used.
        y : ndarray or None, optional
            Targets of shape (n_samples,). Only its number of samples is
            checked. Defaults to None.

        Yields
        ------
        train_idx : ndarray
            Indices of the training samples of the fold, in increasing order.
        val_idx : ndarray
            Indices of the validation samples of the fold, in increasing order.

        Raises
        ------
        InvalidParameterError
            If n_splits is greater than the number of samples.
        ShapeMismatchError
            If X and y have a different number of samples.

        Notes
        -----
        ``split`` is a generator, so the errors above are raised when the first
        fold is requested, not when ``split`` is called.
        """
        n_samples = X.shape[0]
        if y is not None and y.shape[0] != n_samples:
            raise ShapeMismatchError(
                "Found input variables with inconsistent numbers of samples: "
                f"[{n_samples}, {y.shape[0]}]"
            )
        if self.n_splits > n_samples:
            raise InvalidParameterError(
                f"n_splits={self.n_splits} is greater than the number of samples, {n_samples}."
            )

        # The type is set, because the default integer is 32 bit on Windows with NumPy 1.x.
        indices = np.arange(n_samples, dtype=np.intp)
        if self.shuffle:
            # A new generator on every call, so a fixed random_state gives the same folds each time.
            np.random.default_rng(self.random_state).shuffle(indices)

        # The first n_samples % n_splits folds get one sample more than the others.
        for fold in np.array_split(indices, self.n_splits):
            val_idx = np.sort(fold)
            is_train = np.ones(n_samples, dtype=np.bool_)
            is_train[val_idx] = False
            yield np.flatnonzero(is_train), val_idx
