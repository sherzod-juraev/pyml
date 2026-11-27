"""Evaluating an estimator on the validation folds of a cross-validation."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.clone import clone
from ..core.dtypes import FeatureMatrix
from ..core.protocols import SupervisedLike
from .split import KFold


def cross_val_score(
    estimator: SupervisedLike,
    X: FeatureMatrix,
    y: npt.NDArray[Any],
    *,
    cv: int | KFold = 5,
    scoring: Callable[[SupervisedLike, FeatureMatrix, npt.NDArray[Any]], float] | None = None,
) -> npt.NDArray[np.float64]:
    """Evaluate an estimator by cross-validation.

    .. versionadded:: 0.8.0

    For every fold of ``cv`` a new, unfitted copy of ``estimator`` is made with
    ``clone``, fitted on the training part and scored on the validation part.
    The estimator itself is never fitted. When the preprocessing is a step of a
    ``Pipeline``, it is fitted on each training part only, so nothing from a
    validation part reaches it.

    Parameters
    ----------
    estimator : SupervisedLike
        A supervised model, or a pipeline that ends in one.
    X : FeatureMatrix
        Feature matrix of shape (n_samples, n_features).
    y : ndarray
        Targets of shape (n_samples,).
    cv : int or KFold, optional
        The folds. An integer ``k`` means ``KFold(k)``, which does not shuffle.
        Pass a ``KFold`` to shuffle or to set ``random_state``. Defaults to 5.
    scoring : callable or None, optional
        A function ``scoring(estimator, X_val, y_val)`` that returns a float. It
        is called with the copy fitted on the training part and with the
        validation part. If None, the ``score`` method of the estimator is used,
        that is :math:`R^2` for a regressor and the accuracy for a classifier.
        Defaults to None.

    Returns
    -------
    ndarray
        The score of each fold, of shape (n_splits,).

    Raises
    ------
    InvalidParameterError
        If ``cv`` is not an integer of at least 2 or a ``KFold``, or asks for
        more folds than there are samples.
    ShapeMismatchError
        If X and y have a different number of samples. Errors that ``fit`` or
        ``score`` of the estimator raise are not caught.

    Notes
    -----
    The default score means "higher is better". A ``scoring`` function is used
    as it is, so one that returns an error, such as the mean squared error,
    gives values where lower is better. The sign is not changed.

    When the labels are sorted by class, pass ``KFold(k, shuffle=True,
    random_state=...)``, because the folds are not stratified.
    """
    folds = cv if isinstance(cv, KFold) else KFold(cv)

    scores = np.empty(folds.n_splits, dtype=np.float64)
    for i, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        # Every fold gets a new, unfitted copy, so the estimator itself is never fitted.
        fold_estimator = clone(estimator)
        fold_estimator.fit(X[train_idx], y[train_idx])

        X_val, y_val = X[val_idx], y[val_idx]
        if scoring is None:
            scores[i] = fold_estimator.score(X_val, y_val)
        else:
            scores[i] = scoring(fold_estimator, X_val, y_val)
    return scores
