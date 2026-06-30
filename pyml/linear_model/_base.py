"""Abstract base class for all linear models.

This module defines the common interface and shared functionality
that every linear model must implement, including fit/predict
contracts and fitted-state validation.

Classes
-------
BasicLinearModel
    Abstract base class enforcing the fit/predict interface.

Exceptions
----------
NotFittedError
    Raised when prediction is attempted on an unfitted model.
"""

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from ..exc import NotFittedError


class BasicLinearModel(ABC):
    """Abstract base class for all linear models.

    Provides common initialization and fitted-state checking logic
    shared across linear regression, logistic regression, ridge,
    and lasso implementations.

    Subclasses must implement the ``fit`` and ``predict`` methods,
    which define the model-specific training and inference behavior.

    Attributes
    ----------
    _fitted : bool
        Flag indicating whether the model has been fitted to data.
        Set to ``True`` by concrete ``fit`` implementations after
        successful training.

    Raises
    ------
    NotFittedError
        If ``predict`` is called on an unfitted model instance.
    """

    def __init__(self) -> None:
        """
        Initialize the base linear model.

        Sets the fitted flag to ``False``, indicating the model
        has not yet been trained on any data.

        Returns
        -------
        None
        """
        self._fitted = False

    def _check_fitted(self) -> None:
        """Verify that the model has been fitted before prediction.

        Raises
        ------
        NotFittedError
            If ``_fitted`` is ``False``, indicating ``fit`` has not
            been called or did not complete successfully.
        """
        if not self._fitted:
            raise NotFittedError(self)

    @abstractmethod
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[Any]) -> Self:
        """Fit the model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target vector. Type varies by subclass (continuous for
            regression, binary for classification).

        Returns
        -------
        self : BasicLinearModel
            The fitted estimator instance. Enables method chaining.
        """

    @abstractmethod
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Predict using the fitted model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values. Type varies by subclass (continuous for
            regression, binary labels for classification).

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
