"""Abstract base class for regression estimators.

Defines the fit/predict lifecycle for models that output continuous
targets, including input validation and fit-state tracking. Concrete
models implement the _fit/_predict pair with their own algorithm.
"""

from abc import ABC, abstractmethod
from typing import Self

from ...metrics import r2_score
from ..dtypes import FeatureMatrix, RegressionTarget
from .estimator import BaseEstimator
from .validation import DataValidatorMixin


class Regressor(ABC, BaseEstimator, DataValidatorMixin):
    """Abstract base class for all pyml regressors.

    Subclasses implement _fit and _predict with their model-specific
    logic. The public fit/predict methods handle input validation and
    fit-state tracking automatically.
    """

    @abstractmethod
    def _fit(self, X: FeatureMatrix, y: RegressionTarget, /) -> None:
        """Fit the regressor to training data.

        Subclasses implement the model-specific fitting logic here. Inputs
        are already validated by the time this is called.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : RegressionTarget
            Continuous target values of shape (n_samples,).
        """

    def fit(self, X: FeatureMatrix, y: RegressionTarget, /) -> Self:
        """Validate inputs, fit the regressor, and mark it as fitted.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : RegressionTarget
            Continuous target values of shape (n_samples,).

        Returns
        -------
        Self
            The fitted estimator instance.

        Raises
        ------
        ValueError
            If X or y fail shape/dtype/finiteness validation.
        TypeError
            If X or y have an invalid dtype.
        ShapeMismatchError
            If X and y have a different number of samples.
        """
        self._validate_X(X)
        self._validate_y_numeric(y)
        self._validate_X_y_samples(X, y)
        self._fit(X, y)
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _predict(self, X: FeatureMatrix, /) -> RegressionTarget:
        """Generate predictions for the given input data.

        Subclasses implement the model-specific prediction logic here.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        RegressionTarget
            Predicted continuous values of shape (n_samples,).
        """

    def predict(self, X: FeatureMatrix, /) -> RegressionTarget:
        """Validate input and generate predictions.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        RegressionTarget
            Predicted continuous values of shape (n_samples,).

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If X fails shape/dtype/finiteness validation.
        """
        self._check_is_fitted()
        self._validate_X(X)
        return self._predict(X)

    def score(self, X: FeatureMatrix, y: RegressionTarget, /) -> float:
        r"""Compute the R^2 score of predictions on the given data.

        Equivalent to calling predict(X) and comparing the result against y
        using r2_score.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).
        y : RegressionTarget
            Ground truth target values corresponding to X.

        Returns
        -------
        float
            R^2 score. 1.0 indicates a perfect fit; can be negative for
            predictions worse than always predicting the mean.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If X or y fail shape/dtype/finiteness validation.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
