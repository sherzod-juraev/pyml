"""Abstract base class for transformer estimators.

Defines the fit/transform(/inverse_transform) lifecycle for models
that reshape or rescale feature data without producing a target
prediction, including input validation and fit-state tracking.
Concrete transformers implement the _fit/_transform/_inverse_transform
trio with their own logic.
"""

from abc import ABC, abstractmethod
from typing import Self

from ..dtypes import FeatureMatrix
from .estimator import BaseEstimator
from .validation import DataValidatorMixin


class Transformer(ABC, BaseEstimator, DataValidatorMixin):
    """Abstract base class for all pyml transformers.

    Subclasses implement _fit, _transform, and _inverse_transform with
    their model-specific logic. The public fit/transform/inverse_transform
    methods handle input validation and fit-state tracking automatically.
    """

    @abstractmethod
    def _fit(self, X: FeatureMatrix, /) -> None:
        """Fit the transformer to training data.

        Subclasses implement the model-specific fitting logic here (e.g.
        computing statistics needed for transform). Input is already
        validated by the time this is called.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        """

    def fit(self, X: FeatureMatrix, /) -> Self:
        """Validate input, fit the transformer, and mark it as fitted.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).

        Returns
        -------
        Self
            The fitted estimator instance.

        Raises
        ------
        ValueError
            If X fails shape/dtype/finiteness validation.
        TypeError
            If X has an invalid dtype.
        """
        self._validate_X(X)
        self._fit(X)
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Apply the learned transformation to the given data.

        Subclasses implement the model-specific transformation logic here.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Transformed data of the same shape as X.
        """

    def transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Validate input and apply the learned transformation.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Transformed data of the same shape as X.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If X fails shape/dtype/finiteness validation.
        """
        self._check_is_fitted()
        self._validate_X(X)
        return self._transform(X)

    @abstractmethod
    def _inverse_transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Reverse the learned transformation for the given data.

        Subclasses implement the model-specific inverse-transformation logic
        here.

        Parameters
        ----------
        X : FeatureMatrix
            Transformed data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Data mapped back to its original scale, of the same shape as X.
        """

    def inverse_transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Validate input and reverse the learned transformation.

        Parameters
        ----------
        X : FeatureMatrix
            Transformed data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Data mapped back to its original scale, of the same shape as X.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If X fails shape/dtype/finiteness validation.
        """
        self._check_is_fitted()
        self._validate_X(X)
        return self._inverse_transform(X)

    def fit_transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Fit the transformer to X, then transform X.

        Equivalent to calling fit(X) followed by transform(X), but provided
        as a single convenience method.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Transformed data of the same shape as X.
        """
        return self.fit(X).transform(X)
