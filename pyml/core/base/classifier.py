"""Abstract base class for classification estimators.

Defines the fit/predict lifecycle for models that output discrete
class labels, including input validation and fit-state tracking.
Concrete models implement the _fit/_predict pair with their own
algorithm.
"""

from abc import ABC, abstractmethod
from typing import Self

from ...metrics import accuracy_score
from ..dtypes import ClassificationTarget, FeatureMatrix
from .estimator import BaseEstimator
from .validation import DataValidatorMixin


class Classifier(ABC, BaseEstimator, DataValidatorMixin):
    """Abstract base class for all pyml classifiers.

    Subclasses implement _fit and _predict with their model-specific
    logic. The public fit/predict methods handle input validation and
    fit-state tracking automatically.
    """

    @abstractmethod
    def _fit(self, X: FeatureMatrix, y: ClassificationTarget, /) -> None:
        """Fit the classifier to training data.

        Subclasses implement the model-specific fitting logic here. Inputs
        are already validated by the time this is called.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : ClassificationTarget
            Integer-encoded class labels of shape (n_samples,).
        """

    def fit(self, X: FeatureMatrix, y: ClassificationTarget, /) -> Self:
        """Validate inputs, fit the classifier, and mark it as fitted.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : ClassificationTarget
            Integer-encoded class labels of shape (n_samples,).

        Returns
        -------
        Self
            The fitted estimator instance.

        Raises
        ------
        ValueError
            If X or y fail shape/dtype validation.
        TypeError
            If X or y have an invalid dtype.
        ShapeMismatchError
            If X and y have a different number of samples.
        """
        self._validate_X(X)
        self._validate_y_label(y)
        self._validate_X_y_samples(X, y)
        self._fit(X, y)
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        """Generate predictions for the given input data.

        Subclasses implement the model-specific prediction logic here.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget
            Predicted class labels of shape (n_samples,).
        """

    def predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        """Validate input and generate predictions.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget
            Predicted class labels of shape (n_samples,).

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

    def score(self, X: FeatureMatrix, y: ClassificationTarget, /) -> float:
        r"""Compute the accuracy of predictions on the given data.

        Equivalent to calling predict(X) and comparing the result against y
        using accuracy_score.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).
        y : ClassificationTarget
            Ground truth class labels corresponding to X.

        Returns
        -------
        float
            Fraction of samples where the prediction matches the ground
            truth, between 0.0 and 1.0.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If X or y fail shape/dtype/finiteness validation.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
