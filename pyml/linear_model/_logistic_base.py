"""Abstract base class for gradient-descent-based logistic regression models.

Centralizes the shared fit loop (gradient descent with early
stopping), sigmoid-based prediction, and binary cross-entropy loss
for logistic regression models. Subclasses supply their own
regularization penalty and its gradient, allowing regularized
variants (RidgeClassifier, LassoClassifier) to reuse the same
optimization machinery.
"""

from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.base import Classifier
from ..core.dtypes import ClassificationTarget, FeatureMatrix

_FloatArray = npt.NDArray[np.floating[Any]]


class _LogisticRegressionBase(Classifier):
    r"""Abstract base class for logistic regression models fit via gradient descent.

    All logistic regression models share the same prediction rule: a
    linear combination of features is passed through the sigmoid
    function to produce a probability, which is then thresholded to a
    class label:

    .. math::
        z = Xw + b, \quad \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}

    .. math::
        \hat{y} = \begin{cases} 1 & \text{if } \hat{p} \geq 0.5 \\
        0 & \text{otherwise} \end{cases}

    Fitting minimizes binary cross-entropy loss, optionally plus a
    regularization term:

    .. math::
        J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{p}_i)
            + (1 - y_i) \log(1 - \hat{p}_i) \right]

    Sigmoid and cross-entropy are paired deliberately: their combination
    yields a gradient with the same simple form as ordinary least
    squares, using the probability residual in place of the continuous
    one:

    .. math::
        \frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{p} - y), \quad
        \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{p}_i - y_i)

    and the same parameter-update rule as the linear models:

    .. math::
        w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \quad
        b \leftarrow b - \alpha \frac{\partial J}{\partial b}

    Subclasses differ only in how the regularization penalty and its
    gradient are computed, which lets them add penalty terms without
    duplicating the optimization loop.

    Parameters
    ----------
    learning_rate : float, optional
        Step size alpha for each gradient descent update.
    max_iter : int, optional
        Maximum number of gradient descent iterations.
    tol : float, optional
        Minimum absolute change in loss between consecutive iterations
        required to continue optimizing; if the change falls below this,
        training stops early.
    fit_intercept : bool, optional
        Whether to fit an intercept term b. If False, b remains 0.

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping (either by
        convergence or by reaching max_iter).


    .. note::
        Predicted probabilities are clipped away from exactly 0 or 1
        before use, to avoid ``log(0)`` and division-by-zero issues in
        the loss and its gradient. As with the linear regression models,
        features should be standardized before fitting for stable
        convergence.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize gradient descent hyperparameters and fit-state.

        Parameters
        ----------
        learning_rate : float, optional
            Step size alpha for each gradient descent update. Defaults to
            0.01.
        max_iter : int, optional
            Maximum number of gradient descent iterations. Defaults to 1000.
        tol : float, optional
            Minimum absolute change in loss between consecutive iterations
            required to continue optimizing. Defaults to 1e-4.
        fit_intercept : bool, optional
            Whether to fit an intercept term b. Defaults to True.
        """
        super().__init__()
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.fit_intercept: bool = fit_intercept

    def _fit(self, X: FeatureMatrix, y: ClassificationTarget, /) -> None:
        """Run gradient descent until convergence or max_iter is reached.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : ClassificationTarget
            Binary class labels (0 or 1) of shape (n_samples,).
        """
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        J_old: float = np.inf
        for i in range(self.max_iter):
            y_pred = self.predict_proba(X)
            residuals = y_pred - y
            J_new = self._compute_loss(y, y_pred)
            grad_coef, grad_intercept = self._compute_gradient(X, residuals)
            self.coef_ -= self.learning_rate * grad_coef
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_intercept
            if np.abs(J_new - J_old) <= self.tol:
                self.n_iter_ = i + 1
                break
            J_old = J_new
        else:
            self.n_iter_ = self.max_iter

    def _compute_loss(self, y_true: ClassificationTarget, y_pred: _FloatArray, /) -> float:
        """Compute binary cross-entropy loss for the given probabilities.

        Subclasses implement ``_penalty_loss`` to add a regularization term.

        Parameters
        ----------
        y_true : ClassificationTarget
            Ground truth binary labels.
        y_pred : FloatArray
            Predicted probabilities of the positive class.

        Returns
        -------
        float
            The loss value.
        """
        return (
            -float(
                (1 / y_true.shape[0])
                * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            )
            + self._penalty_loss()
        )

    def _compute_gradient(
        self, X: FeatureMatrix, residuals: _FloatArray, /
    ) -> tuple[_FloatArray, float]:
        """Compute the loss gradient with respect to coef_ and intercept_.

        Subclasses implement ``_penalty_gradient`` to add a regularization
        term's derivative.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        residuals : FloatArray
            Difference between predicted probabilities and true labels
            (y_pred - y_true).

        Returns
        -------
        tuple[FloatArray, float]
            Gradient with respect to coef_, and gradient with respect to
            intercept_.
        """
        grad_coef = (1 / X.shape[0]) * (X.T @ residuals) + self._penalty_gradient()
        grad_intercept = 0.0
        if self.fit_intercept:
            grad_intercept = (1 / X.shape[0]) * np.sum(residuals)
        return grad_coef, float(grad_intercept)

    @abstractmethod
    def _penalty_loss(self) -> float:
        """Compute this model's regularization term for the loss.

        Subclasses implement this to add their specific penalty (e.g. an L2
        term for RidgeClassifier, an L1 term for LassoClassifier) computed
        from the current ``coef_``. Unregularized models return 0.0.

        Returns
        -------
        float
            The penalty value added to the base cross-entropy loss.
        """

    @abstractmethod
    def _penalty_gradient(self) -> _FloatArray:
        """Compute the gradient of this model's regularization term.

        Subclasses implement this to add the derivative of their penalty
        with respect to ``coef_``. The result is added to the base gradient
        before the parameter update. Unregularized models return an array
        of zeros.

        Returns
        -------
        FloatArray
            Gradient of the penalty term with respect to coef_, of shape
            (n_features,).
        """

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        """Predict binary class labels by thresholding predicted probabilities.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget
            Predicted labels (0 or 1) of shape (n_samples,), using a 0.5
            threshold on predicted probabilities.
        """
        y_pred = self.predict_proba(X)
        return np.where(y_pred >= 0.5, 1, 0)

    def predict_proba(self, X: FeatureMatrix, /) -> _FloatArray:
        r"""Compute the predicted probability of the positive class.

        .. math::
            \hat{p} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}

        The linear combination is clipped before the sigmoid to avoid
        overflow, and the resulting probability is clipped away from exactly
        0 or 1 to avoid downstream ``log(0)`` errors.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Predicted probabilities of the positive class, of shape
            (n_samples,), in the open interval (0, 1).
        """
        z = X @ self.coef_ + self.intercept_
        np.clip(z, -3e2, 3e2, out=z)
        y_pred = 1 / (1 + np.exp(-z))
        np.clip(y_pred, 1e-15, 1 - 1e-15, out=y_pred)
        return y_pred
