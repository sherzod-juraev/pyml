"""Abstract base class for gradient-descent-based linear models.

Centralizes the shared fit loop (gradient descent with early
stopping) and prediction logic for linear models. Subclasses supply
their own loss and gradient computation, allowing regularized
variants (Ridge, Lasso) to reuse the same optimization machinery.
"""

from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.base import Regressor
from ..core.dtypes import FeatureMatrix, RegressionTarget
from ..metrics import mean_squared_error

_FloatArray = npt.NDArray[np.floating[Any]]


class _LinearModelBase(Regressor):
    r"""Abstract base class for linear models fit via gradient descent.

    All linear models share the same prediction rule:

    .. math::
        \hat{y} = Xw + b

    and the same parameter-update rule, given a loss gradient:

    .. math::
        w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \quad
        b \leftarrow b - \alpha \frac{\partial J}{\partial b}

    Subclasses differ only in how the loss J and its gradient are
    computed, which lets them add regularization terms without
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
        Gradient descent is sensitive to the scale of the input
        features. If features have very different ranges (e.g. one
        column in the thousands, another between 0 and 1), the loss
        surface becomes elongated and convergence can be slow or
        unstable. Standardizing X (e.g. with a future StandardScaler)
        before fitting is strongly recommended.
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

    def _fit(self, X: FeatureMatrix, y: RegressionTarget, /) -> None:
        """Run gradient descent until convergence or max_iter is reached.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : RegressionTarget
            Continuous target values of shape (n_samples,).
        """
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        J_old: float = np.inf
        for i in range(self.max_iter):
            y_pred = self._predict(X)
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

    def _compute_loss(self, y_true: RegressionTarget, y_pred: RegressionTarget, /) -> float:
        """Compute the loss for the given predictions.

        Subclasses implement this with their own loss function, optionally
        including a regularization term.

        Parameters
        ----------
        y_true : RegressionTarget
            Ground truth target values.
        y_pred : RegressionTarget
            Predicted target values.

        Returns
        -------
        float
            The loss value.
        """
        return mean_squared_error(y_true, y_pred) + self._penalty_loss()

    def _compute_gradient(
        self, X: FeatureMatrix, residuals: _FloatArray, /
    ) -> tuple[_FloatArray, float]:
        """Compute the loss gradient with respect to coef_ and intercept_.

        Subclasses implement this with their own gradient formula,
        optionally including a regularization term's derivative.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        residuals : FloatArray
            Difference between predictions and true targets (y_pred - y_true).

        Returns
        -------
        tuple[FloatArray, float]
            Gradient with respect to coef_, and gradient with respect to
            intercept_.
        """
        grad_coef = (2 / X.shape[0]) * (X.T @ residuals) + self._penalty_gradient()
        grad_intercept = 0.0
        if self.fit_intercept:
            grad_intercept = (2 / X.shape[0]) * np.sum(residuals)
        return grad_coef, float(grad_intercept)

    @abstractmethod
    def _penalty_loss(self) -> float:
        """Compute this model's regularization term for the loss.

        Subclasses implement this to add their specific penalty (e.g.
        an L2 term for Ridge, an L1 term for Lasso) computed from the
        current ``coef_``. Unregularized models (e.g. plain OLS)
        return 0.0.

        Returns
        -------
        float
            The penalty value added to the base mean squared error.
        """

    @abstractmethod
    def _penalty_gradient(self) -> _FloatArray:
        """Compute the gradient of this model's regularization term.

        Subclasses implement this to add the derivative of their
        penalty with respect to ``coef_``. The result is added to
        the base MSE gradient before the parameter update.
        Unregularized models return an array of zeros.

        Returns
        -------
        FloatArray
            Gradient of the penalty term with respect to coef_, of
            shape (n_features,).
        """

    def _predict(self, X: FeatureMatrix, /) -> RegressionTarget:
        """Compute predictions as a linear combination of features.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        RegressionTarget
            Predicted continuous values of shape (n_samples,).
        """
        return X @ self.coef_ + self.intercept_
