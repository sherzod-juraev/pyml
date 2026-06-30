r"""Linear Regression trained via Batch Gradient Descent.

This module provides a pure NumPy implementation of linear regression
with optional L1 (Lasso) and L2 (Ridge) regularization. All parameters
are learned iteratively using full-batch gradient descent on the MSE
loss function.

Classes
-------
LinearRegression
    Linear model for continuous target prediction with optional
    L1/L2 regularization.

Notes
-----
The MSE loss is convex, guaranteeing a single global minimum.
Gradient descent converges given a sufficiently small learning
rate. Convergence is detected using relative tolerance, which
is scale-invariant across datasets with different loss magnitudes.
"""

from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt

from ._base import BasicLinearModel


class LinearRegression(BasicLinearModel):
    r"""Linear Regression using Batch Gradient Descent.

    Models the relationship between features and a continuous target
    variable using a linear combination of inputs:

    .. math::

        \hat{y} = Xw + b

    The model minimizes the Mean Squared Error (MSE) loss with optional
    L1 or L2 regularization:

    .. math::

        J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        + R(w)

    where :math:`R(w)` is the regularization term:

    - L1 (Lasso): :math:`R(w) = \frac{\alpha}{n} \sum_{j=1}^{p} |w_j|`
    - L2 (Ridge): :math:`R(w) = \frac{\alpha}{2n} \sum_{j=1}^{p} w_j^2`

    Parameters are learned via batch gradient descent, updating all
    weights simultaneously using the full training set at each iteration.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size :math:`\eta` for each gradient descent iteration.
        Must be positive. Larger values may speed up convergence but
        risk overshooting; smaller values are more stable but require
        more iterations.
    max_iter : int, default=100
        Maximum number of gradient descent iterations. Serves as an
        upper bound to prevent infinite loops if convergence is slow.
    tol : float, default=1e-3
        Relative tolerance for convergence. Training stops early when
        the relative change in the training loss satisfies:

        .. math::

            \frac{|J_{new} - J_{old}|}{J_{old}} \leq tol

        A scale-invariant criterion suitable for datasets with varying
        loss magnitudes.
    alpha : float, default=1.0
        Regularization strength :math:`\alpha \geq 0`. Larger values
        push weights closer to zero. Ignored when ``penalty=None``.
    penalty : {'l1', 'l2', None}, default='l2'
        Type of regularization to apply:

        - ``'l1'`` — Lasso regression, encourages sparse weights.
        - ``'l2'`` — Ridge regression, shrinks weights smoothly.
        - ``None`` — Ordinary least squares, no regularization.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Learned weight vector after fitting. Each element corresponds
        to the coefficient for one input feature.
    intercept_ : float
        Learned bias term after fitting. Represents the predicted
        value when all features are zero.

    Notes
    -----
    **Gradients**

    The gradients of the MSE component with respect to weights and
    bias are:

    .. math::

        \frac{\partial J_{MSE}}{\partial w} =
        -\frac{2}{n} X^{T}(y - \hat{y})

    .. math::

        \frac{\partial J_{MSE}}{\partial b} =
        -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)

    Regularization adds the following gradient contributions:

    - **L1**:

      .. math::

          \frac{\partial R}{\partial w_j} =
          \frac{\alpha}{n} \cdot \text{sign}(w_j)

    - **L2**:

      .. math::

          \frac{\partial R}{\partial w} =
          \frac{\alpha}{n} w

    The intercept :math:`b` is never regularized.

    **Gradient Descent Updates**

    .. math::

        w := w - \eta \cdot \frac{\partial J}{\partial w}

    .. math::

        b := b - \eta \cdot \frac{\partial J}{\partial b}

    where :math:`\eta` is the learning rate.

    **Convergence**

    The MSE loss is convex, guaranteeing a single global minimum.
    With a sufficiently small learning rate, gradient descent is
    guaranteed to converge. Convergence detection uses **relative**
    tolerance, which is scale-invariant:

    - First iteration (:math:`J_{old} = \infty`) is skipped.
    - If :math:`J_{old} = 0` (perfect fit), training stops immediately.

    **Regularization behavior**

    - L1 regularization encourages sparse weight vectors by driving
      some coefficients exactly to zero, performing implicit feature
      selection. Uses subgradient at :math:`w_j = 0` (treated as 0).
    - L2 regularization shrinks all weights toward zero without
      eliminating them, improving numerical stability when features
      are correlated.

    Examples
    --------
    >>> import numpy as np
    >>> from pyml import LinearRegression
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3)
    >>> true_w = np.array([1.5, -2.0, 0.5])
    >>> true_b = 4.0
    >>> y = X @ true_w + true_b + 0.1 * np.random.randn(100)
    >>>
    >>> model = LinearRegression(
    ...     learning_rate=0.01,
    ...     max_iter=1000,
    ...     penalty=None
    ... )
    >>> model.fit(X, y)
    >>> model.coef_
    array([ 1.48..., -1.98...,  0.49...])
    >>> model.intercept_
    4.00...
    >>>
    >>> ridge = LinearRegression(
    ...     learning_rate=0.01,
    ...     alpha=0.1,
    ...     penalty='l2'
    ... )
    >>> ridge.fit(X, y)
    >>>
    >>> X_new = np.random.randn(10, 3)
    >>> predictions = model.predict(X_new)
    """

    def __init__(
        self,
        learning_rate: float = 1e-1,
        max_iter: int = 100,
        tol: float = 1e-3,
        alpha: float = 1.0,
        penalty: Literal["l1", "l2", None] = "l2",
    ) -> None:
        r"""Initialize the Linear Regression model with hyperparameters.

        Parameters
        ----------
        learning_rate : float, default=0.1
            Step size :math:`\eta` for gradient descent updates.
            Must be positive.
        max_iter : int, default=100
            Maximum number of gradient descent iterations.
        tol : float, default=1e-3
            Relative tolerance for early stopping convergence check.
        alpha : float, default=1.0
            Regularization strength :math:`\alpha \geq 0`. Ignored
            when ``penalty=None``.
        penalty : {'l1', 'l2', None}, default='l2'
            Type of regularization penalty to apply.

        Returns
        -------
        None
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.penalty = penalty
        super().__init__()

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Self:
        r"""Fit the linear regression model to training data.

        Initializes weights and bias to zero, then iteratively updates
        them using batch gradient descent on the MSE loss with optional
        regularization. Training stops when the relative change in loss
        falls below ``tol`` or ``max_iter`` is reached.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix. Should be dense and numeric.
            If 1D, reshape to (-1, 1) before passing.
        y : np.ndarray of shape (n_samples,)
            Continuous target values. Must have the same number of
            samples as ``X``.

        Returns
        -------
        self : LinearRegression
            Fitted estimator instance with ``coef_`` and ``intercept_``
            attributes populated. Enables method chaining.
        """
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0
        J_old = np.inf
        for _ in range(self.max_iter):
            y_pred = self.__cal_y(X)
            error = y - y_pred
            self.param_update(X, error)
            error = y - self.__cal_y(X)
            J_new = self.loss(error)
            if np.isinf(J_old):
                J_old = J_new
                continue
            if J_old == 0 or np.abs(J_new - J_old) / J_old <= self.tol:
                break
            J_old = J_new
        self._fitted = True
        return self

    def loss(self, error: npt.NDArray[np.float64]) -> float:
        r"""Compute the current value of the objective function.

        Combines Mean Squared Error with the configured regularization
        term (if any).

        **No regularization** (``penalty=None``):

        .. math::

            J = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

        **L1 regularization:**

        .. math::

            J = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
            + \frac{\alpha}{n} \sum_{j=1}^{p} |w_j|

        **L2 regularization:**

        .. math::

            J = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
            + \frac{\alpha}{2n} \sum_{j=1}^{p} w_j^2

        Parameters
        ----------
        error : np.ndarray of shape (n_samples,)
            Residuals :math:`(y - \hat{y})` from the current iteration.

        Returns
        -------
        loss : float
            Scalar value of the total loss :math:`J(w, b)`. Lower is better.
        """
        n = error.shape[0]
        J_new = np.mean(error**2)
        if self.penalty == "l1":
            J_new += (self.alpha / n) * np.sum(np.abs(self.coef_))
        elif self.penalty == "l2":
            J_new += (self.alpha / (2 * n)) * np.sum(self.coef_**2)
        return cast(float, J_new)

    def param_update(
        self, X: npt.NDArray[np.float64], error: npt.NDArray[np.float64]
    ) -> None:
        r"""Perform one gradient descent update of weights and bias.

        Computes the full gradient of the loss (MSE + regularization)
        and updates ``coef_`` and ``intercept_`` in-place using the
        configured learning rate.

        **Gradient computation:**

        The MSE gradient is:

        .. math::

            \nabla_w J_{MSE} = -2X^T (y - \hat{y})

        .. math::

            \nabla_b J_{MSE} = -2\sum_{i=1}^{n}
            (y_i - \hat{y}_i)

        Regularization gradients are added when applicable:

        - **L1**: :math:`\nabla_w R = \alpha \cdot
          \text{sign}(w)`, with :math:`\text{sign}(0) = 0`
        - **L2**: :math:`\nabla_w R = \alpha w`

        **Parameter update:**

        .. math::

            w := w - \frac{\eta}{n} \cdot \nabla_w J

        .. math::

            b := b - \frac{\eta}{n} \cdot \nabla_b J

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        error : np.ndarray of shape (n_samples,)
            Residuals :math:`(y - \hat{y})` from the current iteration.

        Returns
        -------
        None
        """
        n = X.shape[0]
        grad_w = -2 * (X.T @ error)
        grad_b = -2 * np.sum(error)
        if self.penalty == "l1":
            sign = np.where(self.coef_ > 0, 1, np.where(self.coef_ < 0, -1, 0))
            grad_w += self.alpha * sign
        elif self.penalty == "l2":
            grad_w += self.alpha * self.coef_
        self.coef_ -= (self.learning_rate / n) * grad_w
        self.intercept_ -= (self.learning_rate / n) * grad_b

    def __cal_y(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Compute predicted values using current weights and bias.

        Evaluates the linear model:

        .. math::

            \hat{y} = X w + b

        where :math:`w` is ``self.coef_`` and :math:`b` is
        ``self.intercept_``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Linear combination of inputs and learned parameters.
        """
        return X @ self.coef_ + self.intercept_

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Predict target values for new data using the fitted model.

        Computes predictions via the learned linear model:

        .. math::

            \hat{y} = X w + b

        The model must be fitted before calling this method.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix. Must have the same number of features
            as the training data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted continuous target values.

        Raises
        ------
        NotFittedError
            If ``predict`` is called before ``fit``. The model must
            be trained before making predictions.
        """
        self._check_fitted()
        return self.__cal_y(X)
