import numpy as np
from typing import Self
from ...exc import NotFitted


class LinReg:
    """
    Linear Regression using Batch Gradient Descent.

    Models the relationship between features and a continuous target
    variable using a linear combination of inputs:

    .. math::

        \\hat{y} = Xw + b

    Parameters are learned by minimizing Mean Squared Error (MSE) loss:

    .. math::

        J(w, b) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2

    Weights are updated iteratively using batch gradient descent:

    .. math::

        w := w - \\eta \\cdot \\frac{\\partial J}{\\partial w}
        = w + \\frac{2\\eta}{n} X^{T}(y - \\hat{y})

        b := b - \\eta \\cdot \\frac{\\partial J}{\\partial b}
        = b + \\frac{2\\eta}{n} \\sum_{i=1}^{n}(y_i - \\hat{y}_i)


    Parameters
    ----------
    eta : float, default=0.1
        Learning rate controlling the step size at each iteration.
    max_iter : int, default=100
        Maximum number of gradient descent iterations.
    tol : float, default=1e-3
        Relative tolerance for convergence. Training stops early
        when the relative change in loss falls below this threshold:

        :math:`\\frac{|J_{new} - J_{old}|}{J_{old}} \\leq tol`

    Attributes
    ----------
    coef\\_ : np.ndarray of shape (n_features,)
        Learned weights after fitting.
    intercept\\_ : float
        Learned bias term after fitting.

    Notes
    -----
    MSE loss is convex with respect to the model parameters,
    guaranteeing a single global minimum. Gradient descent is
    therefore guaranteed to converge given a sufficiently small
    :math:`\\eta`.

    The gradients of MSE with respect to weights and bias are:

    .. math::

        \\frac{\\partial J}{\\partial w} = -\\frac{2}{n} X^{T}(y - \\hat{y})

        \\frac{\\partial J}{\\partial b} =
        -\\frac{2}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)

    Convergence uses relative tolerance (scale-invariant) rather than
    absolute tolerance to handle datasets with varying loss magnitudes.
    Two special cases are handled explicitly:

    - :math:`J_{old} = \\infty` (first iteration) — skipped via ``continue``
    - :math:`J_{old} = 0` (perfect fit) — stopped immediately via ``break``

    Examples
    --------
    >>> model = LinReg(eta=0.01, max_iter=500)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(
            self,
            eta: float = 1e-1,
            max_iter: int = 100,
            tol: float = 1e-3
    ) -> None:

        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.__fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the model to training data using batch gradient descent.

        Initializes weights and bias to zero, then iteratively updates
        them using the MSE gradient. Training stops when the relative
        change in loss falls below ``tol`` or ``max_iter`` is reached.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self : LinReg
            Fitted estimator.
        """

        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0
        J_old = np.inf
        n = X.shape[0]
        for _ in range(self.max_iter):
            y_pred = self.__cal_y(X)
            error = y - y_pred
            self.coef_ += self.eta * (2 / n) * X.T @ error
            self.intercept_ += self.eta * (2 / n) * np.sum(error)
            J_new = np.mean(error ** 2)
            if np.isinf(J_old):
                J_old = J_new
                continue
            if J_old == 0 or \
                    np.abs(J_new - J_old) / J_old <= self.tol:
                break
            J_old = J_new
        self.__fitted = True
        return self

    def __cal_y(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions using current weights and bias.

        Evaluates the linear model:

        .. math::

            \\hat{y} = X w + b

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Linear combination of inputs and learned weights.
        """

        return X @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted continuous target values.

        Raises
        ------
        NotFitted
            If called before fitting the model.
        """

        if not self.__fitted:
            raise NotFitted(self)
        return self.__cal_y(X)