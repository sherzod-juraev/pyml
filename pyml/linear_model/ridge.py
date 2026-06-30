r"""Ridge Regression with L2 regularization trained via Batch Gradient Descent.

This module provides a thin wrapper around :class:`LinearRegression`
with ``penalty`` fixed to ``'l2'``, applying quadratic weight penalty
to prevent overfitting and handle multicollinearity.

Classes
-------
Ridge
    L2-regularized linear regression for stable coefficient estimation.

Notes
-----
L2 regularization adds a quadratic penalty :math:`\frac{\alpha}{2n} \sum w_j^2`
to the MSE loss, making the optimization problem strictly convex even
when features are highly correlated. Unlike L1 (Lasso), L2 never produces
exactly zero coefficients — all features retain some contribution.
"""

from .linear_regression import LinearRegression


class Ridge(LinearRegression):
    r"""Ridge Regression (L2-regularized Linear Regression).

    A specialization of Linear Regression that applies L2 regularization
    to the weight vector, penalizing large coefficients and improving
    generalization on correlated or high-dimensional feature sets.

    The model minimizes the following objective function:

    .. math::

        J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        + \frac{\alpha}{2n} \sum_{j=1}^{p} w_j^2

    where :math:`\alpha` controls the regularization strength.
    Parameters are learned via batch gradient descent.

    This class is a thin wrapper around :class:`LinearRegression` with
    ``penalty`` fixed to ``'l2'``. All training logic, convergence
    detection, and gradient updates are inherited from the parent class.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for each gradient descent iteration. Must be positive.
    max_iter : int, default=100
        Maximum number of gradient descent iterations.
    tol : float, default=1e-3
        Relative tolerance for early stopping. Training halts when:

        .. math::

            \frac{|J_{new} - J_{old}|}{J_{old}} \leq tol

    alpha : float, default=1.0
        Regularization strength. Must be non-negative. Larger values
        shrink weights more aggressively toward zero.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Learned weight vector after fitting. L2 regularization shrinks
        all coefficients toward zero but does not eliminate them.
    intercept_ : float
        Learned bias term. Never regularized.

    Notes
    -----
    **Why L2 regularization?**

    Ordinary least squares becomes unstable when features are highly
    correlated (multicollinearity) or when :math:`p \approx n`.
    Ridge adds a curvature to the loss surface, making the problem
    strictly convex and the solution unique even in ill-conditioned cases:

    .. math::

        w^* = (X^T X + \alpha I)^{-1} X^T y

    Although this class uses gradient descent (not the closed form),
    the L2 penalty ensures the same stable minimum is reached.

    **Gradient of the regularized loss**

    The L2 term contributes the following additive gradient to weights:

    .. math::

        \frac{\partial R}{\partial w} = \frac{\alpha}{n} w

    Combined with the MSE gradient, the full weight update becomes:

    .. math::

        w := w - \eta \left(
            -\frac{2}{n} X^T (y - \hat{y}) + \frac{\alpha}{n} w
        \right)

    The bias :math:`b` is not regularized.

    **Ridge vs Lasso**

    - Ridge (L2) shrinks all weights smoothly — no coefficient reaches
      exactly zero.
    - Lasso (L1) can drive some coefficients to exactly zero, performing
      implicit feature selection.
    - Prefer Ridge when all features are expected to contribute, or when
      multicollinearity is a concern.

    Examples
    --------
    >>> import numpy as np
    >>> from pyml import Ridge
    >>>
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 5)
    >>> X[:, 1] = X[:, 0] + 0.01 * np.random.randn(100)
    >>> true_w = np.array([1.5, -2.0, 0.5, 0.0, 1.0])
    >>> y = X @ true_w + 3.0 + 0.1 * np.random.randn(100)
    >>>
    >>> model = Ridge(learning_rate=0.01, max_iter=1000, alpha=1.0)
    >>> model.fit(X, y)
    >>> print(model.coef_)
    >>> print(model.intercept_)
    >>>
    >>> model_weak = Ridge(alpha=0.01)
    >>> model_weak.fit(X, y)
    >>>
    >>> X_new = np.random.randn(10, 5)
    >>> predictions = model.predict(X_new)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-3,
        alpha: float = 1.0,
    ) -> None:
        r"""Initialize Ridge model with L2 regularization.

        Passes all hyperparameters to the parent :class:`LinearRegression`
        with ``penalty`` fixed to ``'l2'``. No additional configuration
        is needed beyond the standard linear model parameters.

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
            L2 regularization strength :math:`\alpha \geq 0`. Larger
            values shrink all weights more aggressively toward zero.

        Returns
        -------
        None
        """
        super().__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            penalty="l2",
        )
