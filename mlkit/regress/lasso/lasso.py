from ..linear_reg import LinReg


class Lasso(LinReg):
    """
    Lasso Regression (L1-regularized Linear Regression).

    A specialization of Linear Regression that applies L1 regularization
    to the weight vector, encouraging sparse solutions by driving some
    coefficients to exactly zero — performing implicit feature selection.

    The model minimizes the following objective function:

    .. math::

        J(w, b) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
        + \\frac{\\alpha}{n} \\sum_{j=1}^{p} |w_j|

    where :math:`\\alpha` controls the regularization strength.
    Parameters are learned via batch gradient descent.

    This class is a thin wrapper around :class:`LinReg` with
    ``penalty`` fixed to ``'l1'``. All training logic, convergence
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

            \\frac{|J_{new} - J_{old}|}{J_{old}} \\leq tol

    alpha : float, default=1.0
        Regularization strength. Must be non-negative. Larger values
        push more coefficients toward exactly zero.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Learned weight vector after fitting. L1 regularization drives
        some coefficients to exactly zero, producing a sparse solution.

    intercept_ : float
        Learned bias term. Never regularized.

    Notes
    -----
    **Why L1 regularization?**

    When many features are present but only a few are truly predictive,
    L1 regularization is preferred. Unlike L2, the L1 penalty creates
    a non-smooth loss surface with corners at zero, which causes gradient
    descent to land exactly on zero for irrelevant features:

    .. math::

        \\frac{\\partial R}{\\partial w_j} =
        \\frac{\\alpha}{n} \\cdot \\text{sign}(w_j)

    At :math:`w_j = 0`, the subgradient is treated as 0, allowing
    coefficients to remain exactly zero once they reach it.

    **Gradient of the regularized loss**

    The L1 term contributes the following additive gradient to weights:

    .. math::

        \\frac{\\partial R}{\\partial w_j} =
        \\frac{\\alpha}{n} \\cdot \\text{sign}(w_j)

    Combined with the MSE gradient, the full weight update becomes:

    .. math::

        w := w - \\eta \\left(
            -\\frac{2}{n} X^T (y - \\hat{y})
            + \\frac{\\alpha}{n} \\cdot \\text{sign}(w)
        \\right)

    The bias :math:`b` is not regularized.

    **Lasso vs Ridge**

    - Lasso (L1) drives irrelevant coefficients to exactly zero,
      making it suitable for high-dimensional sparse problems.
    - Ridge (L2) shrinks all coefficients smoothly without eliminating
      any of them.
    - Prefer Lasso when you suspect only a few features are relevant,
      or when an interpretable sparse model is desired.

    **Limitation with gradient descent**

    Lasso's closed-form solution does not exist due to the
    non-differentiability of :math:`|w_j|` at zero. Coordinate
    Descent with soft-thresholding is the standard solver in practice
    (e.g., scikit-learn). This implementation uses subgradient-based
    batch gradient descent, which is simpler but converges slower
    and may not reach exact zeros.

    Examples
    --------
    >>> import numpy as np
    >>> from mlkit import Lasso
    >>>
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 5)
    >>> true_w = np.array([2.0, -1.5, 0.0, 0.0, 0.0])
    >>> y = X @ true_w + 1.0 + 0.1 * np.random.randn(100)
    >>>
    >>> model = Lasso(learning_rate=0.01, max_iter=1000, alpha=0.5)
    >>> model.fit(X, y)
    >>> print(model.coef_)
    >>> print(model.intercept_)
    >>>
    >>> model_strong = Lasso(alpha=2.0)
    >>> model_strong.fit(X, y)
    >>>
    >>> X_new = np.random.randn(10, 5)
    >>> predictions = model.predict(X_new)

    See Also
    --------
    LinReg : Base class with full gradient descent implementation.
    Ridge : L2-regularized variant that shrinks without sparsity.
    """

    def __init__(
            self,
            learning_rate: float = 0.1,
            max_iter: int = 100,
            tol: float = 1e-3,
            alpha: float = 1.0
    ):
        super().__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            penalty='l1'
        )