"""Lasso regression (L1-regularized linear regression)."""

import numpy as np

from ._linear_base import _FloatArray, _LinearModelBase


class Lasso(_LinearModelBase):
    r"""Linear regression with L1 (lasso) regularization, fit via gradient descent.

    Predicts a continuous target as a linear combination of features,
    identically to :class:`~pyml.linear_model.linear_regression.LinearRegression`:

    .. math::
        \hat{y} = Xw + b

    Fitting minimizes the mean squared error plus an L1 penalty on the
    coefficients:

    .. math::
        J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_i - y_i \right)^2
            + \alpha \sum_{j=1}^{n} |w_j|

    The intercept b is not penalized, since it only shifts the data and
    does not contribute to model complexity. Unlike the L2 penalty, :math:`|w_j|`
    is not differentiable at :math:`w_j = 0`, so a subgradient is used instead:

    .. math::
        \frac{\partial J}{\partial w} = \frac{2}{m} X^T (\hat{y} - y)
            + \alpha \cdot \text{sign}(w), \quad
        \frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)

    where sign(w) is applied elementwise and sign(0) = 0. This gradient is
    used to update the parameters at each iteration via gradient descent:

    .. math::
        w \leftarrow w - \alpha_{lr} \frac{\partial J}{\partial w}, \quad
        b \leftarrow b - \alpha_{lr} \frac{\partial J}{\partial b}

    where alpha_lr is the learning rate (distinct from the
    regularization strength alpha above).

    Parameters
    ----------
    learning_rate : float, optional
        Step size for each gradient descent update. Defaults to 0.01.
    max_iter : int, optional
        Maximum number of gradient descent iterations. Defaults to 1000.
    tol : float, optional
        Minimum absolute change in loss between consecutive iterations
        required to continue optimizing. Defaults to 1e-4.
    fit_intercept : bool, optional
        Whether to fit an intercept term b. Defaults to True.
    alpha : float, optional
        Regularization strength. Larger values push more coefficients
        exactly to zero. ``alpha=0`` recovers ordinary least squares.
        Defaults to 1.0.

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        Unlike Ridge, Lasso can shrink coefficients exactly to zero,
        since the L1 penalty applies a constant push regardless of a
        coefficient's magnitude. This makes it useful for automatic
        feature selection: features with a coefficient of exactly 0.0
        contribute nothing to the prediction.

    .. note::
        Because the subgradient update takes a fixed-size step
        (independent of the coefficient's magnitude), a learning rate
        or alpha that is too large can cause small coefficients to
        oscillate around zero instead of settling exactly on it. If
        coefficients don't stabilize, try reducing learning_rate.

    .. note::
        As with all gradient-descent-based linear models here, features
        should be standardized before fitting — otherwise the penalty is
        applied unevenly across features of different scales. The
        default ``alpha=1.0`` follows the same convention as
        scikit-learn's ``Lasso``, which assumes standardized inputs.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import LinearRegression, Lasso

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
        y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 1.5, size=30)

        ols = LinearRegression(max_iter=5000).fit(X, y)
        lasso = Lasso(alpha=5.0, max_iter=5000).fit(X, y)
        X_line = np.linspace(0, 10, 100).reshape(-1, 1)

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.6, label="Training data")
        ax.plot(X_line, ols.predict(X_line), color="red", label="OLS")
        ax.plot(X_line, lasso.predict(X_line), color="green", label="Lasso (alpha=5.0)")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("Lasso vs. OLS fit")
        ax.legend()
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        alpha: float = 1.0,
    ) -> None:
        """Initialize this regressor with gradient descent and regularization hyperparameters.

        Parameters
        ----------
        learning_rate : float, optional
            Step size for each gradient descent update. Defaults to 0.01.
        max_iter : int, optional
            Maximum number of gradient descent iterations. Defaults to 1000.
        tol : float, optional
            Minimum absolute change in loss between consecutive iterations
            required to continue optimizing. Defaults to 1e-4.
        fit_intercept : bool, optional
            Whether to fit an intercept term b. Defaults to True.
        alpha : float, optional
            Regularization strength; must be non-negative. Defaults to 1.0.
        """
        super().__init__(
            learning_rate=learning_rate, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept
        )
        self.alpha: float = alpha

    def _penalty_loss(self) -> float:
        """Compute the L1 penalty term, alpha times the sum of absolute coefficients.

        Returns
        -------
        float
            The value of alpha * sum(abs(``coef_``)).
        """
        return float(self.alpha * np.sum(np.abs(self.coef_)))

    def _penalty_gradient(self) -> _FloatArray:
        """Compute the subgradient of the L1 penalty with respect to ``coef_``.

        Returns
        -------
        FloatArray
            The value of alpha * sign(``coef_``), of shape (n_features,).
            sign(0) is taken as 0.
        """
        return self.alpha * np.sign(self.coef_)
