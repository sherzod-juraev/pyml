"""Ridge regression (L2-regularized linear regression)."""

import numpy as np

from ._linear_base import _FloatArray, _LinearModelBase


class Ridge(_LinearModelBase):
    r"""Linear regression with L2 (Tikhonov) regularization, fit via gradient descent.

    Predicts a continuous target as a linear combination of features,
    identically to :class:`~pyml.linear_model.linear_regression.LinearRegression`:

    .. math::
        \hat{y} = Xw + b

    Fitting minimizes the mean squared error plus an L2 penalty on the
    coefficients:

    .. math::
        J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_i - y_i \right)^2
            + \alpha \sum_{j=1}^{n} w_j^2

    The intercept b is not penalized, since it only shifts the data and
    does not contribute to model complexity. The gradient of this loss
    with respect to w and b:

    .. math::
        \frac{\partial J}{\partial w} = \frac{2}{m} X^T (\hat{y} - y) + 2 \alpha w, \quad
        \frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)

    is used to update the parameters at each iteration via gradient
    descent:

    .. math::
        w \leftarrow w - \alpha_{lr} \frac{\partial J}{\partial w}, \quad
        b \leftarrow b - \alpha_{lr} \frac{\partial J}{\partial b}

    where alpha_lr is the learning rate (distinct from the
    regularization strength alpha above).

    Parameters
    ----------
    learning_rate : float, optional
        Step size for each gradient descent update.
    max_iter : int, optional
        Maximum number of gradient descent iterations.
    tol : float, optional
        Minimum absolute change in loss between consecutive iterations
        required to continue optimizing.
    fit_intercept : bool, optional
        Whether to fit an intercept term b.
    alpha : float, optional
        Regularization strength. Larger values shrink the coefficients
        more aggressively toward zero (but never exactly to zero).
        ``alpha=0`` recovers ordinary least squares.

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        Ridge shrinks coefficients smoothly toward zero but never sets
        them exactly to zero, unlike Lasso. It is most useful when
        features are correlated (multicollinearity), where it stabilizes
        the solution that OLS would otherwise leave ill-conditioned.

    .. note::
        As with all gradient-descent-based linear models here, features
        should be standardized before fitting — otherwise the penalty is
        applied unevenly across features of different scales. The
        default ``alpha=1.0`` follows the same convention as
        scikit-learn's ``Ridge``, which assumes standardized inputs.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import LinearRegression, Ridge

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
        y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 1.5, size=30)

        ols = LinearRegression(max_iter=5000).fit(X, y)
        ridge = Ridge(alpha=5.0, max_iter=5000).fit(X, y)
        X_line = np.linspace(0, 10, 100).reshape(-1, 1)

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.6, label="Training data")
        ax.plot(X_line, ols.predict(X_line), color="red", label="OLS")
        ax.plot(X_line, ridge.predict(X_line), color="blue", label="Ridge (alpha=5.0)")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("Ridge vs. OLS fit")
        ax.legend(loc="best")
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
        """Compute the L2 penalty term, alpha times the sum of squared coefficients.

        Returns
        -------
        float
            The value of alpha * sum(``coef_`` ** 2).
        """
        return float(self.alpha * np.sum(self.coef_**2))

    def _penalty_gradient(self) -> _FloatArray:
        """Compute the gradient of the L2 penalty with respect to ``coef_``.

        Returns
        -------
        FloatArray
            The value of 2 * alpha * ``coef_``, of shape (n_features,).
        """
        return 2 * (self.alpha * self.coef_)
