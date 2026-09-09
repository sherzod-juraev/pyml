"""Elastic Net regression (combined L1/L2-regularized linear regression)."""

import numpy as np

from ._linear_base import _FloatArray, _LinearModelBase


class ElasticNet(_LinearModelBase):
    r"""Linear regression with combined L1 and L2 regularization, fit via gradient descent.

    Predicts a continuous target as a linear combination of features,
    identically to :class:`~pyml.linear_model.linear_regression.LinearRegression`:

    .. math::
        \hat{y} = Xw + b

    Fitting minimizes the mean squared error plus a weighted mix of L1
    and L2 penalties on the coefficients:

    .. math::
        J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_i - y_i \right)^2
            + \alpha \left( \rho \sum_{j=1}^{n} |w_j|
            + \frac{1 - \rho}{2} \sum_{j=1}^{n} w_j^2 \right)

    where rho (l1_ratio) controls the mix: rho=1 recovers pure Lasso,
    rho=0 recovers pure Ridge. The intercept b is not penalized. As with
    Lasso, the L1 term is not differentiable at w_j = 0, so a subgradient
    is used:

    .. math::
        \frac{\partial J}{\partial w} = \frac{2}{m} X^T (\hat{y} - y)
            + \alpha \rho \cdot \text{sign}(w) + \alpha (1 - \rho) w, \quad
        \frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)

    where sign(w) is applied elementwise and sign(0) = 0. This gradient
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
        Step size for each gradient descent update. Defaults to 0.01.
    max_iter : int, optional
        Maximum number of gradient descent iterations. Defaults to 1000.
    tol : float, optional
        Minimum absolute change in loss between consecutive iterations
        required to continue optimizing. Defaults to 1e-4.
    fit_intercept : bool, optional
        Whether to fit an intercept term b. Defaults to True.
    alpha : float, optional
        Overall regularization strength. ``alpha=0`` recovers ordinary
        least squares. Defaults to 1.0.
    l1_ratio : float, optional
        Mixing parameter between L1 and L2 penalties, between 0 and 1.
        ``l1_ratio=1`` is pure Lasso, ``l1_ratio=0`` is pure Ridge.
        Defaults to 0.5.

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        Elastic Net combines Ridge's smooth shrinkage with Lasso's
        ability to zero out coefficients entirely. It is particularly
        useful when features are correlated: Lasso alone tends to
        arbitrarily pick one feature from a correlated group and zero
        out the rest, while Elastic Net's L2 component encourages
        correlated features to be shrunk together.

    .. note::
        As with all gradient-descent-based linear models here, features
        should be standardized before fitting. The defaults ``alpha=1.0``
        and ``l1_ratio=0.5`` follow the same convention as scikit-learn's
        ``ElasticNet``.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import ElasticNet, LinearRegression

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
        y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 1.5, size=30)

        ols = LinearRegression(max_iter=5000).fit(X, y)
        elastic = ElasticNet(alpha=5.0, l1_ratio=0.5, max_iter=5000).fit(X, y)
        X_line = np.linspace(0, 10, 100).reshape(-1, 1)

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.6, label="Training data")
        ax.plot(X_line, ols.predict(X_line), color="red", label="OLS")
        ax.plot(X_line, elastic.predict(X_line), color="purple", label="ElasticNet (alpha=5.0)")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("ElasticNet vs. OLS fit")
        ax.legend()
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
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
            Overall regularization strength; must be non-negative. Defaults
            to 1.0.
        l1_ratio : float, optional
            Mixing parameter between L1 and L2 penalties, between 0 and 1.
            Defaults to 0.5.
        """
        super().__init__(
            learning_rate=learning_rate, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept
        )
        self.alpha: float = alpha
        self.l1_ratio: float = l1_ratio

    def _penalty_loss(self) -> float:
        r"""Compute the combined L1/L2 penalty term.

        Returns
        -------
        float
            The value of alpha * (l1_ratio * sum(abs(``coef_``)) +
            (1 - l1_ratio) / 2 * sum(``coef_`` ** 2)).
        """
        return float(
            self.alpha
            * (
                self.l1_ratio * np.sum(np.abs(self.coef_))
                + ((1 - self.l1_ratio) / 2) * np.sum(self.coef_**2)
            )
        )

    def _penalty_gradient(self) -> _FloatArray:
        r"""Compute the subgradient of the combined L1/L2 penalty with respect to ``coef_``.

        Returns
        -------
        FloatArray
            The value of alpha * (l1_ratio * sign(``coef_``) +
            (1 - l1_ratio) * ``coef_``), of shape (n_features,). sign(0) is
            taken as 0.
        """
        l1_term = self.alpha * (self.l1_ratio * np.sign(self.coef_))
        l2_term = self.alpha * ((1 - self.l1_ratio) * self.coef_)
        return l1_term + l2_term
