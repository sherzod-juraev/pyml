"""Elastic Net classifier (combined L1/L2-regularized logistic regression)."""

import numpy as np

from ._logistic_base import _FloatArray, _LogisticRegressionBase


class ElasticNetClassifier(_LogisticRegressionBase):
    r"""Logistic regression with combined L1 and L2 regularization, fit via gradient descent.

    Predicts the probability of the positive class identically to
    :class:`~pyml.linear_model.logistic_regression.LogisticRegression`:

    .. math::
        \hat{p} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}

    Fitting minimizes binary cross-entropy loss plus a weighted mix of
    L1 and L2 penalties on the coefficients:

    .. math::
        J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{p}_i)
            + (1 - y_i) \log(1 - \hat{p}_i) \right]
            + \alpha \left( \rho \sum_{j=1}^{n} |w_j|
            + \frac{1 - \rho}{2} \sum_{j=1}^{n} w_j^2 \right)

    where rho (l1_ratio) controls the mix: rho=1 recovers pure
    LassoClassifier, rho=0 recovers pure RidgeClassifier. The intercept
    b is not penalized. As with LassoClassifier, the L1 term is not
    differentiable at w_j = 0, so a subgradient is used:

    .. math::
        \frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{p} - y)
            + \alpha \rho \cdot \text{sign}(w) + \alpha (1 - \rho) w, \quad
        \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{p}_i - y_i)

    where sign(w) is applied elementwise and sign(0) = 0.

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
        Overall regularization strength. ``alpha=0`` recovers plain
        LogisticRegression. Defaults to 1.0.
    l1_ratio : float, optional
        Mixing parameter between L1 and L2 penalties, between 0 and 1.
        ``l1_ratio=1`` is pure LassoClassifier, ``l1_ratio=0`` is pure
        RidgeClassifier. Defaults to 0.5.

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        Elastic Net combines RidgeClassifier's smooth shrinkage with
        LassoClassifier's ability to zero out coefficients entirely. It
        is particularly useful when features are correlated: Lasso alone
        tends to arbitrarily pick one feature from a correlated group
        and zero out the rest, while the L2 component here encourages
        correlated features to be shrunk together.

    .. note::
        As with all gradient-descent-based models here, features should
        be standardized before fitting. The defaults ``alpha=1.0`` and
        ``l1_ratio=0.5`` follow the same convention as scikit-learn's
        ``ElasticNet``.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import ElasticNetClassifier

        rng = np.random.default_rng(42)
        n = 100
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        model = ElasticNetClassifier(alpha=0.5, l1_ratio=0.5, max_iter=5000).fit(X, y)

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        probs = model.predict_proba(grid).reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.6)
        ax.scatter(X0[:, 0], X0[:, 1], label="Class 0", edgecolor="k")
        ax.scatter(X1[:, 0], X1[:, 1], label="Class 1", edgecolor="k")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("ElasticNetClassifier decision boundary")
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
        """Initialize this classifier with gradient descent and regularization hyperparameters.

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
