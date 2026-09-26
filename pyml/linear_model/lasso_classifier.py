"""Lasso classifier (L1-regularized logistic regression)."""

import numpy as np

from ._logistic_base import _FloatArray, _LogisticRegressionBase


class LassoClassifier(_LogisticRegressionBase):
    r"""Logistic regression with L1 regularization, fit via gradient descent.

    Predicts the probability of the positive class identically to
    :class:`~pyml.linear_model.logistic_regression.LogisticRegression`:

    .. math::
        \hat{p} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}

    Fitting minimizes binary cross-entropy loss plus an L1 penalty on
    the coefficients:

    .. math::
        J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{p}_i)
            + (1 - y_i) \log(1 - \hat{p}_i) \right]
            + \alpha \sum_{j=1}^{n} |w_j|

    The intercept b is not penalized, since it only shifts the decision
    boundary and does not contribute to model complexity. As with
    :class:`~pyml.linear_model.Lasso`, :math:`|w_j|` is not differentiable at
    :math:`w_j = 0`, so a subgradient is used:

    .. math::
        \frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{p} - y)
            + \alpha \cdot \text{sign}(w), \quad
        \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{p}_i - y_i)

    where sign(w) is applied elementwise and sign(0) = 0.

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
        Regularization strength. Larger values push more coefficients
        exactly to zero. ``alpha=0`` recovers plain LogisticRegression.

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        Unlike RidgeClassifier, LassoClassifier can shrink coefficients
        exactly to zero, making it useful for automatic feature
        selection: features with a coefficient of exactly 0.0 contribute
        nothing to the decision boundary.

    .. note::
        Because the subgradient update takes a fixed-size step
        (independent of the coefficient's magnitude), a learning rate or
        alpha that is too large can cause small coefficients to
        oscillate around zero instead of settling exactly on it. If
        coefficients don't stabilize, try reducing learning_rate.

    .. note::
        As with all gradient-descent-based models here, features should
        be standardized before fitting. The default ``alpha=1.0`` follows
        the same convention as scikit-learn's ``Lasso``.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import LassoClassifier

        rng = np.random.default_rng(42)
        n = 100
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        model = LassoClassifier(alpha=0.5, max_iter=5000).fit(X, y)

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
        ax.set_title("LassoClassifier decision boundary")
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
