"""Ridge classifier (L2-regularized logistic regression)."""

import numpy as np

from ._logistic_base import _FloatArray, _LogisticRegressionBase


class RidgeClassifier(_LogisticRegressionBase):
    r"""Logistic regression with L2 regularization, fit via gradient descent.

    Predicts the probability of the positive class identically to
    :class:`~pyml.linear_model.logistic_regression.LogisticRegression`:

    .. math::
        \hat{p} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}

    Fitting minimizes binary cross-entropy loss plus an L2 penalty on
    the coefficients:

    .. math::
        J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{p}_i)
            + (1 - y_i) \log(1 - \hat{p}_i) \right]
            + \alpha \sum_{j=1}^{n} w_j^2

    The intercept b is not penalized, since it only shifts the decision
    boundary and does not contribute to model complexity.

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
        Regularization strength. Larger values shrink the coefficients
        more aggressively toward zero (but never exactly to zero).
        ``alpha=0`` recovers plain LogisticRegression. Defaults to 1.0.

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        L2 regularization is especially useful here when features are
        correlated or when classes are close to linearly separable,
        where unregularized logistic regression can push coefficients
        toward unbounded magnitudes while still reducing loss.

    .. note::
        As with all gradient-descent-based models here, features should
        be standardized before fitting. The default ``alpha=1.0`` follows
        the same convention as scikit-learn's ``RidgeClassifier``.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import RidgeClassifier

        rng = np.random.default_rng(42)
        n = 100
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        model = RidgeClassifier(alpha=1.0, max_iter=5000).fit(X, y)

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
        ax.set_title("RidgeClassifier decision boundary")
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
