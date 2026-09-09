"""Logistic regression for binary classification."""

import numpy as np

from ._logistic_base import _FloatArray, _LogisticRegressionBase


class LogisticRegression(_LogisticRegressionBase):
    r"""Logistic regression for binary classification, fit via gradient descent.

    Predicts the probability of the positive class as a sigmoid of a
    linear combination of features:

    .. math::
        \hat{p} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}

    Fitting minimizes binary cross-entropy loss between predicted
    probabilities and true labels, with no regularization term:

    .. math::
        J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{p}_i)
            + (1 - y_i) \log(1 - \hat{p}_i) \right]

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

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        With no regularization, coefficients can become large or
        unstable when features are highly correlated (multicollinearity)
        or when classes are perfectly (or near-perfectly) separable, in
        which case the loss can be driven toward zero by arbitrarily
        large weights. RidgeClassifier or LassoClassifier address this
        by penalizing large coefficients.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import LogisticRegression

        rng = np.random.default_rng(42)
        n = 100
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        model = LogisticRegression(max_iter=5000).fit(X, y)

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
        ax.set_title("LogisticRegression decision boundary")
        ax.legend()
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize this classifier with the given gradient descent hyperparameters.

        Parameters
        ----------
        learning_rate : float, optional
            Step size alpha for each gradient descent update. Defaults to
            0.01.
        max_iter : int, optional
            Maximum number of gradient descent iterations. Defaults to 1000.
        tol : float, optional
            Minimum absolute change in loss between consecutive iterations
            required to continue optimizing. Defaults to 1e-4.
        fit_intercept : bool, optional
            Whether to fit an intercept term b. Defaults to True.
        """
        super().__init__(
            learning_rate=learning_rate, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept
        )

    def _penalty_loss(self) -> float:
        """Return zero — plain logistic regression applies no penalty.

        Returns
        -------
        float
            Always 0.0.
        """
        return 0.0

    def _penalty_gradient(self) -> _FloatArray:
        """Return a zero vector — plain logistic regression applies no penalty.

        Returns
        -------
        FloatArray
            Array of zeros with the same shape as ``coef_``.
        """
        return np.zeros_like(self.coef_)
