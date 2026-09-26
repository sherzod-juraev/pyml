"""Ordinary least squares linear regression."""

import numpy as np

from ._linear_base import _FloatArray, _LinearModelBase


class LinearRegression(_LinearModelBase):
    r"""Ordinary least squares linear regression, fit via gradient descent.

    Predicts a continuous target as a linear combination of features:

    .. math::
        \hat{y} = Xw + b

    Fitting minimizes the mean squared error between predictions and
    targets, with no regularization term:

    .. math::
        J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_i - y_i \right)^2

    The gradient of this loss with respect to w and b:

    .. math::
        \frac{\partial J}{\partial w} = \frac{2}{m} X^T (\hat{y} - y), \quad
        \frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)

    is used to update the parameters at each iteration via gradient
    descent:

    .. math::
        w \leftarrow w - \alpha \frac{\partial J}{\partial w}, \quad
        b \leftarrow b - \alpha \frac{\partial J}{\partial b}

    where alpha is the learning rate.

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

    Attributes
    ----------
    coef_ : FeatureMatrix
        Learned weight vector, of shape (n_features,).
    intercept_ : float
        Learned intercept term. Remains 0.0 if fit_intercept is False.
    n_iter_ : int
        Number of iterations actually run before stopping.


    .. note::
        With no regularization, OLS coefficients can become large or
        unstable when features are highly correlated (multicollinearity)
        or when there are more features than samples. Ridge or Lasso
        regression address this by penalizing large coefficients.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.linear_model import LinearRegression

        rng = np.random.default_rng(42)
        X = np.sort(rng.uniform(0, 10, size=(30, 1)), axis=0)
        y = 2.5 * X.ravel() + 1.0 + rng.normal(0, 1.5, size=30)

        model = LinearRegression(max_iter=5000).fit(X, y)
        X_line = np.linspace(0, 10, 100).reshape(-1, 1)
        y_line = model.predict(X_line)

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.6, label="Training data")
        ax.plot(X_line, y_line, color="red", label="Fitted line")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("LinearRegression fit")
        ax.legend(loc="best")
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize this regressor with the given gradient descent hyperparameters.

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
        """Return zero — ordinary least squares applies no penalty.

        Returns
        -------
        float
            Always 0.0.
        """
        return 0.0

    def _penalty_gradient(self) -> _FloatArray:
        """Return a zero vector — ordinary least squares applies no penalty.

        Returns
        -------
        FloatArray
            Array of zeros with the same shape as ``coef_``.
        """
        return np.zeros_like(self.coef_)
