import numpy as np
from ml_collection.exception import NotFitted
from ml_collection.errors import Error


class LinearRegression:
    """
    Linear Regression model with support for batch, stochastic, and mini-batch gradient descent.

    Parameters
    ----------
    eta : float, default=0.1
        Learning rate for gradient updates.
    max_iter : int, default=100
        Maximum number of iterations for training.
    tol : float, default=1e-3
        Tolerance for convergence. Training stops if the change in loss is below this threshold.

    Attributes
    ----------
    coef : np.ndarray
        Coefficients (weights) of the linear model after fitting.
    intercept : float
        Intercept (bias) term of the linear model after fitting.
    __fitted : bool
        Flag indicating whether the model has been fitted.
    """

    def __init__(
            self,
            eta: float = 1e-1,
            max_iter: int = 100,
            tol: float = 1e-3
    ):

        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.__fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model to the training data.

        Dispatches to the appropriate gradient descent mode (batch, stochastic, or mini-batch).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            Fitted estimator.
        """

        self.coef = np.zeros(X.shape[1])
        self.intercept = 0
        J_last, J_old = None, None
        n = X.shape[0]
        for _ in range(self.max_iter):
            y_pred = self.__cal_y(X)
            error = y - y_pred
            self.coef += self.eta * (2 / n) * X.T @ error
            self.intercept += self.eta * (2 / n) * np.sum(error)
            J_last = Error.mse(y, y_pred)
            if J_old is not None and np.abs(J_last - J_old) <= self.tol:
                break
            J_old = J_last
        self.__fitted = True
        return self

    def __cal_y(self, X: np.ndarray) -> np.ndarray:
        """
         Calculate predicted values for given input using current coefficients.

         Parameters
         ----------
         x : np.ndarray
             Input data (single sample or batch).

         Returns
         -------
         y_pred : np.ndarray
             Predicted target values.
         """

        return X @ self.coef + self.intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input data using the fitted model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted target values.

        Raises
        ------
        NotFitted
            If the model has not been fitted yet.
        """

        if not self.__fitted:
            raise NotFitted('LinearRegression not fitted yet')
        return self.__cal_y(X)