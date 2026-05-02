import numpy as np
from mlkit.exc import NotFitted


class LinReg:
    """
    Linear Regression using Batch Gradient Descent.

    Parameters
    ----------
    eta : float, default=0.1
        Learning rate controlling the step size at each iteration.
    max_iter : int, default=100
        Maximum number of gradient descent iterations.
    tol : float, default=1e-3
        Relative tolerance for convergence. Training stops early
        when the relative change in loss falls below this threshold.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Learned weights after fitting.
    intercept_ : float
        Learned bias term after fitting.
    Examples
    --------
    >>> model = LinReg(eta=0.01, max_iter=500)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinReg':
        """
          Fit the model to training data using batch gradient descent.

          Parameters
          ----------
          X : np.ndarray of shape (n_samples, n_features)
              Training feature matrix.
          y : np.ndarray of shape (n_samples,)
              Target values.

          Returns
          -------
          self : LinReg
              Fitted estimator.
          """

        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0
        J_old = np.inf
        n = X.shape[0]
        for _ in range(self.max_iter):
            y_pred = self.__cal_y(X)
            error = y - y_pred
            self.coef_ += self.eta * (2 / n) * X.T @ error
            self.intercept_ += self.eta * (2 / n) * np.sum(error)
            J_new = np.mean(error ** 2)
            if np.isinf(J_old):
                J_old = J_new
                continue
            if J_old == 0 or \
                    np.abs(J_new - J_old) / J_old <= self.tol:
                break
            J_old = J_new
        self.__fitted = True
        return self

    def __cal_y(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions using current weights and bias.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Linear combination of inputs and weights.
        """

        return X @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
         Predict target values for input data.

         Parameters
         ----------
         X : np.ndarray of shape (n_samples, n_features)
             Input feature matrix.

         Returns
         -------
         y_pred : np.ndarray of shape (n_samples,)
             Predicted target values.

         Raises
         ------
         NotFitted
             If called before fitting the model.
         """

        if not self.__fitted:
            raise NotFitted('LinearRegression not fitted yet')
        return self.__cal_y(X)