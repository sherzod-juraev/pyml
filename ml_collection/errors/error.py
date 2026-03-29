import numpy as np


class Error:
    """
    Static class providing common regression error metrics.
    """

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted target values.

        Returns
        -------
        float
            Mean squared error.
        """

        error =  np.mean((y_true - y_pred) ** 2)
        return error

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted target values.

        Returns
        -------
        float
            Root mean squared error.
        """

        return np.sqrt(Error.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Mean Absolute Error (MAE) between true and predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted target values.

        Returns
        -------
        float
            Mean absolute error.
        """

        error = np.mean(np.abs(y_true - y_pred))
        return error