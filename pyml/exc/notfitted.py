r"""NotFittedError exception for unfitted estimator access.

This module defines the exception raised when a prediction or
transformation method is called on an estimator that has not
yet been fitted to training data.

Classes
-------
NotFittedError
    Exception raised when an unfitted estimator is used.

Examples
--------
>>> from pyml import LinearRegression
>>> from pyml.exc import NotFittedError
>>> import numpy as np
>>>
>>> model = LinearRegression()
>>> try:
...     model.predict(np.array([[1., 2.]]))
... except NotFittedError as e:
...     print(e)
LinearRegression is not fitted yet. Call fit() before using predict().
"""

from .base import PymlError


class NotFittedError(PymlError):
    r"""Exception raised when an estimator is used before calling fit().

    This exception is thrown by the ``_check_fitted`` method in
    :class:`BasicLinearModel` and any other estimator that requires
    fitting before making predictions or transformations.

    Parameters
    ----------
    estimator : object
        The unfitted estimator instance. Its class name is used
        in the error message.
    before : str, default='predict()'
        Name of the method that was called before fitting, used
        in the error message to guide the user.

    Examples
    --------
    >>> from pyml.exc import NotFittedError
    >>> from pyml import Kmeans
    >>>
    >>> model = Kmeans(k=3)
    >>> try:
    ...     model.predict([[1., 2.]])
    ... except NotFittedError as e:
    ...     print(f"Error: {e}")
    Error: Kmeans is not fitted yet. Call fit() before using predict().
    """

    def __init__(self, estimator: object, before: str = "predict()") -> None:
        r"""Initialize the NotFittedError with the offending estimator.

        Parameters
        ----------
        estimator : object
            The unfitted estimator instance that triggered the error.
        before : str, default='predict()'
            Name of the method that was called before fitting, included
            in the error message.

        Returns
        -------
        None
        """
        name = type(estimator).__name__
        super().__init__(f"{name} is not fitted yet. Call fit() before using {before}.")
