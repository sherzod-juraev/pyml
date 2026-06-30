r"""Custom exception classes for the pyml library.

This subpackage defines exceptions used throughout pyml to signal
error conditions specific to machine learning workflows.

Available exceptions
--------------------
- :class:`PymlError` — Base exception for all pyml-specific errors.
  All custom exceptions inherit from this class, enabling users to
  catch any library error with a single ``except PymlError`` clause.
- :class:`NotFittedError` — Raised when an estimator's ``predict``,
  ``transform``, or ``inverse_transform`` method is called before
  the model has been fitted to training data.

Examples
--------
>>> from pyml.exc import PymlError, NotFittedError
>>>
>>> # Catching all pyml errors
>>> try:
...     raise NotFittedError("model")
... except PymlError as e:
...     print(f"Caught: {e}")
Caught: model
>>>
>>> # Catching specific error
>>> from pyml import LinearRegression
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
from .notfitted import NotFittedError

__all__ = [
    "NotFittedError",
    "PymlError",
]
