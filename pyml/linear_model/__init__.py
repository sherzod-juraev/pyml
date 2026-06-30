"""
Linear models for regression and classification.

This subpackage provides pure NumPy implementations of fundamental
linear models trained via Batch Gradient Descent. All models ultimately
inherit from BasicLinearModel and share a consistent
``fit`` / ``predict`` interface.

Available models
----------------
**Regression (continuous targets):**

- :class:`LinearRegression` — Ordinary Least Squares with optional
  L1/L2 regularization.
- :class:`Ridge` — Linear regression with L2 (Tikhonov)
  regularization for handling multicollinearity.
- :class:`Lasso` — Linear regression with L1 regularization for
  automatic feature selection via sparse coefficients.

**Classification (binary targets):**

- :class:`LogisticRegression` — Binary classifier with sigmoid
  activation and cross-entropy loss. Supports L1/L2
  regularization.

Common interface
----------------
All models implement:

- ``fit(X, y)`` — Train on feature matrix and target vector.
- ``predict(X)`` — Generate predictions for new samples.
- ``_check_fitted()`` — Validate that the model has been trained
  before prediction.

Notes
-----
All models use batch gradient descent rather than closed-form
solutions or coordinate descent. This choice prioritizes
educational clarity over computational efficiency.

Examples
--------
>>> from pyml import LinearRegression, Ridge, Lasso
>>> import numpy as np
>>>
>>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
>>> y = np.array([3., 7., 11.])
>>>
>>> model = LinearRegression(learning_rate=0.01, max_iter=500)
>>> model.fit(X, y)
>>> model.predict(np.array([[7., 8.]]))
array([15.])
"""

from .lasso import Lasso
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .ridge import Ridge

__all__ = [
    "Lasso",
    "LinearRegression",
    "LogisticRegression",
    "Ridge",
]
