r"""Feature preprocessing and scaling utilities.

This subpackage provides pure NumPy implementations of common
feature scaling techniques for normalizing data before feeding
it to machine learning models.

Available scalers
-----------------
- :class:`MinMaxScaler` — Scale features to a fixed [0, 1]
  range using min-max normalization.
- :class:`StandardScaler` — Standardize features to zero mean
  and unit variance (Z-score normalization).
- :class:`RobustScaler` — Scale features using median and
  interquartile range (IQR), robust to outliers.

Common interface
----------------
All scalers implement the scikit-learn transformer API:

- ``fit(X)`` — Compute scaling statistics from training data.
- ``transform(X)`` — Apply scaling to new data.
- ``fit_transform(X)`` — Fit and transform in one step.
- ``inverse_transform(X)`` — Reverse the scaling back to
  original space.

Notes
-----
All scalers handle constant features (zero variance) gracefully
by avoiding division by zero. Choose the appropriate scaler based
on your data characteristics:

- Use :class:`StandardScaler` when data is approximately normal.
- Use :class:`MinMaxScaler` when a fixed range is required.
- Use :class:`RobustScaler` when data contains significant outliers.

Examples
--------
>>> from pyml import StandardScaler, MinMaxScaler, RobustScaler
>>> import numpy as np
>>>
>>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
>>>
>>> scaler = StandardScaler()
>>> X_scaled = scaler.fit_transform(X)
>>> X_scaled
array([[-1.224..., -1.224...],
       [ 0.     ,  0.     ],
       [ 1.224...,  1.224...]])
>>>
>>> X_original = scaler.inverse_transform(X_scaled)
"""

from .scalers import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

__all__ = [
    "MinMaxScaler",
    "RobustScaler",
    "StandardScaler",
]
