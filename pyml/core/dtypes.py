"""Type aliases for feature matrices and prediction/cluster targets used across pyml.

These aliases exist purely for readability and documentation — at the
type-checker level several of them resolve to the same underlying
NumPy dtype categories, but naming them separately clarifies intent
at call sites such as ``fit(X: FeatureMatrix, y: RegressionTarget)``
or ``fit_predict(X: FeatureMatrix) -> ClusterLabels``.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

FeatureMatrix = npt.NDArray[np.floating[Any]]
"""2D array of shape (n_samples, n_features) with floating dtype."""

ClassificationTarget = npt.NDArray[np.integer[Any]]
"""1D array of integer-encoded class labels, shape (n_samples,)."""

RegressionTarget = npt.NDArray[np.floating[Any]]
"""1D array of continuous target values, shape (n_samples,)."""

ClusterLabels = npt.NDArray[np.integer[Any]]
"""1D array of algorithm-assigned cluster indices, shape (n_samples,)."""
