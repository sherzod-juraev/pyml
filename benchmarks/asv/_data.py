"""Data builders shared by the benchmarks.

Two versions of pyml are compared only if both are timed on the same data, so every benchmark
creates its data here, from a fixed seed. The sizes are constants for the same reason: a
benchmark is named by its parameters, and changing a size drops its old results out of the
comparison.
"""

import numpy as np
import numpy.typing as npt

# Number of samples of the small and the large case.
N_SMALL = 500
N_LARGE = 5000
# The tree benchmark keeps 4000 as its large case, the results of ``fit`` are named by it.
N_TREE_LARGE = 4000

_SEED = 0

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int_]


def regression_data(n_samples: int, n_features: int, /) -> tuple[FloatArray, FloatArray]:
    """Create a linear data set with noise: ``y = X @ coef + 1 + noise``."""
    rng = np.random.default_rng(_SEED)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=n_features)
    y = X @ coef + 1.0 + rng.normal(0.0, 0.5, size=n_samples)
    return X, y


def nonlinear_regression_data(n_samples: int, n_features: int, /) -> tuple[FloatArray, FloatArray]:
    """Create a non-linear data set with noise: ``y = X[:, 0] ** 2 + X[:, 1] + noise``."""
    rng = np.random.default_rng(_SEED)
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] ** 2 + X[:, 1] + rng.normal(0.0, 0.1, size=n_samples)
    return X, y


def classification_data(n_samples: int, n_features: int, /) -> tuple[FloatArray, IntArray]:
    """Create two overlapping classes, the sign of ``X @ coef + noise``."""
    rng = np.random.default_rng(_SEED)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=n_features)
    noise = rng.normal(0.0, 1.0, size=n_samples)
    y = (X @ coef + noise > 0.0).astype(int)
    return X, y
