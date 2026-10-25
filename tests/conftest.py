"""Shared pytest fixtures for the pyml test suite.

Provides a seeded NumPy random generator so that tests using random
data are reproducible across runs, and small dataset builders for the
two shapes of data most model tests need: a linear regression problem
and two well-separated Gaussian blobs.
"""

from collections.abc import Callable

import numpy as np
import pytest

LinearDataBuilder = Callable[..., tuple[np.ndarray, np.ndarray]]
TwoBlobsBuilder = Callable[..., tuple[np.ndarray, np.ndarray]]


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a NumPy random generator seeded for reproducible tests.

    Returns
    -------
    numpy.random.Generator
        A fresh generator seeded with 42, created anew for each test
        that requests it.
    """
    return np.random.default_rng(seed=42)


@pytest.fixture
def linear_data(rng: np.random.Generator) -> LinearDataBuilder:
    """Return a builder of noise-free linear regression problems.

    The builder draws ``X`` uniformly from ``[-5, 5]`` and computes
    ``y = X @ coef + intercept``. It consumes the shared ``rng``, so a
    test that also uses ``rng`` directly sees a continued stream.

    Returns
    -------
    callable
        ``builder(n_samples=200, coef=(3.0, -2.0), intercept=5.0)``
        returning ``(X, y)``.
    """

    def build(
        n_samples: int = 200,
        coef: tuple[float, ...] = (3.0, -2.0),
        intercept: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        true_coef = np.asarray(coef, dtype=np.float64)
        X = rng.uniform(-5, 5, size=(n_samples, true_coef.size))
        y = X @ true_coef + intercept
        return X, y

    return build


@pytest.fixture
def two_blobs(rng: np.random.Generator) -> TwoBlobsBuilder:
    """Return a builder of two linearly separable Gaussian blobs.

    Class 0 is centred at ``-center`` and class 1 at ``+center`` in
    every feature. It consumes the shared ``rng``, class 0 first.

    Returns
    -------
    callable
        ``builder(n_per_class=50, center=3.0, scale=0.5, n_features=2)``
        returning ``(X, y)`` with integer labels ``0`` and ``1``.
    """

    def build(
        n_per_class: int = 50,
        center: float = 3.0,
        scale: float = 0.5,
        n_features: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        X0 = rng.normal(loc=(-center,) * n_features, scale=scale, size=(n_per_class, n_features))
        X1 = rng.normal(loc=(center,) * n_features, scale=scale, size=(n_per_class, n_features))
        X = np.vstack([X0, X1])
        y = np.array([0] * n_per_class + [1] * n_per_class)
        return X, y

    return build
