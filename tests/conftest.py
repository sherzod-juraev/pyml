"""Shared pytest fixtures for the pyml test suite.

Provides a seeded NumPy random generator so that tests using random
data are reproducible across runs.
"""

import numpy as np
import pytest


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
