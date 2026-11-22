"""Benchmarks of the linear models: the time of ``fit``.

Both cases run a fixed number of iterations (``tol=0.0``), so that two versions of pyml do the
same amount of work and only the cost of one iteration is compared.
"""

from typing import ClassVar

import numpy as np

from pyml.linear_model import LogisticRegression, Ridge

_N_FEATURES = 8


class RidgeFit:
    """Time of ``Ridge.fit`` for a growing number of samples."""

    params: ClassVar[list[list[int]]] = [[500, 5000]]
    param_names: ClassVar[list[str]] = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        """Create a linear data set with noise. This is not timed."""
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(n_samples, _N_FEATURES))
        coef = rng.normal(size=_N_FEATURES)
        self.y = self.X @ coef + 1.0 + rng.normal(0.0, 0.5, size=n_samples)

    def time_fit(self, n_samples: int) -> None:
        """Fit a ridge regression for 2000 iterations."""
        Ridge(alpha=0.1, learning_rate=0.01, max_iter=2000, tol=0.0).fit(self.X, self.y)


class LogisticRegressionFit:
    """Time of ``LogisticRegression.fit`` for a growing number of samples."""

    params: ClassVar[list[list[int]]] = [[500, 5000]]
    param_names: ClassVar[list[str]] = ["n_samples"]

    def setup(self, n_samples: int) -> None:
        """Create two overlapping classes. This is not timed."""
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(n_samples, _N_FEATURES))
        coef = rng.normal(size=_N_FEATURES)
        noise = rng.normal(0.0, 1.0, size=n_samples)
        self.y = (self.X @ coef + noise > 0.0).astype(int)

    def time_fit(self, n_samples: int) -> None:
        """Fit a logistic regression for 500 iterations."""
        LogisticRegression(learning_rate=0.1, max_iter=500, tol=0.0).fit(self.X, self.y)
