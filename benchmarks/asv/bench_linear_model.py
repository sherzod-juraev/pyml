"""Benchmarks of the linear models: the time of ``fit``.

Every case runs a fixed number of iterations (``tol=0.0``), so that two versions of pyml do the
same amount of work and only the cost of one iteration is compared.
"""

from typing import ClassVar

import numpy as np

from pyml.linear_model import (
    ElasticNetClassifier,
    LassoClassifier,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)

_N_FEATURES = 8
# The classifiers use as many features as the examples (breast cancer data).
_N_FEATURES_CLASSIFIER = 30
_CLASSIFIER_ITERATIONS = 5000
# Small on purpose: with ``alpha=0.1`` RidgeClassifier reaches an exactly equal loss after about
# 600 iterations and stops (``tol=0.0``), so it would time 12% of the work of the other models.
_ALPHA = 0.001


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


def _make_classifier(
    model: str,
) -> LogisticRegression | RidgeClassifier | LassoClassifier | ElasticNetClassifier:
    """Create a classifier of the logistic family by name, with the same settings for all."""
    learning_rate = 0.1
    max_iter = _CLASSIFIER_ITERATIONS
    if model == "LogisticRegression":
        return LogisticRegression(learning_rate=learning_rate, max_iter=max_iter, tol=0.0)
    if model == "RidgeClassifier":
        return RidgeClassifier(
            learning_rate=learning_rate, max_iter=max_iter, tol=0.0, alpha=_ALPHA
        )
    if model == "LassoClassifier":
        return LassoClassifier(
            learning_rate=learning_rate, max_iter=max_iter, tol=0.0, alpha=_ALPHA
        )
    return ElasticNetClassifier(
        learning_rate=learning_rate, max_iter=max_iter, tol=0.0, alpha=_ALPHA, l1_ratio=0.5
    )


class LogisticFamilyFit:
    """Time of ``fit`` of the four logistic models for a growing number of samples.

    The number of iterations is the one of the examples (5000), so that one case runs for
    about 0.1 s and is not lost in the noise.
    """

    params: ClassVar[list[list[int] | list[str]]] = [
        ["LogisticRegression", "RidgeClassifier", "LassoClassifier", "ElasticNetClassifier"],
        [500, 5000],
    ]
    param_names: ClassVar[list[str]] = ["model", "n_samples"]

    def setup(self, model: str, n_samples: int) -> None:
        """Create two overlapping classes. This is not timed."""
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(n_samples, _N_FEATURES_CLASSIFIER))
        coef = rng.normal(size=_N_FEATURES_CLASSIFIER)
        noise = rng.normal(0.0, 1.0, size=n_samples)
        self.y = (self.X @ coef + noise > 0.0).astype(int)

    def time_fit(self, model: str, n_samples: int) -> None:
        """Fit the classifier for 5000 iterations."""
        _make_classifier(model).fit(self.X, self.y)
