"""Tests for RidgeClassifier."""

import numpy as np

from pyml.linear_model import LogisticRegression, RidgeClassifier


class TestFit:
    def test_separates_linearly_separable_classes(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = RidgeClassifier(alpha=0.01, learning_rate=0.1, max_iter=5000).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_higher_alpha_shrinks_coefficients_more(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        low_alpha = RidgeClassifier(alpha=0.1, learning_rate=0.05, max_iter=5000).fit(X, y)
        high_alpha = RidgeClassifier(alpha=5.0, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.sum(high_alpha.coef_**2) < np.sum(low_alpha.coef_**2)

    def test_never_zeroes_out_coefficients(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = RidgeClassifier(alpha=5.0, learning_rate=0.01, max_iter=10000).fit(X, y)

        assert np.all(model.coef_ != 0.0)

    def test_zero_alpha_matches_logistic_regression(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        ridge = RidgeClassifier(alpha=0.0, learning_rate=0.05, max_iter=5000).fit(X, y)
        logistic = LogisticRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(ridge.coef_, logistic.coef_, atol=1e-6)
