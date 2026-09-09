"""Tests for Ridge."""

import numpy as np

from pyml.linear_model import LinearRegression, Ridge


class TestFit:
    def test_recovers_approximate_coefficients_with_low_alpha(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        true_coef = np.array([3.0, -2.0])
        y = X @ true_coef + 5.0

        model = Ridge(alpha=0.01, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(model.coef_, true_coef, atol=0.1)

    def test_higher_alpha_shrinks_coefficients_more(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        y = X @ np.array([3.0, -2.0]) + 5.0

        low_alpha = Ridge(alpha=0.1, learning_rate=0.05, max_iter=5000).fit(X, y)
        high_alpha = Ridge(alpha=10.0, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.sum(high_alpha.coef_**2) < np.sum(low_alpha.coef_**2)

    def test_never_zeroes_out_coefficients(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        y = X @ np.array([3.0, -2.0]) + 5.0

        model = Ridge(alpha=20.0, learning_rate=0.001, max_iter=20000).fit(X, y)

        assert np.all(model.coef_ != 0.0)

    def test_zero_alpha_matches_linear_regression(self, rng):
        X = rng.uniform(-5, 5, size=(100, 2))
        y = X @ np.array([1.0, 2.0]) + 1.0

        ridge = Ridge(alpha=0.0, learning_rate=0.05, max_iter=5000).fit(X, y)
        ols = LinearRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(ridge.coef_, ols.coef_, atol=1e-6)
