"""Tests for Lasso."""

import numpy as np

from pyml.linear_model import Lasso, LinearRegression


class TestFit:
    def test_recovers_approximate_coefficients_with_low_alpha(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        true_coef = np.array([3.0, -2.0])
        y = X @ true_coef + 5.0

        model = Lasso(alpha=0.01, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(model.coef_, true_coef, atol=0.2)

    def test_high_alpha_can_zero_out_a_coefficient(self, rng):
        # feature 1 is informative, feature 2 is pure noise
        X = rng.uniform(-5, 5, size=(200, 2))
        y = 3.0 * X[:, 0] + 5.0 + rng.normal(0, 0.1, size=200)

        model = Lasso(alpha=2.0, learning_rate=0.01, max_iter=5000).fit(X, y)

        assert abs(model.coef_[1]) < 0.05

    def test_zero_alpha_matches_linear_regression(self, rng):
        X = rng.uniform(-5, 5, size=(100, 2))
        y = X @ np.array([1.0, 2.0]) + 1.0

        lasso = Lasso(alpha=0.0, learning_rate=0.05, max_iter=5000).fit(X, y)
        ols = LinearRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(lasso.coef_, ols.coef_, atol=1e-6)
