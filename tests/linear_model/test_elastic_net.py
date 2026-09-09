"""Tests for ElasticNet."""

import numpy as np

from pyml.linear_model import ElasticNet, Lasso, Ridge


class TestFit:
    def test_l1_ratio_one_behaves_like_lasso(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        y = X @ np.array([3.0, -2.0]) + 5.0

        elastic = ElasticNet(alpha=1.0, l1_ratio=1.0, learning_rate=0.05, max_iter=5000).fit(X, y)
        lasso = Lasso(alpha=1.0, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(elastic.coef_, lasso.coef_, atol=1e-2)

    def test_l1_ratio_zero_behaves_like_ridge(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        y = X @ np.array([3.0, -2.0]) + 5.0

        elastic = ElasticNet(alpha=1.0, l1_ratio=0.0, learning_rate=0.05, max_iter=5000).fit(X, y)
        # ElasticNet's L2 term is halved (alpha/2 * sum(w^2)) relative to
        # Ridge's alpha * sum(w^2), so alpha=1.0 here matches Ridge's alpha=0.5.
        ridge = Ridge(alpha=0.5, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(elastic.coef_, ridge.coef_, atol=1e-2)

    def test_recovers_approximate_coefficients_with_low_alpha(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        true_coef = np.array([3.0, -2.0])
        y = X @ true_coef + 5.0

        model = ElasticNet(alpha=0.01, l1_ratio=0.5, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(model.coef_, true_coef, atol=0.2)
