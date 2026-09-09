"""Tests for LassoClassifier."""

import numpy as np

from pyml.linear_model import LassoClassifier, LogisticRegression


class TestFit:
    def test_separates_linearly_separable_classes(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = LassoClassifier(alpha=0.01, learning_rate=0.1, max_iter=5000).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_high_alpha_can_zero_out_a_coefficient(self, rng):
        # feature 0 is informative, feature 1 is pure noise
        n = 100
        feature0 = np.concatenate([rng.normal(-3, 0.5, n // 2), rng.normal(3, 0.5, n // 2)])
        feature1 = rng.normal(0, 1.0, n)
        X = np.column_stack([feature0, feature1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        model = LassoClassifier(alpha=0.5, learning_rate=0.01, max_iter=5000).fit(X, y)

        assert abs(model.coef_[1]) < 0.05

    def test_zero_alpha_matches_logistic_regression(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        lasso = LassoClassifier(alpha=0.0, learning_rate=0.05, max_iter=5000).fit(X, y)
        logistic = LogisticRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(lasso.coef_, logistic.coef_, atol=1e-6)
