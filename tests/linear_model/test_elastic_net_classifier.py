"""Tests for ElasticNetClassifier."""

import numpy as np

from pyml.linear_model import ElasticNetClassifier, LassoClassifier, RidgeClassifier


class TestFit:
    def test_l1_ratio_one_behaves_like_lasso_classifier(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        elastic = ElasticNetClassifier(
            alpha=0.5, l1_ratio=1.0, learning_rate=0.05, max_iter=5000
        ).fit(X, y)
        lasso = LassoClassifier(alpha=0.5, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(elastic.coef_, lasso.coef_, atol=1e-2)

    def test_l1_ratio_zero_behaves_like_ridge_classifier(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        elastic = ElasticNetClassifier(
            alpha=1.0, l1_ratio=0.0, learning_rate=0.05, max_iter=5000
        ).fit(X, y)
        # ElasticNetClassifier's L2 term is halved (alpha/2 * sum(w^2))
        # relative to RidgeClassifier's alpha * sum(w^2), so alpha=1.0 here
        # matches RidgeClassifier's alpha=0.5.
        ridge = RidgeClassifier(alpha=0.5, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(elastic.coef_, ridge.coef_, atol=1e-2)

    def test_separates_linearly_separable_classes(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = ElasticNetClassifier(
            alpha=0.01, l1_ratio=0.5, learning_rate=0.1, max_iter=5000
        ).fit(X, y)

        assert model.score(X, y) > 0.95
