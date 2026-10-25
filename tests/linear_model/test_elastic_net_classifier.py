"""Tests for ElasticNetClassifier."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.linear_model import ElasticNetClassifier, LassoClassifier, RidgeClassifier


class TestFit:
    def test_l1_ratio_one_behaves_like_lasso_classifier(self, two_blobs):
        X, y = two_blobs()

        elastic = ElasticNetClassifier(
            alpha=0.5, l1_ratio=1.0, learning_rate=0.05, max_iter=5000
        ).fit(X, y)
        lasso = LassoClassifier(alpha=0.5, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(elastic.coef_, lasso.coef_, atol=1e-2)

    def test_l1_ratio_zero_behaves_like_ridge_classifier(self, two_blobs):
        X, y = two_blobs()

        elastic = ElasticNetClassifier(
            alpha=1.0, l1_ratio=0.0, learning_rate=0.05, max_iter=5000
        ).fit(X, y)
        # ElasticNetClassifier's L2 term is halved (alpha/2 * sum(w^2))
        # relative to RidgeClassifier's alpha * sum(w^2), so alpha=1.0 here
        # matches RidgeClassifier's alpha=0.5.
        ridge = RidgeClassifier(alpha=0.5, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(elastic.coef_, ridge.coef_, atol=1e-2)

    def test_separates_linearly_separable_classes(self, two_blobs):
        X, y = two_blobs()

        model = ElasticNetClassifier(
            alpha=0.01, l1_ratio=0.5, learning_rate=0.1, max_iter=5000
        ).fit(X, y)

        assert model.score(X, y) > 0.95


class TestValidateParams:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("alpha", -1.0),
            ("l1_ratio", -0.1),
            ("l1_ratio", 1.5),
            ("learning_rate", 0.0),
        ],
    )
    def test_error_message_names_the_parameter(self, two_blobs, name, value):
        X, y = two_blobs(n_per_class=20)

        with pytest.raises(InvalidParameterError, match=name):
            ElasticNetClassifier(**{name: value}).fit(X, y)

    @pytest.mark.parametrize("params", [{"l1_ratio": 0.0}, {"l1_ratio": 1.0}])
    def test_accepts_boundary_values(self, two_blobs, params):
        X, y = two_blobs(n_per_class=20)

        model = ElasticNetClassifier(**params).fit(X, y)

        assert model.is_fitted_ is True
