"""Tests for the lazy-loading mechanism in pyml.linear_model."""

import pytest

from pyml import linear_model


class TestLazyImports:
    def test_linear_regression_is_importable(self):
        assert linear_model.LinearRegression.__name__ == "LinearRegression"

    def test_ridge_is_importable(self):
        assert linear_model.Ridge.__name__ == "Ridge"

    def test_lasso_is_importable(self):
        assert linear_model.Lasso.__name__ == "Lasso"

    def test_elastic_net_is_importable(self):
        assert linear_model.ElasticNet.__name__ == "ElasticNet"

    def test_logistic_regression_is_importable(self):
        assert linear_model.LogisticRegression.__name__ == "LogisticRegression"

    def test_ridge_classifier_is_importable(self):
        assert linear_model.RidgeClassifier.__name__ == "RidgeClassifier"

    def test_lasso_classifier_is_importable(self):
        assert linear_model.LassoClassifier.__name__ == "LassoClassifier"

    def test_elastic_net_classifier_is_importable(self):
        assert linear_model.ElasticNetClassifier.__name__ == "ElasticNetClassifier"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = linear_model.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(linear_model)) == set(linear_model.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(linear_model) == sorted(linear_model.__all__)  # type: ignore[attr-defined]
