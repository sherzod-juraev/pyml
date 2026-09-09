"""Tests for the lazy-loading mechanism in pyml.neighbors."""

import pytest

from pyml import neighbors


class TestLazyImports:
    def test_knn_classifier_is_importable(self):
        assert neighbors.KNNClassifier.__name__ == "KNNClassifier"

    def test_knn_regressor_is_importable(self):
        assert neighbors.KNNRegressor.__name__ == "KNNRegressor"

    def test_radius_neighbors_classifier_is_importable(self):
        assert neighbors.RadiusNeighborsClassifier.__name__ == "RadiusNeighborsClassifier"

    def test_radius_neighbors_regressor_is_importable(self):
        assert neighbors.RadiusNeighborsRegressor.__name__ == "RadiusNeighborsRegressor"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = neighbors.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(neighbors)) == set(neighbors.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(neighbors) == sorted(neighbors.__all__)  # type: ignore[attr-defined]
