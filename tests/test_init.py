"""Test for lazy-loading mechanism in pyml."""

import pytest

import pyml


class TestLazyImports:
    def test_cluster_is_importable(self):
        assert pyml.cluster.__name__ == "pyml.cluster"

    def test_core_is_importable(self):
        assert pyml.core.__name__ == "pyml.core"

    def test_linear_model_is_importable(self):
        assert pyml.linear_model.__name__ == "pyml.linear_model"

    def test_metrics_is_importable(self):
        assert pyml.metrics.__name__ == "pyml.metrics"

    def test_model_selection_is_importable(self):
        assert pyml.model_selection.__name__ == "pyml.model_selection"

    def test_naive_bayes_is_importable(self):
        assert pyml.naive_bayes.__name__ == "pyml.naive_bayes"

    def test_neighbors_is_importable(self):
        assert pyml.neighbors.__name__ == "pyml.neighbors"

    def test_preprocessing_is_importable(self):
        assert pyml.preprocessing.__name__ == "pyml.preprocessing"

    def test_tree_is_importable(self):
        assert pyml.tree.__name__ == "pyml.tree"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = pyml.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(pyml)) == set(pyml.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(pyml) == sorted(pyml.__all__)  # type: ignore[attr-defined]
