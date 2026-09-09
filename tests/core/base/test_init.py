"""Tests for the lazy-loading mechanism in pyml.core.base."""

import pytest

from pyml.core import base


class TestLazyImports:
    def test_regressor_is_importable(self):
        assert base.Regressor.__name__ == "Regressor"

    def test_classifier_is_importable(self):
        assert base.Classifier.__name__ == "Classifier"

    def test_transformer_is_importable(self):
        assert base.Transformer.__name__ == "Transformer"

    def test_clusterer_is_importable(self):
        assert base.Clusterer.__name__ == "Clusterer"

    def test_predictable_clusterer_is_importable(self):
        assert base.PredictableClusterer.__name__ == "PredictableClusterer"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            base.NonExistent  # type: ignore[attr-defined] # noqa: B018


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(base)) == set(base.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(base) == sorted(base.__all__)  # type: ignore[attr-defined]
