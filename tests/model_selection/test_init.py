"""Test for lazy-loading mechanism in pyml.model_selection."""

import pytest

from pyml import model_selection


class TestLazyImports:
    def test_train_test_split_is_importable(self):
        assert model_selection.train_test_split.__name__ == "train_test_split"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = model_selection.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(model_selection)) == set(model_selection.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(model_selection) == sorted(model_selection.__all__)  # type: ignore[attr-defined]
