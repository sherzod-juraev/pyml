"""Test for lazy-loading mechanism in pyml.tree."""

import pytest

from pyml import tree


class TestLazyImports:
    def test_decision_tree_classifier_is_importable(self):
        assert tree.DecisionTreeClassifier.__name__ == "DecisionTreeClassifier"

    def test_decision_tree_regressor_is_importable(self):
        assert tree.DecisionTreeRegressor.__name__ == "DecisionTreeRegressor"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = tree.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(tree)) == set(tree.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(tree) == sorted(tree.__all__)  # type: ignore[attr-defined]
