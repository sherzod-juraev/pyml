"""Test for lazy-loading mechanism in pyml.preprocessing."""

import pytest

from pyml import preprocessing


class TestLazyImports:
    def test_min_max_scaler_is_importable(self):
        assert preprocessing.MinMaxScaler.__name__ == "MinMaxScaler"

    def test_robust_scaler_is_importable(self):
        assert preprocessing.RobustScaler.__name__ == "RobustScaler"

    def test_standard_scaler_is_importable(self):
        assert preprocessing.StandardScaler.__name__ == "StandardScaler"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = preprocessing.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(preprocessing)) == set(preprocessing.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(preprocessing) == sorted(preprocessing.__all__)  # type: ignore[attr-defined]
