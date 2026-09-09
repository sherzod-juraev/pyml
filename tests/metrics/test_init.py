"""Tests for the lazy-loading mechanism in pyml.metrics."""

import pytest

from pyml import metrics


class TestLazyImports:
    def test_mean_squared_error_is_importable(self):
        assert metrics.mean_squared_error.__name__ == "mean_squared_error"

    def test_mean_absolute_error_is_importable(self):
        assert metrics.mean_absolute_error.__name__ == "mean_absolute_error"

    def test_root_mean_squared_error_is_importable(self):
        assert metrics.root_mean_squared_error.__name__ == "root_mean_squared_error"

    def test_r2_score_is_importable(self):
        assert metrics.r2_score.__name__ == "r2_score"

    def test_accuracy_score_is_importable(self):
        assert metrics.accuracy_score.__name__ == "accuracy_score"

    def test_f1_score_is_importable(self):
        assert metrics.f1_score.__name__ == "f1_score"

    def test_precision_score_is_importable(self):
        assert metrics.precision_score.__name__ == "precision_score"

    def test_recall_score_is_importable(self):
        assert metrics.recall_score.__name__ == "recall_score"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = metrics.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(metrics)) == set(metrics.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(metrics) == sorted(metrics.__all__)  # type: ignore[attr-defined]
