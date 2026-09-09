"""Tests for the lazy-loading mechanism in pyml.core.exceptions."""

import pytest

from pyml.core import exceptions


class TestLazyImports:
    def test_pyml_error_is_importable(self):
        assert exceptions.PymlError.__name__ == "PymlError"

    def test_fitting_error_is_importable(self):
        assert exceptions.FittingError.__name__ == "FittingError"

    def test_not_fitted_error_is_importable(self):
        assert exceptions.NotFittedError.__name__ == "NotFittedError"

    def test_no_neighbors_error_is_importable(self):
        assert exceptions.NoNeighborsError.__name__ == "NoNeighborsError"

    def test_validation_error_is_importable(self):
        assert exceptions.ValidationError.__name__ == "ValidationError"

    def test_invalid_parameter_error_is_importable(self):
        assert exceptions.InvalidParameterError.__name__ == "InvalidParameterError"

    def test_shape_mismatch_error_is_importable(self):
        assert exceptions.ShapeMismatchError.__name__ == "ShapeMismatchError"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = exceptions.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(exceptions)) == set(exceptions.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(exceptions) == sorted(exceptions.__all__)  # type: ignore[attr-defined]
