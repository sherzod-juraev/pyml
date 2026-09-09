"""Tests for the lazy-loading mechanism in pyml.core."""

import types

import pytest

import pyml.core as core


class TestLazyImports:
    def test_base_is_importable_and_is_a_module(self):
        assert isinstance(core.base, types.ModuleType)

    def test_exceptions_is_importable_and_is_a_module(self):
        assert isinstance(core.exceptions, types.ModuleType)

    def test_dtypes_is_importable_and_is_a_module(self):
        assert isinstance(core.dtypes, types.ModuleType)

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = core.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(core)) == set(core.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(core) == sorted(core.__all__)  # type: ignore[attr-defined]
