"""Tests for the lazy-loading mechanism in pyml.naive_bayes."""

import pytest

from pyml import naive_bayes


class TestLazyImports:
    def test_gaussian_nb_is_importable(self):
        assert naive_bayes.GaussianNB.__name__ == "GaussianNB"

    def test_multinomial_nb_is_importable(self):
        assert naive_bayes.MultinomialNB.__name__ == "MultinomialNB"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = naive_bayes.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(naive_bayes)) == set(naive_bayes.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(naive_bayes) == sorted(naive_bayes.__all__)  # type: ignore[attr-defined]
