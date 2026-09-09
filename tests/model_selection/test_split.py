"""Tests for train_test_split."""

import numpy as np
import pytest

from pyml.core.exceptions import ShapeMismatchError
from pyml.model_selection import train_test_split


class TestValidation:
    def test_rejects_test_size_zero(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            train_test_split(X, y, test_size=0.0)

    def test_rejects_test_size_one(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=10)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            train_test_split(X, y, test_size=1.0)

    def test_rejects_test_size_that_yields_empty_test_set(self, rng):
        X = rng.uniform(0, 10, size=(3, 2))
        y = rng.uniform(0, 10, size=3)

        with pytest.raises(ValueError, match="empty"):
            train_test_split(X, y, test_size=0.1)

    def test_rejects_mismatched_samples(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))
        y = rng.uniform(0, 10, size=5)

        with pytest.raises(ShapeMismatchError):
            train_test_split(X, y)


class TestSplit:
    def test_split_sizes_match_test_size(self, rng):
        X = rng.uniform(0, 10, size=(100, 2))
        y = rng.uniform(0, 10, size=100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        assert X_test.shape[0] == 25
        assert X_train.shape[0] == 75
        assert y_test.shape[0] == 25
        assert y_train.shape[0] == 75

    def test_no_samples_lost_or_duplicated(self, rng):
        X = rng.uniform(0, 10, size=(50, 2))
        y = np.arange(50)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        all_labels = np.concatenate([y_train, y_test])
        assert sorted(all_labels) == list(range(50))

    def test_reproducible_with_same_random_state(self, rng):
        X = rng.uniform(0, 10, size=(50, 2))
        y = np.arange(50)

        split1 = train_test_split(X, y, random_state=42)
        split2 = train_test_split(X, y, random_state=42)

        assert np.array_equal(split1[0], split2[0])
        assert np.array_equal(split1[2], split2[2])

    def test_shuffle_false_preserves_order(self):
        X = np.arange(20.0).reshape(10, 2)
        y = np.arange(10.0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        assert np.array_equal(y_test, y[:3])
        assert np.array_equal(y_train, y[3:])
