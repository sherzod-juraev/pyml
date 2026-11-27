"""Tests for train_test_split and KFold."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError, ShapeMismatchError
from pyml.model_selection import KFold, train_test_split


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

    def test_rejects_1d_X(self, rng):
        with pytest.raises(ValueError, match="2D"):
            train_test_split(rng.uniform(size=10), rng.uniform(size=10))

    def test_rejects_integer_X(self):
        with pytest.raises(TypeError, match="X must be"):
            train_test_split(np.arange(20).reshape(10, 2), np.arange(10))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    def test_rejects_non_finite_X(self, rng, bad_value):
        X = rng.uniform(0, 10, size=(10, 2))
        X[3, 1] = bad_value

        with pytest.raises(ValueError, match="NaN, infinity"):
            train_test_split(X, np.arange(10))

    def test_rejects_non_numeric_y(self, rng):
        X = rng.uniform(0, 10, size=(4, 2))

        with pytest.raises(TypeError, match="y must be"):
            train_test_split(X, np.array(["a", "b", "a", "b"]))

    def test_rejects_2d_y(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))

        with pytest.raises(ValueError, match="1D"):
            train_test_split(X, rng.uniform(size=(10, 2)))


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


class TestKFoldInit:
    def test_stores_the_parameters(self):
        kfold = KFold(3, shuffle=True, random_state=7)

        assert kfold.n_splits == 3
        assert kfold.shuffle is True
        assert kfold.random_state == 7

    def test_defaults(self):
        kfold = KFold()

        assert kfold.n_splits == 5
        assert kfold.shuffle is False
        assert kfold.random_state is None

    def test_accepts_the_smallest_n_splits_and_numpy_integers(self):
        KFold(2)
        KFold(np.int64(3), random_state=np.int64(0))  # type: ignore[arg-type]

    @pytest.mark.parametrize("n_splits", [1, 0, -2, 2.0, True, "3", None])
    def test_rejects_invalid_n_splits(self, n_splits):
        with pytest.raises(InvalidParameterError, match="n_splits"):
            KFold(n_splits)

    @pytest.mark.parametrize("shuffle", [1, 0, "yes", None])
    def test_rejects_invalid_shuffle(self, shuffle):
        with pytest.raises(InvalidParameterError, match="shuffle"):
            KFold(shuffle=shuffle)

    @pytest.mark.parametrize("random_state", [-1, 1.5, "7", True])
    def test_rejects_invalid_random_state(self, random_state):
        with pytest.raises(InvalidParameterError, match="random_state"):
            KFold(shuffle=True, random_state=random_state)


class TestKFoldSplit:
    @pytest.mark.parametrize("shuffle", [False, True])
    def test_yields_one_pair_per_fold(self, rng, shuffle):
        X = rng.uniform(size=(23, 2))

        folds = list(KFold(5, shuffle=shuffle, random_state=0).split(X))

        assert len(folds) == 5
        assert all(len(pair) == 2 for pair in folds)

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_every_sample_is_validated_exactly_once(self, rng, shuffle):
        X = rng.uniform(size=(23, 2))

        folds = list(KFold(5, shuffle=shuffle, random_state=0).split(X))

        validated = np.concatenate([val_idx for _, val_idx in folds])
        assert np.array_equal(np.sort(validated), np.arange(23))

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_train_is_the_complement_of_the_validation_fold(self, rng, shuffle):
        X = rng.uniform(size=(23, 2))

        for train_idx, val_idx in KFold(5, shuffle=shuffle, random_state=0).split(X):
            assert np.intersect1d(train_idx, val_idx).size == 0
            assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(23))

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_indices_are_in_increasing_order(self, rng, shuffle):
        X = rng.uniform(size=(23, 2))

        for train_idx, val_idx in KFold(5, shuffle=shuffle, random_state=0).split(X):
            assert np.all(np.diff(train_idx) > 0)
            assert np.all(np.diff(val_idx) > 0)

    def test_indices_are_integer_arrays(self, rng):
        X = rng.uniform(size=(10, 2))

        train_idx, val_idx = next(KFold(5).split(X))

        assert train_idx.dtype == np.intp
        assert val_idx.dtype == np.intp

    def test_first_folds_get_the_extra_samples(self, rng):
        X = rng.uniform(size=(23, 2))

        sizes = [len(val_idx) for _, val_idx in KFold(5).split(X)]

        assert sizes == [5, 5, 5, 4, 4]

    def test_equal_folds_when_the_samples_divide_evenly(self, rng):
        X = rng.uniform(size=(20, 2))

        sizes = [len(val_idx) for _, val_idx in KFold(4).split(X)]

        assert sizes == [5, 5, 5, 5]

    def test_without_shuffle_the_folds_are_consecutive_blocks(self, rng):
        X = rng.uniform(size=(10, 2))

        folds = list(KFold(5).split(X))

        assert [val_idx.tolist() for _, val_idx in folds] == [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ]
        assert folds[0][0].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]
        assert folds[2][0].tolist() == [0, 1, 2, 3, 6, 7, 8, 9]

    def test_n_splits_equal_to_n_samples_validates_one_sample_at_a_time(self, rng):
        X = rng.uniform(size=(6, 2))

        folds = list(KFold(6).split(X))

        assert [val_idx.tolist() for _, val_idx in folds] == [[i] for i in range(6)]
        assert all(len(train_idx) == 5 for train_idx, _ in folds)

    def test_only_the_number_of_samples_matters(self, rng):
        X = rng.uniform(size=(12, 2))
        other = rng.uniform(size=(12, 7))
        y = rng.uniform(size=12)

        folds = list(KFold(4, shuffle=True, random_state=0).split(X, y))
        same = list(KFold(4, shuffle=True, random_state=0).split(other))

        for (train_a, val_a), (train_b, val_b) in zip(folds, same, strict=True):
            assert np.array_equal(train_a, train_b)
            assert np.array_equal(val_a, val_b)

    def test_rejects_more_folds_than_samples(self, rng):
        X = rng.uniform(size=(4, 2))

        with pytest.raises(InvalidParameterError, match="greater than the number of samples"):
            list(KFold(5).split(X))

    def test_rejects_mismatched_samples_in_y(self, rng):
        X = rng.uniform(size=(10, 2))
        y = rng.uniform(size=7)

        with pytest.raises(ShapeMismatchError):
            list(KFold(5).split(X, y))


class TestKFoldShuffle:
    def test_shuffle_changes_which_samples_form_a_fold(self, rng):
        X = rng.uniform(size=(20, 2))

        plain = [val_idx for _, val_idx in KFold(4).split(X)]
        shuffled = [val_idx for _, val_idx in KFold(4, shuffle=True, random_state=0).split(X)]

        assert any(not np.array_equal(a, b) for a, b in zip(plain, shuffled, strict=True))

    def test_same_random_state_gives_the_same_folds_on_every_call(self, rng):
        X = rng.uniform(size=(30, 2))
        kfold = KFold(5, shuffle=True, random_state=3)

        first = list(kfold.split(X))
        second = list(kfold.split(X))
        other_instance = list(KFold(5, shuffle=True, random_state=3).split(X))

        for a, b, c in zip(first, second, other_instance, strict=True):
            assert np.array_equal(a[1], b[1])
            assert np.array_equal(a[1], c[1])

    def test_different_random_states_give_different_folds(self, rng):
        X = rng.uniform(size=(50, 2))

        first = [val_idx for _, val_idx in KFold(5, shuffle=True, random_state=0).split(X)]
        second = [val_idx for _, val_idx in KFold(5, shuffle=True, random_state=1).split(X)]

        assert any(not np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_folds_are_the_blocks_of_the_indices_shuffled_by_the_seed(self, rng):
        X = rng.uniform(size=(23, 2))
        indices = np.arange(23)
        np.random.default_rng(42).shuffle(indices)

        folds = [val_idx for _, val_idx in KFold(5, shuffle=True, random_state=42).split(X)]

        expected = [np.sort(block) for block in np.array_split(indices, 5)]
        for fold, block in zip(folds, expected, strict=True):
            assert np.array_equal(fold, block)

    def test_random_state_is_ignored_without_shuffle(self, rng):
        X = rng.uniform(size=(12, 2))

        with_seed = list(KFold(4, random_state=7).split(X))
        without = list(KFold(4).split(X))

        for (train_a, val_a), (train_b, val_b) in zip(with_seed, without, strict=True):
            assert np.array_equal(train_a, train_b)
            assert np.array_equal(val_a, val_b)
