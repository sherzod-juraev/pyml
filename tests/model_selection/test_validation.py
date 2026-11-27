"""Tests for cross_val_score.

The folds come from KFold, which has its own tests, so these tests check what
cross_val_score adds: a fresh unfitted copy of the estimator in every fold, the
scoring of each validation part, and the arguments. The expected scores are
computed by hand, fold by fold, with the same estimators.
"""

from typing import Any

import numpy as np
import pytest

from pyml.core.base.estimator import BaseEstimator
from pyml.core.exceptions import InvalidParameterError, ShapeMismatchError
from pyml.linear_model import LinearRegression, LogisticRegression, Ridge
from pyml.model_selection import KFold, cross_val_score
from pyml.pipeline import Pipeline
from pyml.preprocessing import StandardScaler


class _Counter(BaseEstimator):
    """Scores a validation part with its number of samples, so the fold is visible."""

    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset

    def fit(self, X, y):
        self.n_train_ = X.shape[0]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def score(self, X, y):
        return float(X.shape[0]) + self.offset


class _Failing(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        raise RuntimeError("fit failed")

    def predict(self, X):
        return np.zeros(X.shape[0])

    def score(self, X, y):
        return 0.0


@pytest.fixture
def noisy_data(rng):
    X = rng.uniform(-5, 5, size=(40, 3))
    y = X @ np.array([3.0, -2.0, 0.5]) + 5.0 + rng.normal(scale=1.0, size=40)
    return X, y


class TestScores:
    def test_returns_one_score_per_fold(self, noisy_data):
        X, y = noisy_data

        scores = cross_val_score(Ridge(), X, y, cv=4)

        assert scores.shape == (4,)
        assert scores.dtype == np.float64

    def test_equals_fitting_and_scoring_each_fold_by_hand(self, noisy_data):
        X, y = noisy_data

        scores = cross_val_score(Ridge(alpha=0.5), X, y, cv=KFold(4, shuffle=True, random_state=0))

        expected = [
            Ridge(alpha=0.5).fit(X[train_idx], y[train_idx]).score(X[val_idx], y[val_idx])
            for train_idx, val_idx in KFold(4, shuffle=True, random_state=0).split(X)
        ]
        assert np.array_equal(scores, expected)

    def test_classifier_is_scored_with_accuracy(self, two_blobs):
        X, y = two_blobs(n_per_class=30)

        scores = cross_val_score(
            LogisticRegression(), X, y, cv=KFold(5, shuffle=True, random_state=0)
        )

        assert np.array_equal(scores, np.ones(5))

    def test_default_score_is_the_score_method_of_the_estimator(self, noisy_data):
        X, y = noisy_data

        scores = cross_val_score(_Counter(offset=0.5), X, y, cv=4)

        assert np.array_equal(scores, [10.5, 10.5, 10.5, 10.5])

    def test_fold_sizes_reach_the_scores(self, rng):
        X = rng.uniform(size=(23, 2))
        y = rng.uniform(size=23)

        scores = cross_val_score(_Counter(), X, y, cv=5)

        assert np.array_equal(scores, [5, 5, 5, 4, 4])


class TestCv:
    def test_default_is_five_folds(self, noisy_data):
        X, y = noisy_data

        assert cross_val_score(Ridge(), X, y).shape == (5,)

    @pytest.mark.parametrize("n_splits", [2, 3, 7])
    def test_an_integer_gives_that_many_folds(self, noisy_data, n_splits):
        X, y = noisy_data

        assert cross_val_score(Ridge(), X, y, cv=n_splits).shape == (n_splits,)

    def test_an_integer_means_kfold_without_shuffle(self, noisy_data):
        X, y = noisy_data

        by_integer = cross_val_score(Ridge(), X, y, cv=4)
        by_kfold = cross_val_score(Ridge(), X, y, cv=KFold(4))

        assert np.array_equal(by_integer, by_kfold)

    def test_a_kfold_with_a_seed_gives_the_same_scores_every_time(self, noisy_data):
        X, y = noisy_data
        cv = KFold(4, shuffle=True, random_state=1)

        first = cross_val_score(Ridge(), X, y, cv=cv)
        second = cross_val_score(Ridge(), X, y, cv=cv)

        assert np.array_equal(first, second)

    @pytest.mark.parametrize("cv", [1, 0, 2.5, "3", None])
    def test_rejects_an_invalid_cv(self, noisy_data, cv):
        X, y = noisy_data

        with pytest.raises(InvalidParameterError, match="n_splits"):
            cross_val_score(Ridge(), X, y, cv=cv)

    def test_rejects_more_folds_than_samples(self, rng):
        X = rng.uniform(size=(4, 2))
        y = rng.uniform(size=4)

        with pytest.raises(InvalidParameterError, match="greater than the number of samples"):
            cross_val_score(Ridge(), X, y, cv=5)

    def test_rejects_mismatched_samples(self, rng):
        X = rng.uniform(size=(10, 2))
        y = rng.uniform(size=7)

        with pytest.raises(ShapeMismatchError):
            cross_val_score(Ridge(), X, y, cv=2)


class TestScoring:
    def test_a_callable_replaces_the_score_method(self, noisy_data):
        X, y = noisy_data
        folds = iter(range(4))

        scores = cross_val_score(_Counter(), X, y, cv=4, scoring=lambda est, Xv, yv: next(folds))

        assert np.array_equal(scores, [0, 1, 2, 3])

    def test_callable_gets_the_fitted_copy_and_the_validation_part(self, noisy_data):
        X, y = noisy_data
        calls = []

        def scoring(estimator, X_val, y_val):
            calls.append((estimator, X_val, y_val))
            return 0.0

        original = _Counter()
        cross_val_score(original, X, y, cv=KFold(4), scoring=scoring)

        assert len(calls) == 4
        for (estimator, X_val, y_val), (train_idx, val_idx) in zip(
            calls, KFold(4).split(X), strict=True
        ):
            assert estimator is not original
            assert estimator.is_fitted_ is True
            assert estimator.n_train_ == len(train_idx)
            assert np.array_equal(X_val, X[val_idx])
            assert np.array_equal(y_val, y[val_idx])

    def test_the_value_of_the_callable_is_not_changed(self, noisy_data):
        X, y = noisy_data

        scores = cross_val_score(Ridge(), X, y, cv=3, scoring=lambda est, Xv, yv: -2.5)

        assert np.array_equal(scores, [-2.5, -2.5, -2.5])

    def test_an_error_metric_is_used_as_it_is(self, noisy_data):
        X, y = noisy_data

        def squared_error(estimator, X_val, y_val):
            return float(np.mean((y_val - estimator.predict(X_val)) ** 2))

        scores = cross_val_score(Ridge(), X, y, cv=4, scoring=squared_error)

        assert np.all(scores > 0)


class TestEstimatorState:
    def test_the_given_estimator_is_not_fitted(self, noisy_data):
        X, y = noisy_data
        estimator = LinearRegression()

        cross_val_score(estimator, X, y, cv=3)

        assert estimator.is_fitted_ is False
        assert not hasattr(estimator, "coef_")

    def test_a_fitted_estimator_keeps_what_it_learned(self, noisy_data):
        X, y = noisy_data
        estimator = LinearRegression().fit(X, y)
        coef_before = estimator.coef_.copy()

        cross_val_score(estimator, X[:30] * 3.0, y[:30], cv=3)

        assert estimator.is_fitted_ is True
        assert np.array_equal(estimator.coef_, coef_before)

    def test_every_fold_gets_a_new_copy(self, noisy_data):
        X, y = noisy_data
        copies = []

        def scoring(estimator, X_val, y_val):
            copies.append(estimator)
            return 0.0

        original = Ridge()
        cross_val_score(original, X, y, cv=4, scoring=scoring)

        assert len({id(copy) for copy in copies}) == 4
        assert all(copy is not original for copy in copies)
        assert all(copy.get_params() == original.get_params() for copy in copies)

    def test_errors_of_the_estimator_are_not_caught(self, noisy_data):
        X, y = noisy_data

        with pytest.raises(RuntimeError, match="fit failed"):
            cross_val_score(_Failing(), X, y, cv=3)


class TestPipeline:
    def test_equals_chaining_the_steps_by_hand_in_every_fold(self, noisy_data):
        X, y = noisy_data
        X = X * np.array([100.0, 0.01, 1.0]) + 50.0
        pipeline = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=0.5))])
        cv = KFold(4, shuffle=True, random_state=0)

        scores = cross_val_score(pipeline, X, y, cv=cv)

        expected = []
        for train_idx, val_idx in cv.split(X):
            scaler = StandardScaler().fit(X[train_idx])
            model = Ridge(alpha=0.5).fit(scaler.transform(X[train_idx]), y[train_idx])
            expected.append(model.score(scaler.transform(X[val_idx]), y[val_idx]))
        assert np.array_equal(scores, expected)

    def test_the_pipeline_and_its_steps_stay_unfitted(self, noisy_data):
        X, y = noisy_data
        pipeline = Pipeline([("scale", StandardScaler()), ("model", Ridge())])

        cross_val_score(pipeline, X, y, cv=3)

        steps: list[Any] = pipeline.steps
        assert pipeline.is_fitted_ is False
        assert all(step.is_fitted_ is False for _, step in steps)
