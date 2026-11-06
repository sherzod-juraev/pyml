"""Tests for LogisticRegression."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.linear_model import (
    ElasticNetClassifier,
    LassoClassifier,
    LogisticRegression,
    RidgeClassifier,
)

BINARY_CLASSIFIERS = [LogisticRegression, RidgeClassifier, LassoClassifier, ElasticNetClassifier]


class TestFit:
    def test_separates_linearly_separable_classes(self, two_blobs):
        X, y = two_blobs()

        model = LogisticRegression(learning_rate=0.1, max_iter=5000).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_predict_proba_increases_with_positive_class_evidence(self, two_blobs):
        X, y = two_blobs()

        model = LogisticRegression(learning_rate=0.1, max_iter=5000).fit(X, y)

        far_negative = model.predict_proba(np.array([[-10.0, -10.0]]))
        far_positive = model.predict_proba(np.array([[10.0, 10.0]]))

        assert far_negative[0] < 0.1
        assert far_positive[0] > 0.9

    def test_predict_proba_stays_in_open_unit_interval(self, two_blobs):
        X, y = two_blobs()

        model = LogisticRegression(learning_rate=0.1, max_iter=5000).fit(X, y)
        probs = model.predict_proba(X)

        assert np.all(probs > 0.0)
        assert np.all(probs < 1.0)

    def test_sets_n_iter(self, two_blobs):
        X, y = two_blobs(n_per_class=20)

        model = LogisticRegression(max_iter=5000).fit(X, y)

        assert 0 < model.n_iter_ <= 5000

    def test_max_iter_is_reported_when_not_converged(self, two_blobs):
        X, y = two_blobs(n_per_class=20)

        model = LogisticRegression(learning_rate=0.01, max_iter=3, tol=0.0).fit(X, y)

        assert model.n_iter_ == 3

    def test_fit_intercept_false_keeps_intercept_at_zero(self, two_blobs):
        X, y = two_blobs(n_per_class=30)

        model = LogisticRegression(learning_rate=0.1, max_iter=2000, fit_intercept=False).fit(X, y)

        assert model.intercept_ == 0.0
        assert model.score(X, y) > 0.95


class TestLabels:
    @pytest.mark.parametrize("cls", BINARY_CLASSIFIERS)
    @pytest.mark.parametrize("labels", [(-1, 1), (1, 2), (3, 7)])
    def test_relabeled_target_gives_the_same_model(self, two_blobs, cls, labels):
        X, y = two_blobs(n_per_class=20)
        low, high = labels
        y_relabeled = np.where(y == 1, high, low)

        reference = cls(learning_rate=0.1, max_iter=500).fit(X, y)
        model = cls(learning_rate=0.1, max_iter=500).fit(X, y_relabeled)

        assert np.allclose(model.coef_, reference.coef_)
        assert model.intercept_ == pytest.approx(reference.intercept_)
        assert np.allclose(model.predict_proba(X), reference.predict_proba(X))
        assert np.array_equal(model.predict(X), np.where(reference.predict(X) == 1, high, low))

    @pytest.mark.parametrize("cls", BINARY_CLASSIFIERS)
    def test_classes_are_the_sorted_distinct_labels(self, two_blobs, cls):
        X, y = two_blobs(n_per_class=20)
        y_relabeled = np.where(y == 1, 7, 3)

        model = cls(max_iter=10).fit(X, y_relabeled)

        assert np.array_equal(model.classes_, [3, 7])

    def test_larger_label_is_the_positive_class_wherever_it_sits(self, two_blobs):
        X, y = two_blobs()
        y_swapped = np.where(y == 1, 2, 5)

        model = LogisticRegression(learning_rate=0.1, max_iter=5000).fit(X, y_swapped)

        far_class_0 = np.array([[-10.0, -10.0]])
        far_class_1 = np.array([[10.0, 10.0]])
        assert model.predict_proba(far_class_0)[0] > 0.9
        assert model.predict_proba(far_class_1)[0] < 0.1
        assert model.predict(far_class_0)[0] == 5
        assert model.predict(far_class_1)[0] == 2

    @pytest.mark.parametrize("cls", BINARY_CLASSIFIERS)
    @pytest.mark.parametrize("n_classes", [1, 3])
    def test_rejects_targets_without_exactly_two_classes(self, two_blobs, cls, n_classes):
        X, _ = two_blobs(n_per_class=20)
        y = np.arange(X.shape[0]) % n_classes

        with pytest.raises(ValueError, match=f"{cls.__name__}.*exactly 2.*found {n_classes}"):
            cls().fit(X, y)

    def test_rejected_refit_keeps_the_previous_fit(self, two_blobs):
        X, y = two_blobs(n_per_class=20)
        model = LogisticRegression(max_iter=50).fit(X, np.where(y == 1, 7, 3))

        with pytest.raises(ValueError, match="exactly 2"):
            model.fit(X, np.arange(X.shape[0]) % 3)

        assert np.array_equal(model.classes_, [3, 7])
        assert set(model.predict(X)) <= {3, 7}


class TestValidateParams:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("learning_rate", 0.0),
            ("max_iter", 0),
            ("tol", -1.0),
        ],
    )
    def test_error_message_names_the_parameter(self, two_blobs, name, value):
        X, y = two_blobs(n_per_class=20)

        with pytest.raises(InvalidParameterError, match=name):
            LogisticRegression(**{name: value}).fit(X, y)

    @pytest.mark.parametrize("params", [{"max_iter": 1, "tol": 0.0}])
    def test_accepts_boundary_values(self, two_blobs, params):
        X, y = two_blobs(n_per_class=20)

        model = LogisticRegression(**params).fit(X, y)

        assert model.is_fitted_ is True
