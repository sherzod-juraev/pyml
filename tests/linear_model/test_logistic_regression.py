"""Tests for LogisticRegression."""

import numpy as np

from pyml.linear_model import LogisticRegression


class TestFit:
    def test_separates_linearly_separable_classes(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = LogisticRegression(learning_rate=0.1, max_iter=5000).fit(X, y)

        assert model.score(X, y) > 0.95

    def test_predict_proba_increases_with_positive_class_evidence(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = LogisticRegression(learning_rate=0.1, max_iter=5000).fit(X, y)

        far_negative = model.predict_proba(np.array([[-10.0, -10.0]]))
        far_positive = model.predict_proba(np.array([[10.0, 10.0]]))

        assert far_negative[0] < 0.1
        assert far_positive[0] > 0.9

    def test_predict_proba_stays_in_open_unit_interval(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = LogisticRegression(learning_rate=0.1, max_iter=5000).fit(X, y)
        probs = model.predict_proba(X)

        assert np.all(probs > 0.0)
        assert np.all(probs < 1.0)

    def test_sets_n_iter(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(20, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(20, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 20 + [1] * 20)

        model = LogisticRegression(max_iter=5000).fit(X, y)

        assert 0 < model.n_iter_ <= 5000

    def test_max_iter_is_reported_when_not_converged(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(20, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(20, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 20 + [1] * 20)

        model = LogisticRegression(learning_rate=0.01, max_iter=3, tol=0.0).fit(X, y)

        assert model.n_iter_ == 3

    def test_fit_intercept_false_keeps_intercept_at_zero(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        model = LogisticRegression(learning_rate=0.1, max_iter=2000, fit_intercept=False).fit(X, y)

        assert model.intercept_ == 0.0
        assert model.score(X, y) > 0.95
