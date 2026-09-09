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
