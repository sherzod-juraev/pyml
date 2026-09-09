"""Tests for LinearRegression."""

import numpy as np
import pytest

from pyml.linear_model import LinearRegression


class TestFit:
    def test_recovers_known_coefficients(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        true_coef = np.array([3.0, -2.0])
        true_intercept = 5.0
        y = X @ true_coef + true_intercept

        model = LinearRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(model.coef_, true_coef, atol=0.1)
        assert model.intercept_ == pytest.approx(true_intercept, abs=0.1)

    def test_fit_intercept_false_keeps_intercept_zero(self, rng):
        X = rng.uniform(-5, 5, size=(100, 2))
        y = X @ np.array([3.0, -2.0])

        model = LinearRegression(fit_intercept=False, max_iter=5000).fit(X, y)

        assert model.intercept_ == 0.0

    def test_sets_n_iter(self, rng):
        X = rng.uniform(-5, 5, size=(50, 2))
        y = X @ np.array([1.0, 1.0]) + 1.0

        model = LinearRegression(max_iter=5000).fit(X, y)

        assert 0 < model.n_iter_ <= 5000


class TestScore:
    def test_perfect_fit_gives_high_r2(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        y = X @ np.array([3.0, -2.0]) + 5.0

        model = LinearRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert model.score(X, y) > 0.99
