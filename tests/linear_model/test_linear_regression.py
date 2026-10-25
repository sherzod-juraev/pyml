"""Tests for LinearRegression."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.linear_model import LinearRegression


class TestFit:
    def test_matches_the_least_squares_solution(self, rng):
        X = rng.uniform(-5, 5, size=(200, 2))
        y = X @ np.array([3.0, -2.0]) + 5.0 + rng.normal(0, 1.0, size=200)
        design = np.column_stack([X, np.ones(len(X))])
        reference = np.linalg.lstsq(design, y, rcond=None)[0]

        model = LinearRegression(learning_rate=0.05, max_iter=20000, tol=1e-14).fit(X, y)

        assert np.allclose(model.coef_, reference[:2], atol=1e-5)
        assert model.intercept_ == pytest.approx(reference[2], abs=1e-5)

    def test_recovers_known_coefficients(self, linear_data):
        true_coef = np.array([3.0, -2.0])
        true_intercept = 5.0
        X, y = linear_data(coef=(3.0, -2.0), intercept=true_intercept)

        model = LinearRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(model.coef_, true_coef, atol=0.1)
        assert model.intercept_ == pytest.approx(true_intercept, abs=0.1)

    def test_fit_intercept_false_keeps_intercept_zero(self, linear_data):
        X, y = linear_data(n_samples=100, intercept=0.0)

        model = LinearRegression(fit_intercept=False, max_iter=5000).fit(X, y)

        assert model.intercept_ == 0.0

    def test_sets_n_iter(self, linear_data):
        X, y = linear_data(n_samples=50, coef=(1.0, 1.0), intercept=1.0)

        model = LinearRegression(max_iter=5000).fit(X, y)

        assert 0 < model.n_iter_ <= 5000


class TestScore:
    def test_perfect_fit_gives_high_r2(self, linear_data):
        X, y = linear_data()

        model = LinearRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert model.score(X, y) > 0.99


class TestValidateParams:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("learning_rate", 0.0),
            ("max_iter", 0),
            ("tol", -1.0),
        ],
    )
    def test_error_message_names_the_parameter(self, linear_data, name, value):
        X, y = linear_data(n_samples=40)

        with pytest.raises(InvalidParameterError, match=name):
            LinearRegression(**{name: value}).fit(X, y)

    @pytest.mark.parametrize("params", [{"max_iter": 1, "tol": 0.0}])
    def test_accepts_boundary_values(self, linear_data, params):
        X, y = linear_data(n_samples=40)

        model = LinearRegression(**params).fit(X, y)

        assert model.is_fitted_ is True
