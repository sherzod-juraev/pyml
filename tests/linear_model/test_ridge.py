"""Tests for Ridge."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError
from pyml.linear_model import LinearRegression, Ridge


class TestFit:
    @pytest.mark.parametrize("alpha", [0.5, 2.0])
    def test_matches_the_closed_form_solution(self, rng, alpha):
        X = rng.uniform(-5, 5, size=(200, 2))
        y = X @ np.array([3.0, -2.0]) + 5.0 + rng.normal(0, 1.0, size=200)
        # objective: mean squared error + alpha * ||w||^2, intercept not penalized
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        n = len(X)
        expected = np.linalg.solve(Xc.T @ Xc / n + alpha * np.eye(2), Xc.T @ yc / n)

        model = Ridge(alpha=alpha, learning_rate=0.05, max_iter=20000, tol=1e-14).fit(X, y)

        assert np.allclose(model.coef_, expected, atol=1e-5)
        assert model.intercept_ == pytest.approx(y.mean() - X.mean(axis=0) @ expected, abs=1e-5)

    def test_recovers_approximate_coefficients_with_low_alpha(self, linear_data):
        true_coef = np.array([3.0, -2.0])
        X, y = linear_data(coef=(3.0, -2.0))

        model = Ridge(alpha=0.01, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(model.coef_, true_coef, atol=0.1)

    def test_higher_alpha_shrinks_coefficients_more(self, linear_data):
        X, y = linear_data()

        low_alpha = Ridge(alpha=0.1, learning_rate=0.05, max_iter=5000).fit(X, y)
        high_alpha = Ridge(alpha=10.0, learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.sum(high_alpha.coef_**2) < np.sum(low_alpha.coef_**2)

    def test_never_zeroes_out_coefficients(self, linear_data):
        X, y = linear_data()

        model = Ridge(alpha=20.0, learning_rate=0.001, max_iter=20000).fit(X, y)

        assert np.all(model.coef_ != 0.0)

    def test_zero_alpha_matches_linear_regression(self, linear_data):
        X, y = linear_data(n_samples=100, coef=(1.0, 2.0), intercept=1.0)

        ridge = Ridge(alpha=0.0, learning_rate=0.05, max_iter=5000).fit(X, y)
        ols = LinearRegression(learning_rate=0.05, max_iter=5000).fit(X, y)

        assert np.allclose(ridge.coef_, ols.coef_, atol=1e-6)


class TestValidateParams:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("alpha", -1.0),
            ("learning_rate", 0.0),
        ],
    )
    def test_error_message_names_the_parameter(self, linear_data, name, value):
        X, y = linear_data(n_samples=40)

        with pytest.raises(InvalidParameterError, match=name):
            Ridge(**{name: value}).fit(X, y)

    @pytest.mark.parametrize("params", [{"alpha": 0.0}])
    def test_accepts_boundary_values(self, linear_data, params):
        X, y = linear_data(n_samples=40)

        model = Ridge(**params).fit(X, y)

        assert model.is_fitted_ is True
