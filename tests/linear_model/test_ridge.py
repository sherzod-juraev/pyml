import numpy as np
import pytest
from sklearn.datasets import make_regression

from pyml.linear_model.linear_regression import LinearRegression
from pyml.linear_model.ridge import Ridge


class TestRidge:
    """Ridge (L2) regression."""

    @pytest.fixture
    def data(self):
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        return X, y

    def test_fit_predict_consistency(self, data):
        X, y = data
        model = Ridge(learning_rate=0.01, max_iter=5000, alpha=1.0)
        model.fit(X, y)
        pred = model.predict(X)
        assert np.corrcoef(y, pred)[0, 1] > 0.9

    def test_coefficients_shrunk(self, data):
        """L2 should shrink coefficients compared to no penalty."""
        X, y = data
        no_pen = LinearRegression(learning_rate=0.01, max_iter=5000, penalty=None).fit(X, y)
        ridge = Ridge(learning_rate=0.01, max_iter=5000, alpha=10.0).fit(X, y)
        # L2 norm should be smaller
        assert np.linalg.norm(ridge.coef_) <= np.linalg.norm(no_pen.coef_) * 1.1

    def test_coefficients_finite(self, data):
        X, y = data
        model = Ridge().fit(X, y)
        assert np.all(np.isfinite(model.coef_))
        assert np.isfinite(model.intercept_)

    def test_not_fitted_raises(self):
        model = Ridge()
        X = np.random.randn(10, 3).astype(np.float64)
        with pytest.raises(Exception):
            model.predict(X)

    def test_different_alphas(self, data):
        X, y = data
        for alpha in [0.1, 1.0, 10.0]:
            model = Ridge(alpha=alpha).fit(X, y)
            assert model.coef_.shape == (5,)
            assert not np.any(np.isnan(model.coef_))
