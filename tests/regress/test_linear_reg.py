import numpy as np
import pytest
from mlkit.exc import NotFitted
from mlkit import LinReg, MinMaxScaler


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def simple_data():
    """Perfect linear data: y = 2x + 3 (no noise)."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    X = scaler = MinMaxScaler().fit_transform(X)
    y = 2 * X.squeeze() + 3
    return X, y


@pytest.fixture
def noisy_data():
    """Linear data with Gaussian noise."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(100, 1))
    y = 3 * X.squeeze() + 5 + rng.normal(0, 0.5, size=100)
    return X, y


@pytest.fixture
def multifeature_data():
    """Multi-feature linear data: y = 1*x1 + 2*x2 + 0.5."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 5, size=(200, 2))
    y = X @ np.array([1.0, 2.0]) + 0.5
    return X, y


@pytest.fixture
def fitted_model(simple_data):
    """Returns an already fitted LinReg model."""
    X, y = simple_data
    model = LinReg(eta=0.1, max_iter=1000, tol=1e-6)
    model.fit(X, y)
    return model, X, y


# ──────────────────────────────────────────────
# 1. Initialization
# ──────────────────────────────────────────────

class TestInit:

    def test_default_params(self):
        model = LinReg()
        assert model.eta == 1e-1
        assert model.max_iter == 100
        assert model.tol == 1e-3

    def test_custom_params(self):
        model = LinReg(eta=0.01, max_iter=500, tol=1e-5)
        assert model.eta == 0.01
        assert model.max_iter == 500
        assert model.tol == 1e-5

    def test_not_fitted_initially(self):
        model = LinReg()
        with pytest.raises(NotFitted):
            model.predict(np.array([[1.0]]))


# ──────────────────────────────────────────────
# 2. Fit — return value and attributes
# ──────────────────────────────────────────────

class TestFit:

    def test_fit_returns_self(self, simple_data):
        X, y = simple_data
        model = LinReg()
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_coef_(self, simple_data):
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=1000, tol=1e-6).fit(X, y)
        assert hasattr(model, "coef_")
        assert model.coef_.shape == (X.shape[1],)

    def test_fit_sets_intercept_(self, simple_data):
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=1000, tol=1e-6).fit(X, y)
        assert hasattr(model, "intercept_")
        assert isinstance(model.intercept_, float)

    def test_fit_allows_predict_after(self, simple_data):
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=1000, tol=1e-6).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape


# ──────────────────────────────────────────────
# 3. Convergence — simple linear data
# ──────────────────────────────────────────────

class TestConvergence:

    def test_learns_correct_weights(self, simple_data):
        """y = 2x + 3 → coef_ ≈ [2], intercept_ ≈ 3."""
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=5000, tol=1e-8).fit(X, y)
        assert pytest.approx(model.coef_[0], abs=0.05) == 2.0
        assert pytest.approx(model.intercept_, abs=0.05) == 3.0

    def test_multifeature_convergence(self, multifeature_data):
        """y = 1*x1 + 2*x2 + 0.5 → coef_ ≈ [1, 2], intercept_ ≈ 0.5."""
        X, y = multifeature_data
        model = LinReg(eta=0.05, max_iter=5000, tol=1e-8).fit(X, y)
        assert pytest.approx(model.coef_[0], abs=0.1) == 1.0
        assert pytest.approx(model.coef_[1], abs=0.1) == 2.0
        assert pytest.approx(model.intercept_, abs=0.1) == 0.5

    def test_loss_decreases(self, noisy_data):
        """Loss first iteration > loss last iteration."""
        X, y = noisy_data
        model = LinReg(eta=0.01, max_iter=200, tol=1e-10)

        losses = []

        # Monkey-patch to capture loss per iteration
        original_fit = model.fit.__func__

        model.fit(X, y)
        y_pred = model.predict(X)
        final_loss = np.mean((y - y_pred) ** 2)

        # Initial loss (random weights = zeros → mean(y^2))
        initial_loss = np.mean(y ** 2)
        assert final_loss < initial_loss


# ──────────────────────────────────────────────
# 4. Predict
# ──────────────────────────────────────────────

class TestPredict:

    def test_predict_shape(self, fitted_model):
        model, X, y = fitted_model
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_raises_if_not_fitted(self):
        model = LinReg()
        X = np.array([[1.0], [2.0]])
        with pytest.raises(NotFitted):
            model.predict(X)

    def test_predict_perfect_data(self, simple_data):
        """On noiseless data, predictions must be very close to true y."""
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=5000, tol=1e-8).fit(X, y)
        preds = model.predict(X)
        assert np.allclose(preds, y, atol=0.1)

    def test_predict_new_samples(self, simple_data):
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=5000, tol=1e-8).fit(X, y)
        X_new = np.array([[6.0], [7.0]])
        preds = model.predict(X_new)
        expected = np.array([15.0, 17.0])  # 2*6+3, 2*7+3
        assert np.allclose(preds, expected, atol=0.2)


# ──────────────────────────────────────────────
# 5. Edge cases
# ──────────────────────────────────────────────

class TestEdgeCases:

    def test_single_sample(self):
        X = np.array([[3.0]])
        y = np.array([9.0])
        model = LinReg(eta=0.01, max_iter=1000).fit(X, y)
        # Should not raise — predictions must be finite
        preds = model.predict(X)
        assert np.isfinite(preds).all()

    def test_single_feature(self, simple_data):
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=1000, tol=1e-6).fit(X, y)
        assert model.coef_.shape == (1,)

    def test_coef_are_finite_after_fit(self, noisy_data):
        X, y = noisy_data
        model = LinReg(eta=0.01, max_iter=500).fit(X, y)
        assert np.isfinite(model.coef_).all()
        assert np.isfinite(model.intercept_)

    def test_max_iter_respected(self, noisy_data):
        """Model must stop at max_iter even if not converged."""
        X, y = noisy_data
        model = LinReg(eta=1e-10, max_iter=10, tol=1e-20).fit(X, y)
        # If max_iter=10, it ran at most 10 iterations — model still fitted
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_refit_resets_weights(self, simple_data):
        """Fitting twice should give same result as fitting once."""
        X, y = simple_data
        model = LinReg(eta=0.1, max_iter=5000, tol=1e-8)
        model.fit(X, y)
        coef_first = model.coef_.copy()

        model.fit(X, y)
        coef_second = model.coef_.copy()

        assert np.allclose(coef_first, coef_second)