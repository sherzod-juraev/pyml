import pytest
import numpy as np
from mlkit import Lasso, Ridge, LinReg
from mlkit.exc import NotFitted


# ─── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def data():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    true_w = np.array([2.0, -1.5, 0.8, -0.3, 1.2])
    y = X @ true_w + 2.0 + 0.1 * np.random.randn(200)
    return X, y


@pytest.fixture
def sparse_data():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    true_w = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0, 0, 0])
    y = X @ true_w + 0.1 * np.random.randn(200)
    return X, y


# ════════════════════════════════════════════════════════════
#  Init
# ════════════════════════════════════════════════════════════

class TestInit:

    def test_default_params(self):
        model = Lasso()
        assert model.learning_rate == 0.1
        assert model.max_iter == 100
        assert model.tol == 1e-3
        assert model.alpha == 1.0

    def test_penalty_locked_l1(self):
        assert Lasso().penalty == 'l1'

    def test_penalty_cannot_be_overridden(self):
        assert Lasso(alpha=0.5).penalty == 'l1'

    def test_custom_params(self):
        model = Lasso(learning_rate=0.01, max_iter=500, tol=1e-5, alpha=0.5)
        assert model.learning_rate == 0.01
        assert model.max_iter == 500
        assert model.tol == 1e-5
        assert model.alpha == 0.5

    def test_not_fitted_initially(self):
        with pytest.raises(NotFitted):
            Lasso().predict(np.random.randn(5, 3))


# ════════════════════════════════════════════════════════════
#  Fit
# ════════════════════════════════════════════════════════════

class TestFit:

    def test_fit_returns_self(self, data):
        X, y = data
        model = Lasso()
        assert model.fit(X, y) is model

    def test_fit_sets_coef(self, data):
        X, y = data
        model = Lasso()
        model.fit(X, y)
        assert model.coef_ is not None
        assert model.coef_.shape == (X.shape[1],)

    def test_fit_sets_intercept(self, data):
        X, y = data
        model = Lasso()
        model.fit(X, y)
        assert np.isscalar(model.intercept_)

    def test_coef_dtype_float(self, data):
        X, y = data
        model = Lasso()
        model.fit(X, y)
        assert np.issubdtype(model.coef_.dtype, np.floating)

    def test_refit_resets_weights(self, data):
        X, y = data
        model = Lasso(learning_rate=0.01, max_iter=1000)
        model.fit(X, y)
        coef_first = model.coef_.copy()
        X2 = np.random.randn(200, 5)
        y2 = np.random.randn(200)
        model.fit(X2, y2)
        assert not np.array_equal(model.coef_, coef_first)


# ════════════════════════════════════════════════════════════
#  Predict
# ════════════════════════════════════════════════════════════

class TestPredict:

    def test_raises_not_fitted(self):
        with pytest.raises(NotFitted):
            Lasso().predict(np.random.randn(5, 3))

    def test_output_shape(self, data):
        X, y = data
        model = Lasso(learning_rate=0.01, max_iter=1000)
        model.fit(X, y)
        assert model.predict(X).shape == (X.shape[0],)

    def test_no_nan_or_inf(self, data):
        X, y = data
        model = Lasso(learning_rate=0.01, max_iter=1000)
        model.fit(X, y)
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))
        assert not np.any(np.isinf(preds))

    def test_predict_new_samples(self, data):
        X, y = data
        model = Lasso(learning_rate=0.01, max_iter=1000)
        model.fit(X, y)
        X_new = np.random.randn(10, X.shape[1])
        assert model.predict(X_new).shape == (10,)


# ════════════════════════════════════════════════════════════
#  Regularization
# ════════════════════════════════════════════════════════════

class TestRegularization:

    def test_produces_sparse_weights(self, sparse_data):
        X, y = sparse_data
        model = Lasso(learning_rate=0.01, max_iter=5000, tol=1e-6, alpha=1.0)
        model.fit(X, y)
        zero_count = np.sum(np.abs(model.coef_) < 1e-1)
        assert zero_count >= 1

    def test_sparser_than_ridge(self, sparse_data):
        X, y = sparse_data
        lasso = Lasso(learning_rate=0.01, max_iter=3000, alpha=1.0)
        lasso.fit(X, y)

        ridge = Ridge(learning_rate=0.01, max_iter=3000, alpha=1.0)
        ridge.fit(X, y)

        zeros_lasso = np.sum(np.abs(lasso.coef_) < 1e-2)
        zeros_ridge = np.sum(np.abs(ridge.coef_) < 1e-2)
        assert zeros_lasso >= zeros_ridge

    def test_larger_alpha_more_sparse(self, sparse_data):
        X, y = sparse_data
        model_small = Lasso(learning_rate=0.01, max_iter=3000, alpha=0.1)
        model_small.fit(X, y)

        model_large = Lasso(learning_rate=0.01, max_iter=3000, alpha=5.0)
        model_large.fit(X, y)

        zeros_small = np.sum(np.abs(model_small.coef_) < 1e-2)
        zeros_large = np.sum(np.abs(model_large.coef_) < 1e-2)
        assert zeros_large >= zeros_small

    def test_weights_smaller_than_no_penalty(self, data):
        X, y = data
        model_none = LinReg(learning_rate=0.01, max_iter=1000, penalty=None)
        model_none.fit(X, y)

        model_lasso = Lasso(learning_rate=0.01, max_iter=1000, alpha=5.0)
        model_lasso.fit(X, y)

        assert np.linalg.norm(model_lasso.coef_) < np.linalg.norm(model_none.coef_)

    def test_r2_reasonable(self, data):
        X, y = data
        model = Lasso(learning_rate=0.01, max_iter=1000, alpha=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 >= 0.80