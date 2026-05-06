import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinReg
from mlkit import LinReg
from mlkit.exc import NotFitted


# ─── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def data():
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    return X, y


@pytest.fixture
def fitted(data):
    X, y = data
    model = LinReg(learning_rate=0.01, max_iter=1000, tol=1e-4, penalty=None)
    model.fit(X, y)
    return model, X, y


# ════════════════════════════════════════════════════════════
#  Init
# ════════════════════════════════════════════════════════════

class TestInit:

    def test_default_params(self):
        model = LinReg()
        assert model.learning_rate == 0.1
        assert model.max_iter == 100
        assert model.tol == 1e-3
        assert model.alpha == 1.0
        assert model.penalty == 'l2'

    def test_custom_params(self):
        model = LinReg(learning_rate=0.01, max_iter=500, tol=1e-5, alpha=0.5, penalty='l1')
        assert model.learning_rate == 0.01
        assert model.max_iter == 500
        assert model.tol == 1e-5
        assert model.alpha == 0.5
        assert model.penalty == 'l1'

    def test_not_fitted_initially(self):
        model = LinReg()
        with pytest.raises(NotFitted):
            model.predict(np.random.randn(5, 3))


# ════════════════════════════════════════════════════════════
#  Fit
# ════════════════════════════════════════════════════════════

class TestFit:

    def test_fit_returns_self(self, data):
        X, y = data
        model = LinReg()
        assert model.fit(X, y) is model

    def test_fit_sets_coef(self, data):
        X, y = data
        model = LinReg()
        model.fit(X, y)
        assert model.coef_ is not None
        assert model.coef_.shape == (X.shape[1],)

    def test_fit_sets_intercept(self, data):
        X, y = data
        model = LinReg()
        model.fit(X, y)
        assert np.isscalar(model.intercept_)

    def test_coef_dtype_float(self, data):
        X, y = data
        model = LinReg()
        model.fit(X, y)
        assert np.issubdtype(model.coef_.dtype, np.floating)

    def test_refit_resets_weights(self, data):
        X, y = data
        model = LinReg(penalty=None)
        model.fit(X, y)
        coef_first = model.coef_.copy()
        X2, y2 = make_regression(n_samples=100, n_features=5, random_state=0)
        model.fit(X2, y2)
        assert not np.array_equal(model.coef_, coef_first)


# ════════════════════════════════════════════════════════════
#  Predict
# ════════════════════════════════════════════════════════════

class TestPredict:

    def test_raises_not_fitted(self):
        model = LinReg()
        with pytest.raises(NotFitted):
            model.predict(np.random.randn(5, 3))

    def test_output_shape(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_output_is_float(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_no_nan_in_output(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))

    def test_no_inf_in_output(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert not np.any(np.isinf(preds))

    def test_predict_perfect_data(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = LinReg(learning_rate=0.1, max_iter=2000, tol=1e-6, penalty=None)
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_almost_equal(preds, y, decimal=1)

    def test_predict_new_samples(self, fitted):
        model, X, y = fitted
        X_new = np.random.randn(10, X.shape[1])
        preds = model.predict(X_new)
        assert preds.shape == (10,)


# ════════════════════════════════════════════════════════════
#  Convergence
# ════════════════════════════════════════════════════════════

class TestConvergence:

    def test_loss_decreases(self, data):
        X, y = data
        n = X.shape[0]

        model_few = LinReg(max_iter=5, tol=0.0, penalty=None)
        model_few.fit(X, y)
        error_few = y - model_few.predict(X)
        loss_few = model_few.loss(error_few)

        model_many = LinReg(max_iter=1000, tol=0.0, penalty=None)
        model_many.fit(X, y)
        error_many = y - model_many.predict(X)
        loss_many = model_many.loss(error_many)

        assert loss_many < loss_few

    def test_learns_correct_weights(self):
        np.random.seed(42)
        X = np.random.randn(200, 3)
        true_w = np.array([2.0, -1.0, 0.5])
        true_b = 3.0
        y = X @ true_w + true_b + 0.01 * np.random.randn(200)

        model = LinReg(learning_rate=0.01, max_iter=2000, tol=1e-6, penalty=None)
        model.fit(X, y)

        np.testing.assert_array_almost_equal(model.coef_, true_w, decimal=1)
        assert pytest.approx(model.intercept_, abs=0.2) == true_b

    def test_r2_score_high(self, data):
        X, y = data
        model = LinReg(learning_rate=0.01, max_iter=1000, tol=1e-6, penalty=None)
        model.fit(X, y)
        preds = model.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 >= 0.95

    def test_sklearn_comparison(self, data):
        X, y = data
        model = LinReg(learning_rate=0.01, max_iter=2000, tol=1e-6, penalty=None)
        model.fit(X, y)

        sk = SklearnLinReg()
        sk.fit(X, y)

        np.testing.assert_array_almost_equal(model.coef_, sk.coef_, decimal=1)


# ════════════════════════════════════════════════════════════
#  Regularization
# ════════════════════════════════════════════════════════════

class TestRegularization:

    def test_l2_weights_smaller_than_no_penalty(self, data):
        X, y = data

        model_none = LinReg(learning_rate=0.01, max_iter=1000, penalty=None)
        model_none.fit(X, y)

        model_l2 = LinReg(learning_rate=0.01, max_iter=1000, penalty='l2', alpha=5.0)
        model_l2.fit(X, y)

        assert np.linalg.norm(model_l2.coef_) < np.linalg.norm(model_none.coef_)

    def test_l1_produces_sparse_weights(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        true_w = np.array([5.0, -3.0, 2.0, 0, 0, 0, 0, 0, 0, 0])
        y = X @ true_w + 0.1 * np.random.randn(200)

        model = LinReg(
            learning_rate=0.01,
            max_iter=5000,
            tol=1e-6,
            penalty='l1',
            alpha=1.0
        )
        model.fit(X, y)
        zero_count = np.sum(np.abs(model.coef_) < 1e-1)
        assert zero_count >= 1

    def test_larger_alpha_smaller_norm_l2(self, data):
        X, y = data

        model_small = LinReg(learning_rate=0.01, max_iter=1000, penalty='l2', alpha=0.01)
        model_small.fit(X, y)

        model_large = LinReg(learning_rate=0.01, max_iter=1000, penalty='l2', alpha=10.0)
        model_large.fit(X, y)

        assert np.linalg.norm(model_large.coef_) < np.linalg.norm(model_small.coef_)

    def test_l1_sparser_than_l2(self, data):
        X, y = data

        model_l1 = LinReg(learning_rate=0.01, max_iter=2000, penalty='l1', alpha=2.0)
        model_l1.fit(X, y)

        model_l2 = LinReg(learning_rate=0.01, max_iter=2000, penalty='l2', alpha=2.0)
        model_l2.fit(X, y)

        zeros_l1 = np.sum(np.abs(model_l1.coef_) < 1e-3)
        zeros_l2 = np.sum(np.abs(model_l2.coef_) < 1e-3)
        assert zeros_l1 >= zeros_l2

    def test_penalty_none_runs(self, data):
        X, y = data
        model = LinReg(penalty=None, max_iter=1000)
        model.fit(X, y)
        assert model.predict(X).shape == (X.shape[0],)


# ════════════════════════════════════════════════════════════
#  Edge cases
# ════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        model = LinReg(learning_rate=0.01, max_iter=2000, penalty=None)
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_almost_equal(preds, y, decimal=1)

    def test_coef_finite_after_fit(self, data):
        X, y = data
        model = LinReg()
        model.fit(X, y)
        assert np.all(np.isfinite(model.coef_))
        assert np.isfinite(model.intercept_)

    def test_max_iter_respected(self, data):
        X, y = data
        model = LinReg(max_iter=5, tol=0.0)
        model.fit(X, y)
        assert model.coef_ is not None