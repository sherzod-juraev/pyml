import numpy as np
import pytest

from pyml import Lasso, LinearRegression, Ridge
from pyml.exc import NotFittedError


class TestLinearRegressionBasic:
    def test_recovers_known_linear_relationship(self):
        # y = 2x + 1
        rng = np.random.default_rng(0)
        X = rng.uniform(-5, 5, size=(200, 1))
        y = 2.0 * X[:, 0] + 1.0

        model = LinearRegression(learning_rate=0.05, max_iter=2000, penalty=None)
        model.fit(X, y)

        assert model.coef_[0] == pytest.approx(2.0, abs=0.05)
        assert model.intercept_ == pytest.approx(1.0, abs=0.05)

    def test_predict_matches_manual_formula(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([3.0, 5.0, 7.0])  # y = 2x + 1

        model = LinearRegression(learning_rate=0.05, max_iter=2000, penalty=None)
        model.fit(X, y)

        X_new = np.array([[10.0]])
        pred = model.predict(X_new)
        manual = X_new[0, 0] * model.coef_[0] + model.intercept_
        assert pred[0] == pytest.approx(manual)

    def test_loss_decreases_monotonically_with_small_lr(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 2))
        y = X @ np.array([1.0, -1.0]) + 0.5

        model = LinearRegression(learning_rate=0.01, max_iter=1, penalty=None)
        model.coef_ = np.zeros(2)
        model.intercept_ = 0.0

        losses = []
        y_pred = X @ model.coef_ + model.intercept_
        losses.append(model.loss(y - y_pred))
        for _ in range(20):
            y_pred = X @ model.coef_ + model.intercept_
            error = y - y_pred
            model.param_update(X, error)
            y_pred = X @ model.coef_ + model.intercept_
            losses.append(model.loss(y - y_pred))

        diffs = np.diff(losses)
        assert np.all(diffs <= 1e-8)

    def test_perfect_fit_zero_loss(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1

        model = LinearRegression(learning_rate=0.05, max_iter=5000, penalty=None)
        model.fit(X, y)

        preds = model.predict(X)
        final_loss = model.loss(y - preds)
        assert final_loss == pytest.approx(0.0, abs=1e-4)


class TestNotFitted:
    def test_predict_before_fit_raises_linear(self):
        model = LinearRegression()
        with pytest.raises(NotFittedError):
            model.predict(np.array([[1.0]]))

    def test_predict_before_fit_raises_ridge(self):
        model = Ridge()
        with pytest.raises(NotFittedError):
            model.predict(np.array([[1.0]]))

    def test_predict_before_fit_raises_lasso(self):
        model = Lasso()
        with pytest.raises(NotFittedError):
            model.predict(np.array([[1.0]]))


class TestLossFormula:
    def test_no_penalty_loss_is_pure_mse(self):
        model = LinearRegression(penalty=None)
        model.coef_ = np.array([1.0, 2.0])
        model.intercept_ = 0.0

        error = np.array([1.0, -2.0, 3.0])  # MSE = (1+4+9)/3 = 4.6667
        loss = model.loss(error)
        assert loss == pytest.approx(np.mean(error**2))

    def test_l2_penalty_adds_correct_term(self):
        model = LinearRegression(penalty="l2", alpha=2.0)
        model.coef_ = np.array([1.0, 2.0])  # sum(w^2) = 1 + 4 = 5
        model.intercept_ = 0.0

        error = np.array([0.0, 0.0])  # MSE = 0
        n = error.shape[0]
        expected_penalty = (2.0 / (2 * n)) * 5.0  # alpha/(2n) * sum(w^2)

        loss = model.loss(error)
        assert loss == pytest.approx(expected_penalty)

    def test_l1_penalty_adds_correct_term(self):
        model = LinearRegression(penalty="l1", alpha=2.0)
        model.coef_ = np.array([1.0, -2.0])  # sum|w| = 1 + 2 = 3
        model.intercept_ = 0.0

        error = np.array([0.0, 0.0])  # MSE = 0
        n = error.shape[0]
        expected_penalty = (2.0 / n) * 3.0  # alpha/n * sum|w|

        loss = model.loss(error)
        assert loss == pytest.approx(expected_penalty)

    def test_intercept_excluded_from_penalty(self):
        model_small_b = LinearRegression(penalty="l2", alpha=1.0)
        model_small_b.coef_ = np.array([1.0])
        model_small_b.intercept_ = 0.0

        model_big_b = LinearRegression(penalty="l2", alpha=1.0)
        model_big_b.coef_ = np.array([1.0])
        model_big_b.intercept_ = 1000.0

        error = np.array([0.0, 0.0])
        assert model_small_b.loss(error) == pytest.approx(model_big_b.loss(error))


class TestGradientStep:
    def test_single_step_matches_manual_gradient_no_penalty(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([3.0, 5.0])
        lr = 0.1

        model = LinearRegression(learning_rate=lr, penalty=None)
        model.coef_ = np.zeros(1)
        model.intercept_ = 0.0

        y_pred = X @ model.coef_ + model.intercept_  # [0, 0]
        error = y - y_pred  # [3, 5]

        n = X.shape[0]
        grad_w_manual = -2 * (X.T @ error) / n
        grad_b_manual = -2 * np.sum(error) / n

        expected_w = model.coef_ - lr * grad_w_manual
        expected_b = model.intercept_ - lr * grad_b_manual

        model.param_update(X, error)

        assert model.coef_ == pytest.approx(expected_w)
        assert model.intercept_ == pytest.approx(expected_b)

    def test_l1_subgradient_zero_at_zero_weight(self):
        X = np.array([[1.0, 0.0], [2.0, 0.0]])
        y = np.array([0.0, 0.0])  # error = 0

        model = LinearRegression(learning_rate=0.1, penalty="l1", alpha=1.0)
        model.coef_ = np.array([0.0, 0.0])
        model.intercept_ = 0.0

        error = y - (X @ model.coef_ + model.intercept_)  # [0, 0]
        model.param_update(X, error)

        assert model.coef_[0] == pytest.approx(0.0)
        assert model.coef_[1] == pytest.approx(0.0)


class TestRidge:
    def test_ridge_shrinks_weights_compared_to_no_penalty(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 3))
        true_w = np.array([3.0, -2.0, 1.0])
        y = X @ true_w + 0.5 + 0.3 * rng.normal(size=100)

        model_none = LinearRegression(learning_rate=0.05, max_iter=2000, penalty=None)
        model_none.fit(X, y)

        model_ridge = Ridge(learning_rate=0.05, max_iter=2000, alpha=50.0)
        model_ridge.fit(X, y)

        norm_none = np.linalg.norm(model_none.coef_)
        norm_ridge = np.linalg.norm(model_ridge.coef_)
        assert norm_ridge < norm_none

    def test_ridge_never_produces_exact_zeros_with_strong_penalty(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 3))
        y = X @ np.array([0.01, 0.01, 0.01]) + rng.normal(scale=0.01, size=50)

        model = Ridge(learning_rate=0.05, max_iter=2000, alpha=100.0)
        model.fit(X, y)

        assert np.all(model.coef_ != 0.0)

    def test_ridge_alpha_zero_behaves_like_no_penalty(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(60, 2))
        y = X @ np.array([2.0, -1.0]) + 1.0

        model_ridge = Ridge(learning_rate=0.05, max_iter=2000, alpha=0.0)
        model_ridge.fit(X, y)

        model_none = LinearRegression(learning_rate=0.05, max_iter=2000, penalty=None)
        model_none.fit(X, y)

        assert model_ridge.coef_ == pytest.approx(model_none.coef_, abs=1e-3)


class TestLasso:
    def test_lasso_drives_irrelevant_weights_toward_zero(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(300, 5))
        true_w = np.array([3.0, -2.0, 0.0, 0.0, 0.0])
        y = X @ true_w + 1.0 + 0.05 * rng.normal(size=300)

        model = Lasso(learning_rate=0.05, max_iter=3000, alpha=5.0)
        model.fit(X, y)

        assert np.abs(model.coef_[2]) < 0.3
        assert np.abs(model.coef_[3]) < 0.3
        assert np.abs(model.coef_[4]) < 0.3
        assert np.abs(model.coef_[0]) > 1.0
        assert np.abs(model.coef_[1]) > 1.0

    def test_lasso_sparser_than_ridge_with_equal_alpha(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(300, 5))
        true_w = np.array([3.0, -2.0, 0.0, 0.0, 0.0])
        y = X @ true_w + 1.0 + 0.05 * rng.normal(size=300)

        lasso = Lasso(learning_rate=0.05, max_iter=3000, alpha=5.0)
        lasso.fit(X, y)

        ridge = Ridge(learning_rate=0.05, max_iter=3000, alpha=5.0)
        ridge.fit(X, y)

        lasso_irrelevant = np.sum(np.abs(lasso.coef_[2:]))
        ridge_irrelevant = np.sum(np.abs(ridge.coef_[2:]))
        assert lasso_irrelevant < ridge_irrelevant

    def test_lasso_alpha_zero_behaves_like_no_penalty(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 2))
        y = X @ np.array([2.0, -1.0]) + 1.0

        model_lasso = Lasso(learning_rate=0.05, max_iter=2000, alpha=0.0)
        model_lasso.fit(X, y)

        model_none = LinearRegression(learning_rate=0.05, max_iter=2000, penalty=None)
        model_none.fit(X, y)

        assert model_lasso.coef_ == pytest.approx(model_none.coef_, abs=1e-3)

    def test_lasso_inherits_l1_penalty_fixed(self):
        model = Lasso()
        assert model.penalty == "l1"

    def test_ridge_inherits_l2_penalty_fixed(self):
        model = Ridge()
        assert model.penalty == "l2"
