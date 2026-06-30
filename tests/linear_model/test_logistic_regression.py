import numpy as np
import pytest

from pyml import (
    LogisticRegression,
    StandardScaler,
)
from pyml.exc import NotFittedError


class TestBasicCorrectness:
    def test_learns_simple_linear_boundary(self):
        rng = np.random.default_rng(0)

        X = rng.normal(size=(300, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty=None,
        )

        model.fit(X, y)

        preds = model.predict(X)
        accuracy = np.mean(preds == y)

        assert accuracy > 0.98

    def test_predict_returns_binary_labels(self):
        rng = np.random.default_rng(1)

        X = rng.normal(size=(200, 3))
        y = (2 * X[:, 0] - X[:, 1] > 0).astype(int)

        model = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty=None,
        )

        model.fit(X, y)

        preds = model.predict(X)

        assert preds.dtype.kind in ("i", "u")
        assert set(np.unique(preds)).issubset({0, 1})

    def test_multiple_query_points(self):
        X = np.array([
            [-2., -2.],
            [-1., -1.],
            [1., 1.],
            [2., 2.],
        ])

        y = np.array([0, 0, 1, 1])

        model = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty=None,
        )

        model.fit(X, y)

        preds = model.predict(
            np.array([
                [-3., -3.],
                [3., 3.],
            ])
        )

        assert preds.shape == (2,)
        assert np.array_equal(preds, np.array([0, 1]))

    def test_training_accuracy_is_high_after_standard_scaling(self):
        rng = np.random.default_rng(2)

        X = rng.normal(size=(500, 4))
        y = (
            X[:, 0]
            - 2 * X[:, 1]
            + 0.5 * X[:, 2]
            > 0
        ).astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty=None,
        )

        model.fit(X, y)

        preds = model.predict(X)

        assert np.mean(preds == y) > 0.99


class TestNotFitted:
    def test_predict_before_fit_raises(self):
        model = LogisticRegression()

        with pytest.raises(NotFittedError):
            model.predict(np.array([[1.0, 2.0]]))


class TestSigmoid:
    def test_sigmoid_outputs_between_zero_and_one(self):
        model = LogisticRegression()

        model.w_ = np.array([1.0])
        model.b_ = 0.0

        probs = model.sigmoid(
            np.array([
                [-1000.0],
                [0.0],
                [1000.0],
            ])
        )

        assert np.all(probs > 0)
        assert np.all(probs < 1)

    def test_sigmoid_zero_input_equals_half(self):
        model = LogisticRegression()

        model.w_ = np.array([0.0])
        model.b_ = 0.0

        prob = model.sigmoid(np.array([[0.0]]))

        assert prob[0] == pytest.approx(0.5)

    def test_sigmoid_clips_extreme_values(self):
        model = LogisticRegression()

        model.w_ = np.array([1000.0])
        model.b_ = 0.0

        probs = model.sigmoid(
            np.array([
                [1000.0],
                [-1000.0],
            ])
        )

        assert probs[0] < 1.0
        assert probs[0] > 0.999

        assert probs[1] > 0.0
        assert probs[1] < 1e-10


class TestLossFormula:
    def test_bce_without_penalty(self):
        model = LogisticRegression(penalty=None)

        model.w_ = np.array([0.0])
        model.b_ = 0.0

        y = np.array([0, 1])
        y_pred = np.array([0.2, 0.8])

        expected = -np.mean(
            y * np.log(y_pred)
            + (1 - y) * np.log(1 - y_pred)
        )

        assert model.loss(y, y_pred, len(y)) == pytest.approx(expected)

    def test_l2_penalty_added_correctly(self):
        model = LogisticRegression(
            penalty="l2",
            alpha=2.0,
        )

        model.w_ = np.array([1.0, 2.0])
        model.b_ = 100.0

        y = np.array([0, 1])
        y_pred = np.array([0.5, 0.5])

        bce = -np.mean(
            y * np.log(y_pred)
            + (1 - y) * np.log(1 - y_pred)
        )

        expected = bce + (2.0 / (2 * len(y))) * 5

        assert model.loss(y, y_pred, len(y)) == pytest.approx(expected)

    def test_l1_penalty_added_correctly(self):
        model = LogisticRegression(
            penalty="l1",
            alpha=2.0,
        )

        model.w_ = np.array([1.0, -2.0])
        model.b_ = 50.0

        y = np.array([0, 1])
        y_pred = np.array([0.5, 0.5])

        bce = -np.mean(
            y * np.log(y_pred)
            + (1 - y) * np.log(1 - y_pred)
        )

        expected = bce + (2.0 / len(y)) * 3

        assert model.loss(y, y_pred, len(y)) == pytest.approx(expected)

    def test_intercept_is_not_regularized(self):
        model1 = LogisticRegression(alpha=5.0)
        model2 = LogisticRegression(alpha=5.0)

        model1.w_ = np.array([2.0])
        model2.w_ = np.array([2.0])

        model1.b_ = 0.0
        model2.b_ = 1000.0

        y = np.array([0])
        pred = np.array([0.5])

        assert model1.loss(y, pred, 1) == pytest.approx(
            model2.loss(y, pred, 1)
        )


class TestGradientStep:
    def test_gradient_step_without_penalty_matches_manual_formula(self):
        X = np.array([
            [1.0],
            [2.0],
        ])

        error = np.array([
            0.2,
            -0.4,
        ])

        lr = 0.1
        n = X.shape[0]

        model = LogisticRegression(
            learning_rate=lr,
            penalty=None,
        )

        model.w_ = np.array([0.0])
        model.b_ = 0.0

        grad_w = (X.T @ error) / n
        grad_b = np.sum(error) / n

        expected_w = model.w_ - lr * grad_w
        expected_b = model.b_ - lr * grad_b

        model.param_update(X, error)

        assert model.w_ == pytest.approx(expected_w)
        assert model.b_ == pytest.approx(expected_b)

    def test_l2_gradient_matches_manual_formula(self):
        X = np.array([
            [1.0],
            [2.0],
        ])

        error = np.array([
            0.2,
            -0.4,
        ])

        lr = 0.1
        alpha = 2.0
        n = X.shape[0]

        model = LogisticRegression(
            learning_rate=lr,
            penalty="l2",
            alpha=alpha,
        )

        model.w_ = np.array([3.0])
        model.b_ = 0.0

        grad_w = ((X.T @ error) + alpha * model.w_) / n
        grad_b = np.sum(error) / n

        expected_w = model.w_ - lr * grad_w
        expected_b = model.b_ - lr * grad_b

        model.param_update(X, error)

        assert model.w_ == pytest.approx(expected_w)
        assert model.b_ == pytest.approx(expected_b)

    def test_l1_zero_weight_remains_zero_when_error_zero(self):
        X = np.zeros((5, 3))
        error = np.zeros(5)

        model = LogisticRegression(
            learning_rate=0.1,
            penalty="l1",
            alpha=5.0,
        )

        model.w_ = np.zeros(3)
        model.b_ = 0.0

        model.param_update(X, error)

        assert np.allclose(model.w_, 0.0)
        assert model.b_ == pytest.approx(0.0)

    def test_bias_is_updated_even_when_weights_are_zero(self):
        X = np.zeros((4, 2))
        error = np.ones(4)

        model = LogisticRegression(
            learning_rate=0.1,
            penalty=None,
        )

        model.w_ = np.zeros(2)
        model.b_ = 0.0

        model.param_update(X, error)

        assert np.allclose(model.w_, 0.0)
        assert model.b_ < 0


class TestPenaltyBehavior:
    def test_l2_shrinks_weight_norm(self):
        rng = np.random.default_rng(10)

        X = rng.normal(size=(400, 4))
        y = (X @ np.array([3.0, -2.0, 1.0, 0.5]) > 0).astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model_none = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty=None,
        )

        model_l2 = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty="l2",
            alpha=20.0,
        )

        model_none.fit(X, y)
        model_l2.fit(X, y)

        assert np.linalg.norm(model_l2.w_) < np.linalg.norm(model_none.w_)

    def test_l1_produces_sparser_solution_than_l2(self):
        rng = np.random.default_rng(11)

        X = rng.normal(size=(500, 6))
        y = (
            3 * X[:, 0]
            - 2 * X[:, 1]
            + 0.2 * rng.normal(size=500)
            > 0
        ).astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        ridge = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty="l2",
            alpha=10.0,
        )

        lasso = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty="l1",
            alpha=10.0,
        )

        ridge.fit(X, y)
        lasso.fit(X, y)

        ridge_noise = np.sum(np.abs(ridge.w_[2:]))
        lasso_noise = np.sum(np.abs(lasso.w_[2:]))

        assert lasso_noise < ridge_noise

    def test_alpha_zero_matches_no_penalty(self):
        rng = np.random.default_rng(12)

        X = rng.normal(size=(300, 2))
        y = (X[:, 0] - X[:, 1] > 0).astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model_none = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty=None,
        )

        model_l2 = LogisticRegression(
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-6,
            penalty="l2",
            alpha=0.0,
        )

        model_none.fit(X, y)
        model_l2.fit(X, y)

        assert model_l2.w_ == pytest.approx(model_none.w_, abs=1e-3)
        assert model_l2.b_ == pytest.approx(model_none.b_, abs=1e-3)


class TestPredictOutput:
    def test_predict_returns_binary_labels(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler

        X, y = load_breast_cancer(return_X_y=True)
        X = StandardScaler().fit_transform(X)

        model = LogisticRegression(
            learning_rate=0.1,
            max_iter=2000,
            penalty=None,
            tol=1e-6,
        )

        model.fit(X, y)
        pred = model.predict(X)

        assert np.all(np.isin(pred, [0, 1]))


class TestSigmoidRange:
    def test_sigmoid_output_between_zero_and_one(self):
        rng = np.random.default_rng(0)

        X = rng.normal(size=(20, 3))
        y = rng.integers(0, 2, size=20)

        model = LogisticRegression()
        model.fit(X, y)

        p = model.sigmoid(X)

        assert np.all(p > 0)
        assert np.all(p < 1)


class TestNumericalStability:
    def test_sigmoid_handles_large_numbers(self):
        model = LogisticRegression()

        model.w_ = np.array([1000.0])
        model.b_ = 1000.0

        X = np.array([[1000.0]])

        p = model.sigmoid(X)

        assert np.isfinite(p).all()
        assert p[0] < 1.0


class TestSanityAccuracy:
    def test_accuracy_above_random(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X, y = load_breast_cancer(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(
            learning_rate=0.1,
            max_iter=3000,
            penalty="l2",
            tol=1e-6,
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = np.mean(pred == y_test)

        assert accuracy > 0.9
