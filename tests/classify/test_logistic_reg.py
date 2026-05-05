import pytest
import numpy as np
from sklearn.datasets import make_classification
from mlkit import LogReg
from mlkit.exc import NotFitted


# ─── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    return X, y


@pytest.fixture
def fitted_l2(data):
    X, y = data
    model = LogReg(learning_rate=0.1, max_iter=1000, penalty='l2', alpha=0.01)
    model.fit(X, y)
    return model


# ─── Fit ────────────────────────────────────────────────────

class TestFit:

    def test_weights_not_none_after_fit(self, data):
        X, y = data
        model = LogReg()
        model.fit(X, y)
        assert model.w_ is not None
        assert model.b_ is not None

    def test_weights_shape(self, data):
        X, y = data
        model = LogReg()
        model.fit(X, y)
        assert model.w_.shape == (X.shape[1],)

    def test_bias_is_scalar(self, data):
        X, y = data
        model = LogReg()
        model.fit(X, y)
        assert np.isscalar(model.b_)

    def test_fit_returns_self(self, data):
        X, y = data
        model = LogReg()
        assert model.fit(X, y) is model

    def test_weights_are_float(self, data):
        X, y = data
        model = LogReg()
        model.fit(X, y)
        assert model.w_.dtype == float


# ─── Predict ────────────────────────────────────────────────

class TestPredict:

    def test_raises_not_fitted(self):
        model = LogReg()
        X = np.random.randn(10, 3)
        with pytest.raises(NotFitted):
            model.predict(X)

    def test_output_shape(self, fitted_l2, data):
        X, _ = data
        assert fitted_l2.predict(X).shape == (X.shape[0],)

    def test_output_only_binary(self, fitted_l2, data):
        X, _ = data
        y_pred = fitted_l2.predict(X)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_accuracy_above_threshold(self, fitted_l2, data):
        X, y = data
        y_pred = fitted_l2.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.80


# ─── Sigmoid ────────────────────────────────────────────────

class TestSigmoid:

    def test_output_in_valid_range(self, fitted_l2, data):
        X, _ = data
        out = fitted_l2.sigmoid(X)
        assert np.all(out >= 1e-15)
        assert np.all(out <= 1 - 1e-15)

    def test_output_shape(self, fitted_l2, data):
        X, _ = data
        out = fitted_l2.sigmoid(X)
        assert out.shape == (X.shape[0],)

    def test_no_nan_or_inf(self, fitted_l2, data):
        X, _ = data
        out = fitted_l2.sigmoid(X)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))


# ─── Loss ───────────────────────────────────────────────────

class TestLoss:

    def test_loss_is_positive(self, fitted_l2, data):
        X, y = data
        sig = fitted_l2.sigmoid(X)
        loss = fitted_l2.loss(y, sig, X.shape[0])
        assert loss > 0

    def test_loss_decreases(self, data):
        X, y = data
        n = X.shape[0]

        model_few = LogReg(max_iter=5, tol=0.0, penalty=None)
        model_few.fit(X, y)
        loss_few = model_few.loss(y, model_few.sigmoid(X), n)

        model_many = LogReg(max_iter=1000, tol=0.0, penalty=None)
        model_many.fit(X, y)
        loss_many = model_many.loss(y, model_many.sigmoid(X), n)

        assert loss_many < loss_few

    def test_l2_loss_greater_than_bce(self, data):
        X, y = data
        n = X.shape[0]
        model = LogReg(penalty='l2', alpha=1.0, max_iter=1000)
        model.fit(X, y)
        sig = model.sigmoid(X)

        total = model.loss(y, sig, n)
        bce = -(1 / n) * np.sum(
            y * np.log(sig) + (1 - y) * np.log(1 - sig)
        )
        assert total > bce

    def test_l1_loss_greater_than_bce(self, data):
        X, y = data
        n = X.shape[0]
        model = LogReg(penalty='l1', alpha=1.0, max_iter=1000)
        model.fit(X, y)
        sig = model.sigmoid(X)

        total = model.loss(y, sig, n)
        bce = -(1 / n) * np.sum(
            y * np.log(sig) + (1 - y) * np.log(1 - sig)
        )
        assert total > bce


# ─── Regularization ─────────────────────────────────────────

class TestRegularization:

    def test_l2_weights_smaller_than_no_penalty(self, data):
        X, y = data

        model_none = LogReg(penalty=None, max_iter=1000, tol=1e-4)
        model_none.fit(X, y)

        model_l2 = LogReg(penalty='l2', alpha=5.0, max_iter=1000, tol=1e-4)
        model_l2.fit(X, y)

        assert np.linalg.norm(model_l2.w_) < np.linalg.norm(model_none.w_)

    def test_l1_produces_sparse_weights(self, data):
        X, y = data
        model = LogReg(penalty='l1', alpha=5.0, max_iter=2000, tol=1e-4)
        model.fit(X, y)
        zero_count = np.sum(np.abs(model.w_) < 1e-3)
        assert zero_count >= 1

    def test_larger_alpha_smaller_norm_l2(self, data):
        X, y = data

        model_small = LogReg(penalty='l2', alpha=0.01, max_iter=1000, tol=1e-4)
        model_small.fit(X, y)

        model_large = LogReg(penalty='l2', alpha=10.0, max_iter=1000, tol=1e-4)
        model_large.fit(X, y)

        assert np.linalg.norm(model_large.w_) < np.linalg.norm(model_small.w_)

    def test_penalty_none_runs_without_error(self, data):
        X, y = data
        model = LogReg(penalty=None, max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (X.shape[0],)

    def test_l1_sparser_than_l2(self, data):
        X, y = data

        model_l1 = LogReg(penalty='l1', alpha=1.0, max_iter=1000, tol=1e-4)
        model_l1.fit(X, y)

        model_l2 = LogReg(penalty='l2', alpha=1.0, max_iter=1000, tol=1e-4)
        model_l2.fit(X, y)

        zeros_l1 = np.sum(np.abs(model_l1.w_) < 1e-6)
        zeros_l2 = np.sum(np.abs(model_l2.w_) < 1e-6)
        assert zeros_l1 >= zeros_l2