import numpy as np
import pytest
from sklearn.datasets import make_classification

from pyml.linear_model.logistic_regression import LogisticRegression


class TestLogisticRegression:
    """LogisticRegression core functionality."""

    @pytest.fixture
    def data(self):
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        X, y = make_classification(n_samples=100, n_features=5, n_classes=3,
                                    n_clusters_per_class=1, random_state=42)
        return X, y

    def test_binary_output(self, data):
        X, y = data
        model = LogisticRegression(max_iter=500).fit(X, y)
        pred = model.predict(X)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_predict_shape(self, data):
        X, y = data
        model = LogisticRegression().fit(X, y)
        X_test = np.random.randn(20, 5).astype(np.float64)
        pred = model.predict(X_test)
        assert pred.shape == (20,)

    def test_single_sample(self, data):
        X, y = data
        model = LogisticRegression().fit(X, y)
        pred = model.predict(X[:1])
        assert pred.shape == (1,)

    def test_accuracy_above_chance(self, data):
        X, y = data
        model = LogisticRegression(max_iter=500).fit(X, y)
        pred = model.predict(X)
        acc = np.mean(pred == y)
        assert acc > 0.6, f"Accuracy too low: {acc}"

    def test_coefficients_finite(self, data):
        X, y = data
        model = LogisticRegression().fit(X, y)
        assert np.all(np.isfinite(model.w_))
        assert np.isfinite(model.b_)

    def test_no_penalty(self, data):
        X, y = data
        model = LogisticRegression(penalty=None).fit(X, y)
        assert model.w_.shape == (5,)

    def test_l1_penalty(self, data):
        X, y = data
        model = LogisticRegression(penalty='l1', alpha=0.1).fit(X, y)
        assert model.w_.shape == (5,)
        assert not np.any(np.isnan(model.w_))

    def test_l2_penalty(self, data):
        X, y = data
        model = LogisticRegression(penalty='l2', alpha=1.0).fit(X, y)
        assert model.w_.shape == (5,)
        assert not np.any(np.isnan(model.w_))

    def test_convergence(self, data):
        X, y = data
        model = LogisticRegression(learning_rate=0.1, max_iter=1000, penalty=None)
        model.fit(X, y)
        assert np.all(np.abs(model.w_) < 50), "Weights exploded"

    def test_not_fitted_raises(self):
        model = LogisticRegression()
        X = np.random.randn(10, 3).astype(np.float64)
        with pytest.raises(Exception):
            model.predict(X)

    def test_constant_target(self):
        X = np.random.randn(50, 3).astype(np.float64)
        y = np.ones(50, dtype=np.int64)
        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        pred = model.predict(X)
        assert np.all(pred == 1)

    def test_fit_multiple_times(self, data):
        X, y = data
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        pred1 = model.predict(X)
        model.fit(X, y)
        pred2 = model.predict(X)
        assert np.array_equal(pred1, pred2)
