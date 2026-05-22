import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as SkKNN

from pyml.neighbors.knn_regressor import KNNRegressor


class TestKNNRegressorPredictions:
    """Prediction consistency and correctness."""

    @pytest.fixture
    def data(self):
        X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
        return X, y

    def test_predict_shape(self, data):
        X, y = data
        model = KNNRegressor(k=3).fit(X, y)
        X_test = np.random.randn(20, 3).astype(np.float64)
        pred = model.predict(X_test)
        assert pred.shape == (20,)

    def test_predict_single_sample(self, data):
        X, y = data
        model = KNNRegressor(k=3).fit(X, y)
        pred = model.predict(X[:1])
        assert pred.shape == (1,)

    def test_predict_1d_input(self, data):
        X, y = data
        model = KNNRegressor(k=3).fit(X, y)
        pred = model.predict(X[0])
        assert pred.shape == (1,)

    def test_predict_values_are_finite(self, data):
        X, y = data
        model = KNNRegressor(k=3).fit(X, y)
        pred = model.predict(X)
        assert np.all(np.isfinite(pred))

    def test_prediction_in_range(self, data):
        X, y = data
        model = KNNRegressor(k=5).fit(X, y)
        pred = model.predict(X)
        # Predictions should be within or close to y range
        assert np.min(pred) >= np.min(y) - 3 * np.std(y)
        assert np.max(pred) <= np.max(y) + 3 * np.std(y)

    def test_different_k_values(self, data):
        X, y = data
        for k in [1, 3, 5, 10]:
            model = KNNRegressor(k=k).fit(X, y)
            pred = model.predict(X[:5])
            assert pred.shape == (5,)

    def test_different_metrics(self, data):
        X, y = data
        for metric in ['euclidean', 'cityblock', 'chebyshev']:
            model = KNNRegressor(k=3, metric=metric).fit(X, y)
            pred = model.predict(X[:5])
            assert pred.shape == (5,)

    def test_reproducibility(self, data):
        X, y = data
        model = KNNRegressor(k=3).fit(X, y)
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        assert np.allclose(pred1, pred2)

    def test_distance_weighting_produces_different_result(self, data):
        X, y = data
        uniform = KNNRegressor(k=5, weighting='uniform').fit(X, y)
        distance = KNNRegressor(k=5, weighting='distance').fit(X, y)

        pred_u = uniform.predict(X[:10])
        pred_d = distance.predict(X[:10])

        assert pred_u.shape == (10,)
        assert pred_d.shape == (10,)

    def test_k1_predicts_training_values(self, data):
        """k=1 should predict exact training values for training points."""
        X, y = data
        model = KNNRegressor(k=1).fit(X, y)
        pred = model.predict(X)
        assert np.allclose(pred, y, rtol=1e-10)

    def test_uniform_weighting_is_average(self, data):
        """With k=n_samples, prediction should be mean of all y."""
        X, y = data
        model = KNNRegressor(k=X.shape[0], weighting='uniform').fit(X, y)
        pred = model.predict(X[:1])
        assert np.allclose(pred, np.mean(y), rtol=1e-10)


class TestKNNRegressorEdgeCases:
    """Edge cases and error handling."""

    def test_not_fitted_raises(self):
        model = KNNRegressor(k=3)
        X = np.random.randn(10, 3).astype(np.float64)
        with pytest.raises(Exception):
            model.predict(X)

    def test_k_larger_than_samples_raises(self):
        X = np.random.randn(5, 2).astype(np.float64)
        y = np.random.randn(5)
        model = KNNRegressor(k=10)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(X)

    def test_constant_target(self):
        X = np.random.randn(20, 2).astype(np.float64)
        y = np.ones(20) * 5.0
        model = KNNRegressor(k=3).fit(X, y)
        pred = model.predict(X)
        assert np.allclose(pred, 5.0)

    def test_single_sample_training(self):
        X = np.array([[1.0, 2.0]], dtype=np.float64)
        y = np.array([3.0])
        model = KNNRegressor(k=1).fit(X, y)
        pred = model.predict(X)
        assert np.allclose(pred, 3.0)


class TestKNNRegressorAccuracy:
    """Accuracy against sklearn (ballpark)."""

    def test_r2_close_to_sklearn(self):
        X, y = make_regression(n_samples=300, n_features=5, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        sk = SkKNN(n_neighbors=5, weights='uniform', metric='euclidean')
        my = KNNRegressor(k=5, weighting='uniform', metric='euclidean')

        sk.fit(X_train, y_train)
        my.fit(X_train, y_train)

        sk_r2 = sk.score(X_test, y_test)
        my_r2 = 1 - np.sum((y_test - my.predict(X_test))**2) / np.sum((y_test - np.mean(y_test))**2)

        assert abs(sk_r2 - my_r2) < 0.05, f"R² gap: {abs(sk_r2 - my_r2)}"
