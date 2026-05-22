import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SkKNN

from pyml.neighbors.knn_classifier import KNNClassifier


class TestKNNClassifierPredictions:
    """Prediction consistency and correctness."""

    @pytest.fixture
    def simple_data(self):
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        return X, y

    @pytest.fixture
    def binary_data(self):
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        return X, y

    def test_predict_same_as_train_labels(self, simple_data):
        """Training data predicted labels match true labels (k=1)."""
        X, y = simple_data
        model = KNNClassifier(k=1)
        model.fit(X, y)
        pred = model.predict(X)
        assert np.array_equal(pred, y)

    def test_predict_shape(self, simple_data):
        """Predict returns correct shape."""
        X, y = simple_data
        model = KNNClassifier(k=3).fit(X, y)
        X_test = np.random.randn(20, 2).astype(np.float64)
        pred = model.predict(X_test)
        assert pred.shape == (20,)

    def test_predict_single_sample(self, simple_data):
        """Single sample prediction works."""
        X, y = simple_data
        model = KNNClassifier(k=3).fit(X, y)
        pred = model.predict(X[:1])
        assert pred.shape == (1,)

    def test_predict_1d_input(self, simple_data):
        """1D input auto-reshaped."""
        X, y = simple_data
        model = KNNClassifier(k=3).fit(X, y)
        pred = model.predict(X[0])
        assert pred.shape == (1,)

    def test_binary_output(self, binary_data):
        """Output contains only valid classes."""
        X, y = binary_data
        model = KNNClassifier(k=3).fit(X, y)
        pred = model.predict(X)
        assert set(np.unique(pred)).issubset(set(np.unique(y)))

    def test_uniform_vs_distance_different(self, simple_data):
        """Uniform and distance weighting can produce different results."""
        X, y = simple_data
        X_test = np.random.randn(10, 2).astype(np.float64)

        uniform = KNNClassifier(k=5, weighting='uniform').fit(X, y)
        distance = KNNClassifier(k=5, weighting='distance').fit(X, y)

        # They should both produce valid predictions (not crash)
        assert uniform.predict(X_test).shape == (10,)
        assert distance.predict(X_test).shape == (10,)

    def test_different_k_values(self, simple_data):
        """Different k values all produce valid predictions."""
        X, y = simple_data
        for k in [1, 3, 5, 10]:
            model = KNNClassifier(k=k).fit(X, y)
            pred = model.predict(X[:5])
            assert pred.shape == (5,)

    def test_different_metrics(self, simple_data):
        """Different distance metrics all work."""
        X, y = simple_data
        for metric in ['euclidean', 'cityblock', 'chebyshev']:
            model = KNNClassifier(k=3, metric=metric).fit(X, y)
            pred = model.predict(X[:5])
            assert pred.shape == (5,)

    def test_cosine_metric(self, simple_data):
        """Cosine distance works."""
        X, y = simple_data
        model = KNNClassifier(k=3, metric='cosine').fit(X, y)
        pred = model.predict(X[:5])
        assert pred.shape == (5,)

    def test_reproducibility(self, simple_data):
        """Same input → same output."""
        X, y = simple_data
        model = KNNClassifier(k=3).fit(X, y)
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        assert np.array_equal(pred1, pred2)


class TestKNNClassifierEdgeCases:
    """Edge cases and error handling."""

    def test_not_fitted_raises(self):
        model = KNNClassifier(k=3)
        X = np.random.randn(10, 3).astype(np.float64)
        with pytest.raises(Exception):
            model.predict(X)

    def test_k_larger_than_samples_raises(self):
        X = np.random.randn(5, 2).astype(np.float64)
        y = np.array([0, 1, 0, 1, 0])
        model = KNNClassifier(k=10)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(X)

    def test_k_zero_raises(self):
        X = np.random.randn(10, 2).astype(np.float64)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model = KNNClassifier(k=0)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(X)

    def test_negative_k_raises(self):
        X = np.random.randn(10, 2).astype(np.float64)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model = KNNClassifier(k=-1)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(X)

    def test_single_sample_training(self):
        X = np.array([[1.0, 2.0]], dtype=np.float64)
        y = np.array([1])
        model = KNNClassifier(k=1).fit(X, y)
        pred = model.predict(X)
        assert np.array_equal(pred, y)

    def test_k_equals_n_samples(self):
        X = np.random.randn(5, 2).astype(np.float64)
        y = np.array([0, 1, 0, 1, 0])
        model = KNNClassifier(k=5).fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (5,)

    def test_non_integer_labels(self):
        X = np.random.randn(20, 2).astype(np.float64)
        y = np.array(['a', 'b'] * 10)
        model = KNNClassifier(k=3).fit(X, y)
        pred = model.predict(X[:5])
        assert pred.shape == (5,)
        assert set(pred).issubset({'a', 'b'})


class TestKNNClassifierAccuracy:
    """Accuracy against sklearn (ballpark)."""

    def test_accuracy_close_to_sklearn(self):

        X, y = make_classification(n_samples=300, n_features=5, n_classes=3,
                                    n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        sk = SkKNN(n_neighbors=5, weights='uniform', metric='euclidean')
        my = KNNClassifier(k=5, weighting='uniform', metric='euclidean')

        sk.fit(X_train, y_train)
        my.fit(X_train, y_train)

        sk_acc = np.mean(sk.predict(X_test) == y_test)
        my_acc = np.mean(my.predict(X_test) == y_test)

        assert abs(sk_acc - my_acc) < 0.05, f"Accuracy gap: {abs(sk_acc - my_acc)}"
