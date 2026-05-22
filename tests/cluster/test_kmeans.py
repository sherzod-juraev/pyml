import numpy as np
import pytest
from sklearn.datasets import make_blobs

from pyml.cluster.kmeans import Kmeans


class TestKmeans:
    """KMeans clustering."""

    @pytest.fixture
    def data(self):
        X, y = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
        return X, y

    @pytest.fixture
    def high_dim_data(self):
        X, y = make_blobs(n_samples=100, centers=4, n_features=10, random_state=42)
        return X, y

    def test_fit_predict_same_data(self, data):
        """Labels assigned to all training samples."""
        X, _ = data
        model = Kmeans(k=3, random_state=42).fit(X)
        labels = model.predict(X)
        assert labels.shape == (150,)
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_fit_creates_centroids(self, data):
        X, _ = data
        model = Kmeans(k=3, random_state=42).fit(X)
        assert model.centroids_.shape == (3, 2)

    def test_predict_shape(self, data):
        X, _ = data
        model = Kmeans(k=3, random_state=42).fit(X)
        X_test = np.random.randn(20, 2).astype(np.float64)
        labels = model.predict(X_test)
        assert labels.shape == (20,)

    def test_predict_single_sample(self, data):
        X, _ = data
        model = Kmeans(k=3, random_state=42).fit(X)
        label = model.predict(X[:1])
        assert label.shape == (1,)

    def test_convergence(self, data):
        X, _ = data
        model = Kmeans(k=3, max_iter=50, random_state=42).fit(X)
        assert model.centroids_.shape == (3, 2)
        assert np.all(np.isfinite(model.centroids_))

    def test_cluster_sizes_reasonable(self, data):
        """Each cluster should have at least some points."""
        X, _ = data
        model = Kmeans(k=3, random_state=42).fit(X)
        labels = model.predict(X)
        _, counts = np.unique(labels, return_counts=True)
        # Each cluster at least 10% of data
        assert np.all(counts >= 0.1 * X.shape[0]), f"Cluster sizes: {counts}"

    def test_high_dimensional(self, high_dim_data):
        X, _ = high_dim_data
        model = Kmeans(k=4, random_state=42).fit(X)
        assert model.centroids_.shape == (4, 10)
        labels = model.predict(X)
        assert labels.shape == (100,)

    def test_kmeans_plus_plus_init(self, data):
        X, _ = data
        model = Kmeans(k=3, init='kmeans++', random_state=42).fit(X)
        assert model.centroids_.shape == (3, 2)

    def test_uniform_init(self, data):
        X, _ = data
        model = Kmeans(k=3, init='uniform', random_state=42).fit(X)
        assert model.centroids_.shape == (3, 2)

    def test_different_metrics(self, data):
        X, _ = data
        for metric in ['euclidean', 'cityblock', 'chebyshev']:
            model = Kmeans(k=3, metric=metric, random_state=42).fit(X)
            assert model.centroids_.shape == (3, 2)

    def test_reproducibility(self, data):
        X, _ = data
        model1 = Kmeans(k=3, random_state=42).fit(X)
        model2 = Kmeans(k=3, random_state=42).fit(X)
        assert np.allclose(model1.centroids_, model2.centroids_)

    def test_not_fitted_raises(self):
        model = Kmeans(k=3)
        X = np.random.randn(10, 3).astype(np.float64)
        with pytest.raises(Exception):
            model.predict(X)

    def test_k_equals_samples(self):
        X = np.random.randn(5, 2).astype(np.float64)
        model = Kmeans(k=5, random_state=42).fit(X)
        labels = model.predict(X)
        assert labels.shape == (5,)

    def test_single_feature(self):
        X = np.random.randn(50, 1).astype(np.float64)
        model = Kmeans(k=3, random_state=42).fit(X)
        assert model.centroids_.shape == (3, 1)
        labels = model.predict(X)
        assert labels.shape == (50,)

    def test_wcss_decreases(self, data):
        """Within-cluster sum of squares should be reasonable."""
        X, _ = data
        model = Kmeans(k=3, random_state=42).fit(X)
        labels = model.predict(X)
        wcss = 0
        for i in range(model.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - model.centroids_[i])**2)
        # WCSS should be less than total variance
        total_var = np.sum((X - X.mean(axis=0))**2)
        assert wcss < total_var

    def test_fit_multiple_times(self, data):
        X, _ = data
        model = Kmeans(k=3, random_state=42)
        model.fit(X)
        labels1 = model.predict(X)
        model.fit(X)
        labels2 = model.predict(X)
        # Labels may differ but shapes should match
        assert labels1.shape == labels2.shape
