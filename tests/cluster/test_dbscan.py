import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

from pyml.cluster.dbscan import DBSCAN


class TestDBSCAN:
    """DBSCAN clustering."""

    @pytest.fixture
    def blob_data(self):
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        return X

    @pytest.fixture
    def moon_data(self):
        X, y = make_moons(n_samples=100, noise=0.05, random_state=42)
        return X

    def test_fit_predict_shape(self, blob_data):
        """Labels match number of samples."""
        model = DBSCAN(eps=1.0, MinPts=5)
        labels = model.fit_predict(blob_data)
        assert labels.shape == (100,)

    def test_noise_label(self, blob_data):
        """Noise points are -1."""
        model = DBSCAN(eps=0.1, MinPts=50)
        labels = model.fit_predict(blob_data)
        assert -1 in labels

    def test_clusters_found(self, blob_data):
        """With reasonable eps, should find clusters."""
        model = DBSCAN(eps=1.0, MinPts=5)
        labels = model.fit_predict(blob_data)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        assert len(unique_labels) >= 2

    def test_all_noise_with_small_eps(self, blob_data):
        """Very small eps → all noise."""
        model = DBSCAN(eps=0.01, MinPts=5)
        labels = model.fit_predict(blob_data)
        assert np.all(labels == -1)

    def test_single_cluster_with_large_eps(self, blob_data):
        """Very large eps → one cluster."""
        model = DBSCAN(eps=100.0, MinPts=5)
        labels = model.fit_predict(blob_data)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        assert len(unique_labels) == 1

    def test_moons_shape(self, moon_data):
        """DBSCAN can find non-convex clusters."""
        model = DBSCAN(eps=0.3, MinPts=5)
        labels = model.fit_predict(moon_data)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        assert len(unique_labels) >= 2

    def test_deterministic(self, blob_data):
        """Same input → same output."""
        model1 = DBSCAN(eps=1.0, MinPts=5)
        model2 = DBSCAN(eps=1.0, MinPts=5)
        assert np.array_equal(model1.fit_predict(blob_data), model2.fit_predict(blob_data))

    def test_different_metrics(self, blob_data):
        """Different metrics produce valid labels."""
        for metric in ['euclidean', 'cityblock', 'chebyshev']:
            model = DBSCAN(eps=1.0, MinPts=5, metric=metric)
            labels = model.fit_predict(blob_data)
            assert labels.shape == (100,)

    def test_invalid_metric_raises(self, blob_data):
        model = DBSCAN(eps=1.0, MinPts=5, metric='cosine')
        with pytest.raises(ValueError):
            model.fit_predict(blob_data)

    def test_single_sample(self):
        X = np.array([[1.0, 2.0]], dtype=np.float64)
        model = DBSCAN(eps=1.0, MinPts=1)
        labels = model.fit_predict(X)
        assert labels.shape == (1,)
        assert labels[0] == 0

    def test_high_dimensional(self):
        X = np.random.randn(50, 10).astype(np.float64)
        model = DBSCAN(eps=5.0, MinPts=3)
        labels = model.fit_predict(X)
        assert labels.shape == (50,)

    def test_labels_are_sequential(self, blob_data):
        """Cluster labels should be sequential from 0."""
        model = DBSCAN(eps=1.0, MinPts=5)
        labels = model.fit_predict(blob_data)
        non_noise = labels[labels != -1]
        if len(non_noise) > 0:
            assert set(non_noise) == set(range(max(non_noise) + 1))
