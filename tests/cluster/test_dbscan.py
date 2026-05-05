import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from mlkit import DBSCAN


# ─── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def blobs():
    X, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.5,
        random_state=42
    )
    return X


@pytest.fixture
def moons():
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    return X


@pytest.fixture
def data_with_noise():
    X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.3, random_state=42)
    noise = np.random.RandomState(0).uniform(-10, 10, (20, 2))
    return np.vstack([X, noise])


# ════════════════════════════════════════════════════════════
#  Init
# ════════════════════════════════════════════════════════════

class TestInit:

    def test_eps_stored(self):
        model = DBSCAN(eps=0.5)
        assert model.eps == 0.5

    def test_minpts_default(self):
        model = DBSCAN(eps=0.5)
        assert model.MinPts == 5

    def test_metric_default(self):
        model = DBSCAN(eps=0.5)
        assert model.metric == 'euclidean'

    def test_custom_params(self):
        model = DBSCAN(eps=1.0, MinPts=10, metric='cityblock')
        assert model.eps == 1.0
        assert model.MinPts == 10
        assert model.metric == 'cityblock'


# ════════════════════════════════════════════════════════════
#  Output
# ════════════════════════════════════════════════════════════

class TestOutput:

    def test_output_shape(self, blobs):
        model = DBSCAN(eps=0.8, MinPts=5)
        labels = model.fit_predict(blobs)
        assert labels.shape == (blobs.shape[0],)

    def test_output_dtype_int(self, blobs):
        model = DBSCAN(eps=0.8, MinPts=5)
        labels = model.fit_predict(blobs)
        assert labels.dtype == int

    def test_noise_label_is_minus_one(self, data_with_noise):
        model = DBSCAN(eps=0.5, MinPts=5)
        labels = model.fit_predict(data_with_noise)
        assert -1 in labels

    def test_cluster_labels_start_from_zero(self, blobs):
        model = DBSCAN(eps=0.8, MinPts=5)
        labels = model.fit_predict(blobs)
        non_noise = labels[labels != -1]
        assert non_noise.min() == 0

    def test_labels_are_contiguous(self, blobs):
        model = DBSCAN(eps=0.8, MinPts=5)
        labels = model.fit_predict(blobs)
        non_noise = sorted(set(labels[labels != -1]))
        assert non_noise == list(range(len(non_noise)))


# ════════════════════════════════════════════════════════════
#  Clustering correctness
# ════════════════════════════════════════════════════════════

class TestCorrectness:

    def test_detects_three_clusters(self, blobs):
        model = DBSCAN(eps=0.8, MinPts=5)
        labels = model.fit_predict(blobs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters == 3

    def test_all_noise_when_eps_too_small(self, blobs):
        model = DBSCAN(eps=0.001, MinPts=5)
        labels = model.fit_predict(blobs)
        assert np.all(labels == -1)

    def test_one_cluster_when_eps_too_large(self, blobs):
        model = DBSCAN(eps=1000, MinPts=5)
        labels = model.fit_predict(blobs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters == 1

    def test_detects_noise_points(self, data_with_noise):
        model = DBSCAN(eps=0.5, MinPts=5)
        labels = model.fit_predict(data_with_noise)
        noise_count = np.sum(labels == -1)
        assert noise_count > 0

    def test_nonlinear_clusters_moons(self, moons):
        model = DBSCAN(eps=0.2, MinPts=5)
        labels = model.fit_predict(moons)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters == 2


# ════════════════════════════════════════════════════════════
#  Determinism
# ════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_result_twice(self, blobs):
        model = DBSCAN(eps=0.8, MinPts=5)
        labels1 = model.fit_predict(blobs)
        labels2 = model.fit_predict(blobs)
        np.testing.assert_array_equal(labels1, labels2)

    def test_two_instances_same_result(self, blobs):
        labels1 = DBSCAN(eps=0.8, MinPts=5).fit_predict(blobs)
        labels2 = DBSCAN(eps=0.8, MinPts=5).fit_predict(blobs)
        np.testing.assert_array_equal(labels1, labels2)


# ════════════════════════════════════════════════════════════
#  Parameters effect
# ════════════════════════════════════════════════════════════

class TestParameterEffect:

    def test_larger_eps_fewer_noise(self, blobs):
        labels_small = DBSCAN(eps=0.3, MinPts=5).fit_predict(blobs)
        labels_large = DBSCAN(eps=1.5, MinPts=5).fit_predict(blobs)
        noise_small = np.sum(labels_small == -1)
        noise_large = np.sum(labels_large == -1)
        assert noise_large <= noise_small

    def test_larger_minpts_more_noise(self, blobs):
        labels_small = DBSCAN(eps=0.8, MinPts=2).fit_predict(blobs)
        labels_large = DBSCAN(eps=0.8, MinPts=50).fit_predict(blobs)
        noise_small = np.sum(labels_small == -1)
        noise_large = np.sum(labels_large == -1)
        assert noise_large >= noise_small


# ════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════

class TestMetrics:

    @pytest.mark.parametrize('metric', ['euclidean', 'cityblock', 'chebyshev'])
    def test_all_metrics_run(self, blobs, metric):
        model = DBSCAN(eps=0.8, MinPts=5, metric=metric)
        labels = model.fit_predict(blobs)
        assert labels.shape == (blobs.shape[0],)

    @pytest.mark.parametrize('metric', ['euclidean', 'cityblock', 'chebyshev'])
    def test_all_metrics_output_valid_labels(self, blobs, metric):
        model = DBSCAN(eps=0.8, MinPts=5, metric=metric)
        labels = model.fit_predict(blobs)
        assert np.all(labels >= -1)


# ════════════════════════════════════════════════════════════
#  Edge cases
# ════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        model = DBSCAN(eps=0.5, MinPts=1)
        labels = model.fit_predict(X)
        assert labels.shape == (1,)

    def test_identical_points(self):
        X = np.ones((50, 2))
        model = DBSCAN(eps=0.5, MinPts=5)
        labels = model.fit_predict(X)
        unique_non_noise = set(labels[labels != -1])
        assert len(unique_non_noise) <= 1

    def test_high_dimensional(self):
        X = np.random.randn(100, 20)
        model = DBSCAN(eps=3.0, MinPts=5)
        labels = model.fit_predict(X)
        assert labels.shape == (100,)

    def test_minpts_one_no_noise(self):
        X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
        model = DBSCAN(eps=0.8, MinPts=1)
        labels = model.fit_predict(X)
        noise_count = np.sum(labels == -1)
        assert noise_count < 10