import numpy as np
import pytest
import warnings
from mlkit import Kmeans
from mlkit.exc import NotFitted


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def blobs():
    """Three well-separated clusters."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(50, 2))
    c2 = rng.normal(loc=[5.0, 0.0], scale=0.3, size=(50, 2))
    c3 = rng.normal(loc=[2.5, 4.0], scale=0.3, size=(50, 2))
    X = np.vstack([c1, c2, c3])
    return X


@pytest.fixture
def simple_data():
    """Small deterministic dataset — 2 obvious clusters."""
    X = np.array([
        [0.0, 0.0], [0.1, 0.1], [0.0, 0.1],
        [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],
    ])
    return X


@pytest.fixture
def fitted_model(blobs):
    """Returns an already fitted Kmeans model."""
    model = Kmeans(k=3, random_state=42)
    model.fit(blobs)
    return model, blobs


# ──────────────────────────────────────────────
# 1. Initialization
# ──────────────────────────────────────────────

class TestInit:

    def test_default_params(self):
        model = Kmeans()
        assert model.k == 3
        assert model.max_iter == 100
        assert model.tol == 1e-3
        assert model.init == 'kmeans++'
        assert model.metric == 'euclidean'
        assert model.random_state is None

    def test_custom_params(self):
        model = Kmeans(k=5, max_iter=200, tol=1e-4, init='uniform',
                       metric='cityblock', random_state=0)
        assert model.k == 5
        assert model.max_iter == 200
        assert model.tol == 1e-4
        assert model.init == 'uniform'
        assert model.metric == 'cityblock'
        assert model.random_state == 0

    def test_not_fitted_initially(self):
        model = Kmeans()
        with pytest.raises(NotFitted):
            model.predict(np.array([[1.0, 2.0]]))


# ──────────────────────────────────────────────
# 2. Centroid initialization
# ──────────────────────────────────────────────

class TestInitialization:

    def test_centroids_shape_after_fit(self, blobs):
        model = Kmeans(k=3, random_state=0).fit(blobs)
        assert model.centroids_.shape == (3, blobs.shape[1])

    def test_uniform_init_produces_valid_centroids(self, blobs):
        model = Kmeans(k=3, init='uniform', random_state=0).fit(blobs)
        assert model.centroids_.shape == (3, blobs.shape[1])
        assert np.isfinite(model.centroids_).all()

    def test_kmeanspp_init_produces_valid_centroids(self, blobs):
        model = Kmeans(k=3, init='kmeans++', random_state=0).fit(blobs)
        assert model.centroids_.shape == (3, blobs.shape[1])
        assert np.isfinite(model.centroids_).all()

    def test_random_state_reproducibility(self, blobs):
        m1 = Kmeans(k=3, random_state=42).fit(blobs)
        m2 = Kmeans(k=3, random_state=42).fit(blobs)
        assert np.allclose(m1.centroids_, m2.centroids_)

    def test_different_random_states_may_differ(self, blobs):
        m1 = Kmeans(k=3, random_state=0)
        m2 = Kmeans(k=3, random_state=99)
        m1.initialize_centroids_(blobs)
        m2.initialize_centroids_(blobs)
        assert not np.allclose(m1.centroids_, m2.centroids_)


# ──────────────────────────────────────────────
# 3. Fit correctness
# ──────────────────────────────────────────────

class TestFitCorrectness:

    def test_fit_returns_self(self, blobs):
        model = Kmeans(k=3, random_state=0)
        result = model.fit(blobs)
        assert result is model

    def test_labels_length_equals_n_samples(self, blobs):
        model = Kmeans(k=3, random_state=0).fit(blobs)
        labels = model.predict(blobs)
        assert labels.shape == (blobs.shape[0],)

    def test_cluster_labels_cover_all_k(self, blobs):
        model = Kmeans(k=3, random_state=0).fit(blobs)
        labels = model.predict(blobs)
        assert len(np.unique(labels)) == 3

    def test_perfect_separation_pure_clusters(self, simple_data):
        """On perfectly separated data — each cluster must be pure."""
        model = Kmeans(k=2, random_state=0).fit(simple_data)
        labels = model.predict(simple_data)
        # First 3 samples must share one label, last 3 must share another
        assert len(np.unique(labels[:3])) == 1
        assert len(np.unique(labels[3:])) == 1
        assert labels[0] != labels[3]

    def test_wcss_decreases_with_more_iterations(self, blobs):
        """More iterations → lower or equal within-cluster sum of squares."""
        def wcss(model, X):
            labels = model.predict(X)
            return sum(
                np.sum((X[labels == i] - model.centroids_[i]) ** 2)
                for i in range(model.k)
            )

        m1 = Kmeans(k=3, max_iter=1, random_state=42).fit(blobs)
        m2 = Kmeans(k=3, max_iter=300, random_state=42).fit(blobs)
        assert wcss(m2, blobs) <= wcss(m1, blobs)


# ──────────────────────────────────────────────
# 4. Convergence
# ──────────────────────────────────────────────

class TestConvergence:

    def test_convergence_returns_true_when_centroids_unchanged(self, blobs):
        model = Kmeans(k=3, random_state=0)
        model.initialize_centroids_(blobs)
        C_old = model.centroids_.copy()
        assert model.convergence(C_old) is True

    def test_convergence_returns_false_when_centroids_move(self, blobs):
        model = Kmeans(k=3, random_state=0)
        model.initialize_centroids_(blobs)
        C_old = model.centroids_.copy()
        model.centroids_ += 999
        assert model.convergence(C_old) is False

    def test_max_iter_respected(self, blobs):
        """Model must stop at max_iter — still fitted and predicts."""
        model = Kmeans(k=3, max_iter=1, random_state=0).fit(blobs)
        labels = model.predict(blobs)
        assert labels.shape == (blobs.shape[0],)

    @pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "cityblock"])
    def test_convergence_consistent_across_metrics(self, blobs, metric):
        model = Kmeans(k=3, metric=metric, random_state=0).fit(blobs)
        labels = model.predict(blobs)
        assert len(np.unique(labels)) == 3


# ──────────────────────────────────────────────
# 5. Predict
# ──────────────────────────────────────────────

class TestPredict:

    def test_predict_raises_if_not_fitted(self):
        model = Kmeans()
        with pytest.raises(NotFitted):
            model.predict(np.array([[1.0, 2.0]]))

    def test_predict_after_fit_returns_array(self, fitted_model):
        model, X = fitted_model
        labels = model.predict(X)
        assert isinstance(labels, np.ndarray)

    def test_predict_label_range(self, fitted_model):
        model, X = fitted_model
        labels = model.predict(X)
        assert labels.min() >= 0
        assert labels.max() < model.k

    def test_predict_is_deterministic(self, fitted_model):
        model, X = fitted_model
        labels1 = model.predict(X)
        labels2 = model.predict(X)
        assert np.array_equal(labels1, labels2)


# ──────────────────────────────────────────────
# 6. Empty cluster
# ──────────────────────────────────────────────

class TestEmptyCluster:

    def test_empty_cluster_emits_runtime_warning(self):
        """Force k > actual clusters to trigger empty cluster."""
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        model = Kmeans(k=3, init='uniform', max_iter=1, random_state=0)
        model.initialize_centroids_(X)
        labels = model.cal_labels(X)
        if len(np.unique(labels)) < model.k:
            with pytest.warns(RuntimeWarning):
                model.update_centroids_(X, labels)

    def test_empty_cluster_centroid_is_reinitialized(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        model = Kmeans(k=3, init='uniform', max_iter=5, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X)
        assert np.isfinite(model.centroids_).all()

    def test_warning_message_contains_cluster_index(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        model = Kmeans(k=3, init='uniform', max_iter=1, random_state=0)
        model.initialize_centroids_(X)
        labels = model.cal_labels(X)
        if len(np.unique(labels)) < model.k:
            with pytest.warns(RuntimeWarning, match=r"Cluster \d+"):
                model.update_centroids_(X, labels)


# ──────────────────────────────────────────────
# 7. Metrics
# ──────────────────────────────────────────────

class TestMetrics:

    @pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "cityblock"])
    def test_fit_and_predict_work_for_all_metrics(self, blobs, metric):
        model = Kmeans(k=3, metric=metric, random_state=0).fit(blobs)
        labels = model.predict(blobs)
        assert labels.shape == (blobs.shape[0],)

    @pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "cityblock"])
    def test_all_metrics_produce_k_clusters(self, blobs, metric):
        model = Kmeans(k=3, metric=metric, random_state=0).fit(blobs)
        labels = model.predict(blobs)
        assert len(np.unique(labels)) == 3


# ──────────────────────────────────────────────
# 8. Edge cases
# ──────────────────────────────────────────────

class TestEdgeCases:

    def test_k_equals_n_samples(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model = Kmeans(k=3, random_state=0).fit(X)
        labels = model.predict(X)
        assert len(np.unique(labels)) == 3

    def test_single_feature(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 1))
        model = Kmeans(k=2, random_state=0).fit(X)
        labels = model.predict(X)
        assert labels.shape == (60,)

    def test_high_dimensional_input(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 50))
        model = Kmeans(k=4, random_state=0).fit(X)
        labels = model.predict(X)
        assert labels.shape == (100,)

    def test_integer_dtype_input(self):
        X = np.array([[1, 2], [3, 4], [10, 11], [12, 13]], dtype=int)
        model = Kmeans(k=2, random_state=0).fit(X)
        labels = model.predict(X)
        assert labels.shape == (4,)

    def test_refit_resets_centroids(self, blobs):
        model = Kmeans(k=3, random_state=42)
        model.fit(blobs)
        c1 = model.centroids_.copy()
        model.fit(blobs)
        c2 = model.centroids_.copy()
        assert np.allclose(c1, c2)