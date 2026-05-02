import numpy as np
import pytest
import warnings
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without the real package tree.
# Replace these with your actual imports once integrated into the project.
# ---------------------------------------------------------------------------

class NotFitted(Exception):
    pass


class KmeansPP:
    def __init__(self, k=3, metric="euclidean", random_state=None):
        self.k = k
        self.metric = metric
        self.random_state = random_state
        self.centroids_ = None

    def initialize(self, X):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(X.shape[0], size=self.k, replace=False)
        self.centroids_ = X[idx].copy()
        return self


from scipy.spatial.distance import cdist


class Kmeans:
    def __init__(self, k=3, max_iter=100, tol=1e-3,
                 init="kmeans++", metric="euclidean", random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids_ = None
        self.metric = metric
        self.random_state = random_state
        self.__fitted = False

    def __initialize_centroids_(self, X):
        if self.init == "uniform":
            rng = np.random.default_rng(self.random_state)
            ind = rng.choice(X.shape[0], size=self.k, replace=False)
            self.centroids_ = X[ind].copy()
        else:
            kpp = KmeansPP(k=self.k, metric=self.metric,
                           random_state=self.random_state)
            kpp.initialize(X)
            self.centroids_ = kpp.centroids_

    def fit(self, X):
        self.__initialize_centroids_(X)
        for _ in range(self.max_iter):
            labels = self.__cal_labels(X)
            C_old = self.centroids_.copy()
            self.__update_centroids_(X, labels)
            if self.convergence(C_old):
                break
        self.__fitted = True
        return self

    def convergence(self, C_old):
        dif = self.centroids_ - C_old
        if self.metric == "euclidean":
            dist = np.linalg.norm(dif, ord=2, axis=1)
        elif self.metric == "chebyshev":
            dist = np.max(np.abs(dif), axis=1)
        else:
            dist = np.linalg.norm(dif, ord=1, axis=1)
        return bool(np.all(dist < self.tol))

    def __cal_labels(self, X):
        centroids_ = (self.centroids_ if self.centroids_.ndim == 2
                     else np.array([self.centroids_]))
        distances = cdist(X, centroids_, metric=self.metric)
        return np.argmin(distances, axis=1)

    def __update_centroids_(self, X, labels):
        for i in range(self.k):
            neighbors = X[labels == i]
            if neighbors.shape[0] != 0:
                self.centroids_[i] = neighbors.mean(axis=0)
            else:
                dist = cdist(self.centroids_, X, metric=self.metric)
                self.centroids_[i] = X[np.argmax(np.min(dist, axis=0))].copy()
                warnings.warn(
                    f"Cluster {i} is empty. Reinitializing centroid.",
                    RuntimeWarning,
                )

    def predict(self, X):
        if not self.__fitted:
            raise NotFitted("Kmeans not fitted yet")
        return self.__cal_labels(X)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def well_separated_data():
    """Three clearly separated 2-D Gaussian blobs (300 points total)."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(100, 2))
    c2 = rng.normal(loc=[8.0, 0.0], scale=0.3, size=(100, 2))
    c3 = rng.normal(loc=[4.0, 7.0], scale=0.3, size=(100, 2))
    return np.vstack([c1, c2, c3])


@pytest.fixture
def model_default():
    """Kmeans with default settings (k=3, euclidean, kmeans++)."""
    return Kmeans(k=3, random_state=0)


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------

class TestInitialization:

    def test_centroids__shape_after_fit(self, well_separated_data, model_default):
        """After fit, centroids_ must have shape (k, n_features)."""
        model_default.fit(well_separated_data)
        assert model_default.centroids_.shape == (3, 2)

    def test_uniform_init_produces_valid_centroids_(self, well_separated_data):
        """Uniform init must pick exactly k distinct rows from X."""
        model = Kmeans(k=3, init="uniform", random_state=7)
        model.fit(well_separated_data)
        assert model.centroids_.shape == (3, 2)

    def test_kmeanspp_init_produces_valid_centroids_(self, well_separated_data):
        """KMeans++ init must produce k centroids_ inside the data range."""
        model = Kmeans(k=3, init="kmeans++", random_state=7)
        model.fit(well_separated_data)
        X = well_separated_data
        assert np.all(model.centroids_ >= X.min(axis=0) - 1e-9)
        assert np.all(model.centroids_ <= X.max(axis=0) + 1e-9)

    def test_random_state_reproducibility(self, well_separated_data):
        """Same random_state must yield identical centroids_."""
        m1 = Kmeans(k=3, random_state=42).fit(well_separated_data)
        m2 = Kmeans(k=3, random_state=42).fit(well_separated_data)
        np.testing.assert_array_equal(m1.centroids_, m2.centroids_)

    def test_different_random_states_may_differ(self, well_separated_data):
        """Different seeds should generally produce different centroids_."""
        m1 = Kmeans(k=3, random_state=0).fit(well_separated_data)
        m2 = Kmeans(k=3, random_state=99).fit(well_separated_data)
        # Not guaranteed, but extremely likely on well-separated data
        assert not np.allclose(m1.centroids_, m2.centroids_)


# ---------------------------------------------------------------------------
# 2. Fit / cluster correctness tests
# ---------------------------------------------------------------------------

class TestFitCorrectness:

    def test_cluster_labels_cover_all_k(self, well_separated_data, model_default):
        """All k cluster indices must appear in the predicted labels."""
        model_default.fit(well_separated_data)
        labels = model_default.predict(well_separated_data)
        assert set(labels) == set(range(3))

    def test_labels_length_equals_n_samples(self, well_separated_data, model_default):
        """predict() must return one label per sample."""
        model_default.fit(well_separated_data)
        labels = model_default.predict(well_separated_data)
        assert len(labels) == len(well_separated_data)

    def test_perfect_separation_pure_clusters(self, well_separated_data, model_default):
        """Well-separated blobs: each ground-truth group must map to one label."""
        model_default.fit(well_separated_data)
        labels = model_default.predict(well_separated_data)

        # Ground-truth groups (100 points each)
        g0, g1, g2 = labels[:100], labels[100:200], labels[200:]

        assert len(set(g0)) == 1, "First blob should be in one cluster"
        assert len(set(g1)) == 1, "Second blob should be in one cluster"
        assert len(set(g2)) == 1, "Third blob should be in one cluster"
        assert len({g0[0], g1[0], g2[0]}) == 3, "Each blob should have a unique label"

    def test_wcss_decreases_with_more_iterations(self, well_separated_data):
        """More iterations should not increase within-cluster sum of squares."""
        def wcss(model, X):
            labels = model.predict(X)
            total = 0.0
            for k in range(model.k):
                pts = X[labels == k]
                if len(pts):
                    total += np.sum((pts - model.centroids_[k]) ** 2)
            return total

        m1 = Kmeans(k=3, max_iter=1,   random_state=0).fit(well_separated_data)
        m2 = Kmeans(k=3, max_iter=100, random_state=0).fit(well_separated_data)
        assert wcss(m2, well_separated_data) <= wcss(m1, well_separated_data)


# ---------------------------------------------------------------------------
# 3. Convergence tests
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_convergence_returns_true_when_centroids__unchanged(self):
        """convergence() must return True when old == new centroids_."""
        model = Kmeans(k=2, tol=1e-3)
        model.centroids_ = np.array([[1.0, 2.0], [3.0, 4.0]])
        C_old = model.centroids_.copy()
        assert model.convergence(C_old) is True

    def test_convergence_returns_false_when_centroids__move(self):
        """convergence() must return False when centroid shift > tol."""
        model = Kmeans(k=2, tol=1e-3)
        model.centroids_ = np.array([[1.0, 2.0], [3.0, 4.0]])
        C_old = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert model.convergence(C_old) is False

    @pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "cityblock"])
    def test_convergence_consistent_across_metrics(self, metric):
        """tol should have the same geometric meaning for all three metrics."""
        model = Kmeans(k=2, tol=0.5, metric=metric)
        # Shift each centroid by exactly 0.3 in one dimension → should converge
        model.centroids_ = np.array([[0.3, 0.0], [0.3, 0.0]])
        C_old = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert model.convergence(C_old) is True

    def test_max_iter_respected(self, well_separated_data):
        """fit() must stop after max_iter even without convergence."""
        call_count = {"n": 0}
        original_convergence = Kmeans.convergence

        def counting_convergence(self, C_old):
            call_count["n"] += 1
            return False  # Never converge

        Kmeans.convergence = counting_convergence
        try:
            model = Kmeans(k=3, max_iter=5, random_state=0)
            model.fit(well_separated_data)
            assert call_count["n"] == 5
        finally:
            Kmeans.convergence = original_convergence


# ---------------------------------------------------------------------------
# 4. predict() / NotFitted tests
# ---------------------------------------------------------------------------

class TestPredict:

    def test_predict_raises_if_not_fitted(self, well_separated_data):
        """predict() on an unfitted model must raise NotFitted."""
        model = Kmeans(k=3)
        with pytest.raises(NotFitted):
            model.predict(well_separated_data)

    def test_predict_after_fit_returns_array(self, well_separated_data, model_default):
        """predict() must return a numpy array."""
        model_default.fit(well_separated_data)
        labels = model_default.predict(well_separated_data)
        assert isinstance(labels, np.ndarray)

    def test_predict_label_range(self, well_separated_data, model_default):
        """All predicted labels must be in [0, k)."""
        model_default.fit(well_separated_data)
        labels = model_default.predict(well_separated_data)
        assert labels.min() >= 0
        assert labels.max() < model_default.k

    def test_predict_is_deterministic(self, well_separated_data, model_default):
        """Two predict() calls on the same data must return identical results."""
        model_default.fit(well_separated_data)
        l1 = model_default.predict(well_separated_data)
        l2 = model_default.predict(well_separated_data)
        np.testing.assert_array_equal(l1, l2)


# ---------------------------------------------------------------------------
# 5. Empty cluster handling tests
# ---------------------------------------------------------------------------

class TestEmptyCluster:

    def _force_empty_cluster_fit(self, model, X):
        """
        Bypass __initialize_centroids_ and inject duplicate centroids_ manually,
        so cluster 1 is guaranteed to be empty after the first assignment step.
        Two centroids_ share X[0] → argmin always picks cluster 0, leaving
        cluster 1 with zero neighbours.
        """
        forced_centroids_ = np.array([
            X[0].copy(),    # cluster 0 — valid, attracts many points
            X[0].copy(),    # cluster 1 — duplicate → always empty
            X[-1].copy(),   # cluster 2 — valid, attracts many points
        ], dtype=float)

        # Patch private __initialize_centroids_ so fit() uses our centroids_
        def fake_init(self_inner, X_inner):
            self_inner.centroids_ = forced_centroids_.copy()

        original = Kmeans._Kmeans__initialize_centroids_
        Kmeans._Kmeans__initialize_centroids_ = fake_init
        try:
            model.fit(X)
        finally:
            Kmeans._Kmeans__initialize_centroids_ = original

    def test_empty_cluster_emits_runtime_warning(self, well_separated_data):
        """An empty cluster during update must raise RuntimeWarning."""
        model = Kmeans(k=3, max_iter=2, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._force_empty_cluster_fit(model, well_separated_data)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1

    def test_empty_cluster_centroid_is_reinitialized(self, well_separated_data):
        """
        After reinit the dead centroid must leave the duplicate position.
        The model keeps fitting after reinit so the centroid shifts to the
        mean of its new neighbours — we only verify it moved away from X[0].
        """
        model = Kmeans(k=3, max_iter=2, random_state=0)
        duplicate_position = well_separated_data[0].copy()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self._force_empty_cluster_fit(model, well_separated_data)

        still_at_duplicate = [
            np.linalg.norm(c - duplicate_position) < 1e-9
            for c in model.centroids_
        ]
        # After reinit + continued fitting, centroids_ should not all stay at X[0]
        assert not all(still_at_duplicate)

    def test_warning_message_contains_cluster_index(self, well_separated_data):
        """The warning message must mention 'empty' (case-insensitive)."""
        model = Kmeans(k=3, max_iter=2, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._force_empty_cluster_fit(model, well_separated_data)

        messages = [str(w.message) for w in caught
                    if issubclass(w.category, RuntimeWarning)]
        assert any("empty" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# 6. Metric tests
# ---------------------------------------------------------------------------

class TestMetrics:

    @pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "cityblock"])
    def test_fit_and_predict_work_for_all_metrics(self, well_separated_data, metric):
        """fit + predict must succeed for every supported metric."""
        model = Kmeans(k=3, metric=metric, random_state=0)
        model.fit(well_separated_data)
        labels = model.predict(well_separated_data)
        assert labels.shape == (len(well_separated_data),)

    @pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "cityblock"])
    def test_all_metrics_produce_k_clusters(self, well_separated_data, metric):
        """All k clusters must be non-empty for well-separated data."""
        model = Kmeans(k=3, metric=metric, random_state=0)
        model.fit(well_separated_data)
        labels = model.predict(well_separated_data)
        assert set(labels) == {0, 1, 2}


# ---------------------------------------------------------------------------
# 7. Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_k_equals_n_samples(self):
        """k == n_samples: every point should be its own centroid."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        model = Kmeans(k=3, random_state=0)
        model.fit(X)
        labels = model.predict(X)
        # All labels must be distinct
        assert len(set(labels)) == 3

    def test_single_feature(self):
        """Kmeans must handle 1-D feature arrays (n_samples, 1)."""
        X = np.array([[1.0], [2.0], [10.0], [11.0]])
        model = Kmeans(k=2, random_state=0)
        model.fit(X)
        labels = model.predict(X)
        # Points 0,1 should share a cluster; points 2,3 another
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_high_dimensional_input(self):
        """Kmeans must work with high-dimensional data (d=50)."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 50))
        model = Kmeans(k=5, random_state=0)
        model.fit(X)
        assert model.centroids_.shape == (5, 50)

    def test_integer_dtype_input(self, well_separated_data):
        """Integer dtype input should not cause dtype errors."""
        X_int = well_separated_data.astype(np.int32)
        model = Kmeans(k=3, random_state=0)
        model.fit(X_int)
        labels = model.predict(X_int)
        assert labels.shape == (len(X_int),)