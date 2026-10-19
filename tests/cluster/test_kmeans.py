"""Tests for KMeans."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pyml.cluster import KMeans
from pyml.core.exceptions import InvalidParameterError, NotFittedError


class TestFit:
    def test_rejects_zero_or_negative_n_clusters(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))

        with pytest.raises(InvalidParameterError):
            KMeans(n_clusters=0).fit(X)

    def test_rejects_n_clusters_exceeding_sample_count(self, rng):
        X = rng.uniform(0, 10, size=(5, 2))

        with pytest.raises(InvalidParameterError):
            KMeans(n_clusters=10).fit(X)

    def test_sets_n_iter(self, rng):
        X0 = rng.normal(loc=(-3, -3), scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=(3, 3), scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])

        model = KMeans(n_clusters=2, random_state=42).fit(X)

        assert 0 < model.n_iter_ <= model.max_iter

    def test_finds_well_separated_clusters(self, rng):
        X0 = rng.normal(loc=(-10, -10), scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=(10, 10), scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])

        model = KMeans(n_clusters=2, random_state=42).fit(X)

        # all points in the first blob should share one label, and all
        # points in the second blob should share a different label
        labels_0 = model.labels_[:30]
        labels_1 = model.labels_[30:]
        assert len(np.unique(labels_0)) == 1
        assert len(np.unique(labels_1)) == 1
        assert labels_0[0] != labels_1[0]

    def test_centroids_are_close_to_true_centers(self, rng):
        X0 = rng.normal(loc=(-10, -10), scale=0.5, size=(100, 2))
        X1 = rng.normal(loc=(10, 10), scale=0.5, size=(100, 2))
        X = np.vstack([X0, X1])

        model = KMeans(n_clusters=2, random_state=42).fit(X)

        true_centers = np.array([[-10.0, -10.0], [10.0, 10.0]])
        # each learned centroid should be close to one of the true centers,
        # regardless of which cluster index it was assigned
        for centroid in model.centroids_:
            distances = np.linalg.norm(true_centers - centroid, axis=1)
            assert np.min(distances) < 1.0

    def test_reproducible_with_same_random_state(self, rng):
        X0 = rng.normal(loc=(-5, -5), scale=1.0, size=(30, 2))
        X1 = rng.normal(loc=(5, 5), scale=1.0, size=(30, 2))
        X = np.vstack([X0, X1])

        model1 = KMeans(n_clusters=2, random_state=42).fit(X)
        model2 = KMeans(n_clusters=2, random_state=42).fit(X)

        assert np.allclose(model1.centroids_, model2.centroids_)


class TestInitialization:
    def test_single_cluster_centroid_is_the_data_mean(self, rng):
        X = rng.normal(loc=(3, -2), scale=2.0, size=(50, 2))

        model = KMeans(n_clusters=1, random_state=42).fit(X)

        assert np.allclose(model.centroids_[0], X.mean(axis=0))

    @pytest.mark.parametrize("seed", range(10))
    def test_kmeans_plus_plus_spreads_centroids_across_blobs(self, rng, seed):
        centers = np.array([[-30.0, 0.0], [0.0, 30.0], [30.0, 0.0]])
        X = np.vstack([rng.normal(loc=c, scale=0.5, size=(20, 2)) for c in centers])
        model = KMeans(n_clusters=3, random_state=seed)

        model._initialize_centroids(X)

        nearest_blob = [np.argmin(np.linalg.norm(centers - c, axis=1)) for c in model.centroids_]
        assert sorted(nearest_blob) == [0, 1, 2]

    def test_max_iter_is_reported_when_not_converged(self, rng):
        X = rng.normal(size=(60, 2))

        model = KMeans(n_clusters=3, max_iter=1, tol=-1.0, random_state=42).fit(X)

        assert model.n_iter_ == 1


class TestEmptyCluster:
    def test_empty_cluster_is_reinitialized_with_warning(self, rng, monkeypatch):
        X = np.vstack(
            [
                rng.normal(loc=(-10, -10), scale=0.5, size=(30, 2)),
                rng.normal(loc=(10, 10), scale=0.5, size=(30, 2)),
            ]
        )

        def far_away_second_centroid(self, X):
            self.centroids_ = np.array([[-10.0, -10.0], [1000.0, 1000.0]])

        monkeypatch.setattr(KMeans, "_initialize_centroids", far_away_second_centroid)

        with pytest.warns(RuntimeWarning, match="empty"):
            model = KMeans(n_clusters=2).fit(X)

        assert len(np.unique(model.labels_)) == 2
        assert np.all(np.abs(model.centroids_) < 20)

    def test_reinitialized_centroid_is_the_sample_farthest_from_the_others(self, rng, monkeypatch):
        X = rng.normal(size=(40, 2))

        def far_away_second_centroid(self, X):
            self.centroids_ = np.array([[0.0, 0.0], [1000.0, 1000.0]])

        monkeypatch.setattr(KMeans, "_initialize_centroids", far_away_second_centroid)

        with pytest.warns(RuntimeWarning):
            model = KMeans(n_clusters=2, max_iter=1, tol=-1.0).fit(X)

        # every sample was assigned to centroid 0, which then moved to the mean
        farthest_from_mean = X[np.argmax(np.linalg.norm(X - X.mean(axis=0), axis=1))]
        assert np.allclose(model.centroids_[1], farthest_from_mean)


class TestChunking:
    @pytest.mark.parametrize("chunk_size", [1, 7, 50, 1000])
    def test_chunked_assignment_matches_single_chunk(self, rng, monkeypatch, chunk_size):
        X = np.vstack(
            [
                rng.normal(loc=(-5, -5), scale=1.0, size=(40, 2)),
                rng.normal(loc=(5, 5), scale=1.0, size=(40, 2)),
                rng.normal(loc=(0, 8), scale=1.0, size=(40, 2)),
            ]
        )
        expected = KMeans(n_clusters=3, random_state=42).fit(X)

        monkeypatch.setattr("pyml.cluster.kmeans._CHUNK_SIZE", chunk_size)
        model = KMeans(n_clusters=3, random_state=42).fit(X)

        assert np.array_equal(model.labels_, expected.labels_)
        assert np.allclose(model.centroids_, expected.centroids_)

    def test_assignment_uses_chunks_of_the_configured_size(self, rng, monkeypatch):
        X = rng.normal(size=(23, 2))
        model = KMeans(n_clusters=2, random_state=42).fit(X)
        chunk_rows = []

        def spy(XA, XB, metric):
            chunk_rows.append(XA.shape[0])
            return cdist(XA, XB, metric)

        monkeypatch.setattr("pyml.cluster.kmeans._CHUNK_SIZE", 10)
        monkeypatch.setattr("pyml.cluster.kmeans.cdist", spy)

        model.predict(X)

        assert chunk_rows == [10, 10, 3]

    @pytest.mark.parametrize("chunk_size", [1, 7, 1000])
    def test_predict_on_training_data_equals_labels(self, rng, monkeypatch, chunk_size):
        X = rng.normal(size=(60, 2))
        model = KMeans(n_clusters=3, random_state=42).fit(X)

        monkeypatch.setattr("pyml.cluster.kmeans._CHUNK_SIZE", chunk_size)

        assert np.array_equal(model.predict(X), model.labels_)


class TestFitPredict:
    def test_matches_fit_then_labels(self, rng):
        X0 = rng.normal(loc=(-5, -5), scale=1.0, size=(30, 2))
        X1 = rng.normal(loc=(5, 5), scale=1.0, size=(30, 2))
        X = np.vstack([X0, X1])

        labels = KMeans(n_clusters=2, random_state=42).fit_predict(X)
        model = KMeans(n_clusters=2, random_state=42).fit(X)

        assert np.array_equal(labels, model.labels_)


class TestPredict:
    def test_raises_not_fitted_before_fit(self, rng):
        X = rng.uniform(0, 10, size=(5, 2))
        model = KMeans(n_clusters=2)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_assigns_new_point_to_nearest_centroid(self, rng):
        X0 = rng.normal(loc=(-10, -10), scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=(10, 10), scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])

        model = KMeans(n_clusters=2, random_state=42).fit(X)
        new_point = np.array([[-10.0, -10.0]])
        predicted_label = model.predict(new_point)[0]

        # the new point should be assigned the same label as the first blob
        assert predicted_label == model.labels_[0]
