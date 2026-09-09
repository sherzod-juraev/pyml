"""Tests for KMeans."""

import numpy as np
import pytest

from pyml.cluster import KMeans
from pyml.core.exceptions import InvalidParameterError


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
        from pyml.core.exceptions import NotFittedError

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
