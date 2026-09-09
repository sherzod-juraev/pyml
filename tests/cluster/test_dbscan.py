"""Tests for DBSCAN."""

import numpy as np
import pytest

from pyml.cluster import DBSCAN
from pyml.core.exceptions import InvalidParameterError


class TestFitPredict:
    def test_rejects_zero_or_negative_eps(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))

        with pytest.raises(InvalidParameterError):
            DBSCAN(eps=0.0).fit_predict(X)

    def test_rejects_zero_or_negative_min_samples(self, rng):
        X = rng.uniform(0, 10, size=(10, 2))

        with pytest.raises(InvalidParameterError):
            DBSCAN(min_samples=0).fit_predict(X)

    def test_rejects_min_samples_exceeding_sample_count(self, rng):
        X = rng.uniform(0, 10, size=(5, 2))

        with pytest.raises(InvalidParameterError):
            DBSCAN(min_samples=10).fit_predict(X)

    def test_finds_well_separated_clusters(self, rng):
        X0 = rng.normal(loc=(-10, -10), scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=(10, 10), scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1])

        labels = DBSCAN(eps=2.0, min_samples=5).fit_predict(X)

        labels_0 = labels[:30]
        labels_1 = labels[30:]
        assert len(np.unique(labels_0)) == 1
        assert len(np.unique(labels_1)) == 1
        assert labels_0[0] != labels_1[0]
        # neither blob should be marked as noise
        assert labels_0[0] != -1
        assert labels_1[0] != -1

    def test_marks_isolated_points_as_noise(self, rng):
        X0 = rng.normal(loc=(0, 0), scale=0.3, size=(30, 2))
        isolated = np.array([[50.0, 50.0]])
        X = np.vstack([X0, isolated])

        labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(X)

        assert labels[-1] == -1

    def test_too_small_eps_marks_everything_as_noise(self, rng):
        X = rng.uniform(0, 100, size=(30, 2))

        labels = DBSCAN(eps=1e-6, min_samples=5).fit_predict(X)

        assert np.all(labels == -1)

    def test_no_leftover_unvisited_label(self, rng):
        X0 = rng.normal(loc=(-5, -5), scale=1.0, size=(20, 2))
        X1 = rng.normal(loc=(5, 5), scale=1.0, size=(20, 2))
        X = np.vstack([X0, X1])

        labels = DBSCAN(eps=1.5, min_samples=5).fit_predict(X)

        # every point should end up either in a cluster (>= 0) or noise (-1),
        # never left in the internal "unvisited" (-2) state
        assert np.all(labels != -2)


class TestGetLabels:
    def test_matches_fit_predict_result(self, rng):
        X0 = rng.normal(loc=(-5, -5), scale=1.0, size=(20, 2))
        X1 = rng.normal(loc=(5, 5), scale=1.0, size=(20, 2))
        X = np.vstack([X0, X1])

        model = DBSCAN(eps=1.5, min_samples=5)
        labels = model.fit_predict(X)

        assert np.array_equal(model.labels_, labels)
