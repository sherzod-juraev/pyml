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


class TestDensityReachability:
    def test_chain_of_core_points_forms_a_single_cluster(self):
        # neighbors are 1.0 apart, so only expansion through core points
        # can link the two ends of the chain (they are 9.0 apart)
        X = np.arange(10.0).reshape(-1, 1)

        labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(X)

        assert len(np.unique(labels)) == 1
        assert labels[0] != -1

    def test_border_point_joins_cluster_but_does_not_extend_it(self):
        # the point at 1.5 is within eps of a core point, but has too few
        # neighbors to be core itself; the point at 2.4 is reachable only
        # through that border point, so it must stay noise
        X = np.array([[0.0], [0.2], [0.4], [0.6], [1.5], [2.4]])

        labels = DBSCAN(eps=1.0, min_samples=4).fit_predict(X)

        assert labels[4] == labels[0]
        assert labels[0] != -1
        assert labels[5] == -1

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    def test_eps_boundary_is_inclusive(self, metric):
        # the two points are exactly eps apart under every metric
        X = np.array([[0.0, 0.0], [1.0, 0.0]])

        labels = DBSCAN(eps=1.0, min_samples=2, metric=metric).fit_predict(X)

        assert np.all(labels == 0)


class TestBlocking:
    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev"])
    @pytest.mark.parametrize("rows_per_block", [1, 7, 50, 1000])
    def test_blocked_fit_matches_single_block(self, rng, monkeypatch, metric, rows_per_block):
        X = np.vstack(
            [
                rng.normal(loc=(-5, -5), scale=0.8, size=(40, 2)),
                rng.normal(loc=(5, 5), scale=0.8, size=(40, 2)),
                rng.uniform(-10, 10, size=(10, 2)),
            ]
        )
        expected = DBSCAN(eps=1.2, min_samples=4, metric=metric).fit_predict(X)

        monkeypatch.setattr("pyml.cluster.dbscan._MAX_BLOCK_ELEMENTS", rows_per_block * X.shape[0])
        labels = DBSCAN(eps=1.2, min_samples=4, metric=metric).fit_predict(X)

        assert np.array_equal(labels, expected)

    def test_budget_smaller_than_one_row_still_fits(self, rng, monkeypatch):
        X = rng.normal(size=(30, 2))
        expected = DBSCAN(eps=0.8, min_samples=3).fit_predict(X)

        monkeypatch.setattr("pyml.cluster.dbscan._MAX_BLOCK_ELEMENTS", 1)

        assert np.array_equal(DBSCAN(eps=0.8, min_samples=3).fit_predict(X), expected)

    def test_neighborhoods_are_computed_in_blocks_of_the_budgeted_size(self, rng, monkeypatch):
        X = rng.normal(size=(23, 2))
        block_sizes = []
        original = DBSCAN._eps_neighborhoods

        def spy(self, X_block, X_all):
            block_sizes.append(X_block.shape[0])
            return original(self, X_block, X_all)

        monkeypatch.setattr(DBSCAN, "_eps_neighborhoods", spy)
        monkeypatch.setattr("pyml.cluster.dbscan._MAX_BLOCK_ELEMENTS", 10 * X.shape[0])

        DBSCAN(eps=0.8, min_samples=3).fit(X)

        assert block_sizes == [10, 10, 3]
