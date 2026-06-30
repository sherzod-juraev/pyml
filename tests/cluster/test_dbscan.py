import numpy as np
import pytest

from pyml import DBSCAN


class TestDBSCAN:
    def test_init(self):
        model = DBSCAN(eps=0.5, MinPts=4, metric="cityblock")

        assert model.eps == 0.5
        assert model.MinPts == 4
        assert model.metric == "cityblock"

    def test_invalid_metric(self):
        model = DBSCAN(eps=0.5, metric="invalid")

        with pytest.raises(ValueError, match="Unsupported metric"):
            model.fit_predict(np.random.rand(5, 2))

    def test_single_cluster(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])

        labels = DBSCAN(eps=1.5, MinPts=2).fit_predict(X)

        assert np.all(labels == 0)

    def test_two_clusters(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [10., 10.],
            [10., 11.]
        ])

        labels = DBSCAN(eps=1.5, MinPts=2).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 0, 1, 1]))

    def test_detects_noise(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [10., 10.]
        ])

        labels = DBSCAN(eps=1.5, MinPts=2).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 0, -1]))

    def test_all_noise(self):
        X = np.array([
            [0., 0.],
            [10., 10.],
            [20., 20.]
        ])

        labels = DBSCAN(eps=1.0, MinPts=2).fit_predict(X)

        assert np.all(labels == -1)

    def test_single_sample(self):
        X = np.array([[1., 2.]])

        labels = DBSCAN(eps=1.0, MinPts=2).fit_predict(X)

        assert np.array_equal(labels, np.array([-1]))

    def test_duplicate_points(self):
        X = np.array([
            [1., 1.],
            [1., 1.],
            [1., 1.]
        ])

        labels = DBSCAN(eps=0.1, MinPts=2).fit_predict(X)

        assert np.all(labels == 0)

    def test_euclidean_metric(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [5., 5.]
        ])

        labels = DBSCAN(
            eps=1.5,
            MinPts=2,
            metric="euclidean"
        ).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 0, -1]))

    def test_cityblock_metric(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [5., 5.]
        ])

        labels = DBSCAN(
            eps=2.0,
            MinPts=2,
            metric="cityblock"
        ).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 0, -1]))

    def test_chebyshev_metric(self):
        X = np.array([
            [0., 0.],
            [1., 1.],
            [5., 5.]
        ])

        labels = DBSCAN(
            eps=1.5,
            MinPts=2,
            metric="chebyshev"
        ).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 0, -1]))

    def test_output_shape(self):
        X = np.random.rand(20, 4)

        labels = DBSCAN(eps=0.5).fit_predict(X)

        assert labels.shape == (20,)

    def test_output_dtype(self):
        X = np.random.rand(10, 2)

        labels = DBSCAN(eps=0.5).fit_predict(X)

        assert np.issubdtype(labels.dtype, np.integer)

    def test_check_params(self):
        model = DBSCAN(eps=0.5, metric="euclidean")

        model.check_params()

    def test_empty_dataset(self):
        X = np.empty((0, 2))

        labels = DBSCAN(eps=1.0).fit_predict(X)

        assert labels.size == 0
        assert labels.shape == (0,)

    def test_minpts_one(self):
        X = np.array([
            [0., 0.],
            [10., 10.]
        ])

        labels = DBSCAN(eps=0.1, MinPts=1).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 1]))

    def test_three_separate_clusters(self):
        X = np.array([
            [0., 0.],
            [0., 0.1],

            [10., 10.],
            [10., 10.1],

            [20., 20.],
            [20., 20.1]
        ])

        labels = DBSCAN(eps=0.5, MinPts=2).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 0, 1, 1, 2, 2]))

    def test_fit_predict_is_deterministic(self):
        rng = np.random.default_rng(42)
        X = rng.random((30, 2))

        model = DBSCAN(eps=0.25, MinPts=3)

        labels1 = model.fit_predict(X)
        labels2 = model.fit_predict(X)

        assert np.array_equal(labels1, labels2)

    def test_cluster_labels_start_from_zero(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [10., 10.],
            [10., 11.]
        ])

        labels = DBSCAN(eps=1.5, MinPts=2).fit_predict(X)

        clusters = sorted(set(labels) - {-1})

        assert clusters == [0, 1]

    def test_all_points_in_single_cluster(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
            [0.5, 0.5]
        ])

        labels = DBSCAN(eps=2.0, MinPts=2).fit_predict(X)

        assert np.unique(labels).tolist() == [0]

    def test_noise_and_cluster(self):
        X = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [10., 10.]
        ])

        labels = DBSCAN(eps=1.5, MinPts=2).fit_predict(X)

        assert np.array_equal(labels, np.array([0, 0, 0, -1]))
