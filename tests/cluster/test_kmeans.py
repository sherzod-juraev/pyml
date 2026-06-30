import numpy as np
import pytest

from sklearn.datasets import make_blobs

from pyml import Kmeans
from pyml.exc import NotFittedError


class TestBasicCorrectness:
    def test_two_well_separated_clusters(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [-0.2, 0.1],
                [10.0, 10.0],
                [10.2, 9.9],
                [9.8, 10.1],
            ]
        )

        model = Kmeans(
            k=2,
            random_state=42,
            init="kmeans++",
            max_iter=100,
        )

        model.fit(X)

        labels = model.predict(X)

        assert len(np.unique(labels)) == 2

        assert np.all(labels[:3] == labels[0])
        assert np.all(labels[3:] == labels[3])

        assert labels[0] != labels[3]

    def test_predict_training_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))

        model = Kmeans(k=4, random_state=0)
        model.fit(X)

        pred = model.predict(X)

        assert pred.shape == (100,)
        assert pred.dtype == np.intp

    def test_centroids_shape(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 5))

        model = Kmeans(k=3, random_state=1)
        model.fit(X)

        assert model.centroids_.shape == (3, 5)


class TestKmeansNotFitted:
    def test_predict_before_fit_raises(self):
        X = np.random.rand(10, 2)

        model = Kmeans(k=2)

        with pytest.raises(NotFittedError):
            model.predict(X)


class TestInitialization:
    def test_uniform_initialization_selects_training_points(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 2))

        model = Kmeans(
            k=5,
            init="uniform",
            random_state=42,
        )

        model.initialize_centroids_(X)

        for centroid in model.centroids_:
            assert np.any(np.all(X == centroid, axis=1))

    def test_kmeanspp_initialization_shape(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 4))

        model = Kmeans(
            k=4,
            init="kmeans++",
            random_state=7,
        )

        model.initialize_centroids_(X)

        assert model.centroids_.shape == (4, 4)

    def test_same_random_state_same_uniform_centroids(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(100, 3))

        model1 = Kmeans(
            k=4,
            init="uniform",
            random_state=123,
        )
        model2 = Kmeans(
            k=4,
            init="uniform",
            random_state=123,
        )

        model1.initialize_centroids_(X)
        model2.initialize_centroids_(X)

        assert np.allclose(model1.centroids_, model2.centroids_)

    def test_different_random_state_changes_uniform_centroids(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(100, 3))

        model1 = Kmeans(
            k=4,
            init="uniform",
            random_state=1,
        )
        model2 = Kmeans(
            k=4,
            init="uniform",
            random_state=2,
        )

        model1.initialize_centroids_(X)
        model2.initialize_centroids_(X)

        assert not np.allclose(model1.centroids_, model2.centroids_)


class TestPredict:
    def test_predict_before_fit_raises(self):
        model = Kmeans(k=2)

        with pytest.raises(NotFittedError):
            model.predict(np.array([[0.0, 0.0]]))

    def test_predict_returns_valid_cluster_indices(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(80, 2))

        model = Kmeans(
            k=5,
            random_state=42,
        )
        model.fit(X)

        labels = model.predict(X)

        assert np.all(labels >= 0)
        assert np.all(labels < 5)

    def test_predict_multiple_query_points(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [10.0, 10.0],
                [10.2, 10.1],
            ]
        )

        model = Kmeans(
            k=2,
            random_state=0,
        )
        model.fit(X)

        X_new = np.array(
            [
                [0.1, 0.0],
                [9.9, 10.1],
                [0.3, -0.1],
            ]
        )

        labels = model.predict(X_new)

        assert labels.shape == (3,)
        assert np.all(labels >= 0)
        assert np.all(labels < 2)

    def test_predict_is_deterministic_after_fit(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(100, 4))

        model = Kmeans(
            k=3,
            random_state=42,
        )
        model.fit(X)

        pred1 = model.predict(X)
        pred2 = model.predict(X)

        assert np.array_equal(pred1, pred2)


class TestDistanceMetrics:
    def test_euclidean_assigns_nearest_centroid(self):
        model = Kmeans(k=2, metric="euclidean")
        model.centroids_ = np.array([[0.0, 0.0], [10.0, 10.0]])

        X = np.array([[1.0, 1.0], [9.0, 9.0]])

        labels = model.cal_labels(X)

        assert np.array_equal(labels, np.array([0, 1]))

    def test_cityblock_assigns_nearest_centroid(self):
        model = Kmeans(k=2, metric="cityblock")
        model.centroids_ = np.array([[0.0, 0.0], [10.0, 10.0]])

        X = np.array([[2.0, 1.0], [8.0, 9.0]])

        labels = model.cal_labels(X)

        assert np.array_equal(labels, np.array([0, 1]))

    def test_chebyshev_assigns_nearest_centroid(self):
        model = Kmeans(k=2, metric="chebyshev")
        model.centroids_ = np.array([[0.0, 0.0], [10.0, 10.0]])

        X = np.array([[0.5, 0.4], [9.5, 9.2]])

        labels = model.cal_labels(X)

        assert np.array_equal(labels, np.array([0, 1]))

    def test_labels_shape_matches_number_of_samples(self):
        rng = np.random.default_rng(10)

        X = rng.normal(size=(50, 3))

        model = Kmeans(k=4, random_state=42)
        model.fit(X)

        labels = model.cal_labels(X)

        assert labels.shape == (50,)
        assert labels.dtype == np.intp


class TestConvergence:
    def test_converged_when_centroids_do_not_move(self):
        model = Kmeans(k=2, tol=1e-3)

        model.centroids_ = np.array(
            [
                [0.0, 0.0],
                [5.0, 5.0],
            ]
        )

        old = model.centroids_.copy()

        assert model.convergence(old)

    def test_not_converged_when_centroids_move(self):
        model = Kmeans(k=2, tol=1e-3)

        old = np.array(
            [
                [0.0, 0.0],
                [5.0, 5.0],
            ]
        )

        model.centroids_ = np.array(
            [
                [0.1, 0.0],
                [5.0, 5.0],
            ]
        )

        assert not model.convergence(old)

    @pytest.mark.parametrize(
        "metric",
        [
            "euclidean",
            "cityblock",
            "chebyshev",
        ],
    )
    def test_convergence_with_all_metrics(self, metric):
        model = Kmeans(
            k=2,
            metric=metric,
            tol=0.5,
        )

        old = np.array(
            [
                [0.0, 0.0],
                [5.0, 5.0],
            ]
        )

        model.centroids_ = np.array(
            [
                [0.1, 0.1],
                [5.1, 5.1],
            ]
        )

        assert model.convergence(old)


class TestEmptyClusterHandling:
    def test_empty_cluster_reinitialized(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [10.0, 10.0],
            ]
        )

        model = Kmeans(k=3)

        model.centroids_ = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [100.0, 100.0],
            ]
        )

        labels = np.array([0, 0, 1])

        with pytest.warns(RuntimeWarning):
            model.update_centroids_(X, labels)

        assert model.centroids_.shape == (3, 2)

    def test_update_centroids_computes_cluster_means(self):
        X = np.array(
            [
                [0.0, 0.0],
                [2.0, 2.0],
                [10.0, 10.0],
                [12.0, 12.0],
            ]
        )

        labels = np.array([0, 0, 1, 1])

        model = Kmeans(k=2)

        model.centroids_ = np.zeros((2, 2))

        model.update_centroids_(X, labels)

        expected = np.array(
            [
                [1.0, 1.0],
                [11.0, 11.0],
            ]
        )

        assert np.allclose(model.centroids_, expected)

    def test_every_centroid_remains_finite(self):
        rng = np.random.default_rng(11)

        X = rng.normal(size=(100, 4))

        model = Kmeans(
            k=5,
            random_state=42,
        )

        model.fit(X)

        assert np.all(np.isfinite(model.centroids_))


class TestRandomState:
    def test_same_random_state_produces_same_centroids(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(150, 3))

        model1 = Kmeans(
            k=4,
            random_state=42,
            init="kmeans++",
        )

        model2 = Kmeans(
            k=4,
            random_state=42,
            init="kmeans++",
        )

        model1.fit(X)
        model2.fit(X)

        assert np.allclose(model1.centroids_, model2.centroids_)

    def test_same_random_state_produces_same_labels(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(150, 2))

        model1 = Kmeans(
            k=3,
            random_state=123,
        )

        model2 = Kmeans(
            k=3,
            random_state=123,
        )

        labels1 = model1.fit(X).predict(X)
        labels2 = model2.fit(X).predict(X)

        assert np.array_equal(labels1, labels2)


class TestSanityCheck:
    def test_three_blobs_are_clustered_correctly(self):
        X, _ = make_blobs(
            n_samples=300,
            centers=3,
            cluster_std=0.5,
            random_state=42,
        )

        model = Kmeans(
            k=3,
            random_state=42,
        )

        labels = model.fit(X).predict(X)

        assert len(np.unique(labels)) == 3

    def test_predict_after_fit_has_correct_shape(self):
        X, _ = make_blobs(
            n_samples=120,
            centers=4,
            random_state=1,
        )

        model = Kmeans(
            k=4,
            random_state=1,
        )

        pred = model.fit(X).predict(X)

        assert pred.shape == (120,)
