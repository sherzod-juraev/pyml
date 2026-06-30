import numpy as np
import pytest

from pyml import KNNClassifier
from pyml.exc import NotFittedError


class TestBasicCorrectness:
    def test_two_well_separated_clusters(self):
        X_train = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],   # class 0
            [10.0, 10.0], [11.0, 10.0], [10.0, 11.0],  # class 1
        ])
        y_train = np.array([0, 0, 0, 1, 1, 1])

        model = KNNClassifier(k=3, metric="euclidean", weighting="uniform")
        model.fit(X_train, y_train)

        pred = model.predict(np.array([[0.5, 0.5]]))
        assert pred[0] == 0

        pred = model.predict(np.array([[10.5, 10.5]]))
        assert pred[0] == 1

    def test_multiple_query_points_at_once(self):
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
        y_train = np.array([0, 0, 1, 1])

        model = KNNClassifier(k=1, metric="euclidean")
        model.fit(X_train, y_train)

        preds = model.predict(np.array([[0.1, 0.1], [10.1, 10.1]]))
        assert preds.shape == (2,)
        assert preds[0] == 0
        assert preds[1] == 1

    def test_k1_on_training_data_is_perfect(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 4))
        y = rng.integers(0, 3, size=30)

        model = KNNClassifier(k=1)
        model.fit(X, y)
        preds = model.predict(X)

        assert np.array_equal(preds, y)


class TestKValidation:
    def test_k_zero_raises(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        model = KNNClassifier(k=0)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(np.array([[0.5, 0.5]]))

    def test_k_negative_raises(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        model = KNNClassifier(k=-3)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(np.array([[0.5, 0.5]]))

    def test_k_exceeds_n_samples_raises(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        model = KNNClassifier(k=5)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(np.array([[0.5, 0.5]]))

    def test_k_equals_n_samples_is_valid(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y = np.array([0, 1, 1])
        model = KNNClassifier(k=3)
        model.fit(X, y)
        pred = model.predict(np.array([[0.0, 0.0]]))
        assert pred.shape == (1,)


class TestNotFitted:
    def test_predict_before_fit_raises(self):
        model = KNNClassifier(k=3)
        with pytest.raises(NotFittedError):
            model.predict(np.array([[0.0, 0.0]]))


class TestDistanceMetrics:
    def test_euclidean_matches_manual_calc(self):
        X_train = np.array([[3.0, 4.0], [100.0, 100.0]])
        y_train = np.array([0, 1])

        model = KNNClassifier(k=1, metric="euclidean")
        model.fit(X_train, y_train)
        pred = model.predict(np.array([[0.0, 0.0]]))
        assert pred[0] == 0

    def test_cityblock_changes_nearest_neighbor(self):
        X_train = np.array([[5.0, 0.0], [3.0, 3.0]])
        y_train = np.array([0, 1])

        model_euclidean = KNNClassifier(k=1, metric="euclidean")
        model_euclidean.fit(X_train, y_train)
        pred_euclidean = model_euclidean.predict(np.array([[0.0, 0.0]]))

        model_cityblock = KNNClassifier(k=1, metric="cityblock")
        model_cityblock.fit(X_train, y_train)
        pred_cityblock = model_cityblock.predict(np.array([[0.0, 0.0]]))

        assert pred_euclidean[0] == 1
        assert pred_cityblock[0] == 0

    def test_chebyshev_uses_max_coordinate_diff(self):
        X_train = np.array([[1.0, 9.0], [5.0, 5.0]])
        y_train = np.array([0, 1])

        model = KNNClassifier(k=1, metric="chebyshev")
        model.fit(X_train, y_train)
        pred = model.predict(np.array([[0.0, 0.0]]))
        assert pred[0] == 1

    def test_cosine_ignores_magnitude(self):
        X_train = np.array([[10.0, 10.0], [1.0, -1.0]])
        y_train = np.array([0, 1])

        model = KNNClassifier(k=1, metric="cosine")
        model.fit(X_train, y_train)
        pred = model.predict(np.array([[1.0, 1.0]]))
        assert pred[0] == 0


class TestWeighting:
    def test_uniform_vs_distance_give_different_results(self):
        X_train = np.array([
            [5.0, 0.0],   # class 0, distance = 5
            [-5.0, 0.0],  # class 0, distance = 5
            [0.1, 0.0],   # class 1, distance = 0.1
        ])
        y_train = np.array([0, 0, 1])

        model_uniform = KNNClassifier(k=3, weighting="uniform")
        model_uniform.fit(X_train, y_train)
        pred_uniform = model_uniform.predict(np.array([[0.0, 0.0]]))

        model_distance = KNNClassifier(k=3, weighting="distance")
        model_distance.fit(X_train, y_train)
        pred_distance = model_distance.predict(np.array([[0.0, 0.0]]))

        assert pred_uniform[0] == 0
        assert pred_distance[0] == 1

    def test_distance_weighting_exact_match_dominates(self):
        X_train = np.array([
            [0.0, 0.0],   # class 1
            [1.0, 0.0],   # class 0
            [2.0, 0.0],   # class 0
        ])
        y_train = np.array([1, 0, 0])

        model = KNNClassifier(k=3, weighting="distance")
        model.fit(X_train, y_train)
        pred = model.predict(np.array([[0.0, 0.0]]))
        assert pred[0] == 1


class TestInputShapes:
    def test_single_1d_query_point_is_reshaped(self):
        X_train = np.array([[0.0, 0.0], [10.0, 10.0]])
        y_train = np.array([0, 1])

        model = KNNClassifier(k=1)
        model.fit(X_train, y_train)

        pred = model.predict(np.array([0.5, 0.5]))  # 1-D input
        assert pred.shape == (1,)
        assert pred[0] == 0

    def test_2d_query_with_single_row_works_too(self):
        X_train = np.array([[0.0, 0.0], [10.0, 10.0]])
        y_train = np.array([0, 1])

        model = KNNClassifier(k=1)
        model.fit(X_train, y_train)

        pred = model.predict(np.array([[0.5, 0.5]]))  # 2-D, 1 qator
        assert pred.shape == (1,)
        assert pred[0] == 0


class TestSanityCheckWithRealDataset:
    def test_iris_k1_perfect_on_training_set(self):
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        model = KNNClassifier(k=1)
        model.fit(X, y)
        preds = model.predict(X)

        assert np.array_equal(preds, y)

    def test_iris_reasonable_accuracy_with_train_test_split(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = KNNClassifier(k=5, weighting="distance")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = np.mean(preds == y_test)
        assert accuracy > 0.8
