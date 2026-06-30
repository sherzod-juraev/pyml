import numpy as np
import pytest

from pyml import KNNRegressor
from pyml.exc import NotFittedError


class TestBasicCorrectness:
    def test_uniform_mean_of_k_neighbors(self):
        X_train = np.array([[1.0], [2.0], [3.0], [100.0]])
        y_train = np.array([10.0, 20.0, 30.0, 1000.0])

        model = KNNRegressor(k=3, weighting="uniform")
        model.fit(X_train, y_train)

        pred = model.predict(np.array([[2.0]]))
        assert pred[0] == pytest.approx(20.0)

    def test_k1_returns_exact_neighbor_value(self):
        X_train = np.array([[0.0], [5.0], [10.0]])
        y_train = np.array([100.0, 200.0, 300.0])

        model = KNNRegressor(k=1)
        model.fit(X_train, y_train)

        pred = model.predict(np.array([[5.1]]))
        assert pred[0] == pytest.approx(200.0)

    def test_k1_on_training_data_is_exact(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        y = rng.normal(size=20)

        model = KNNRegressor(k=1)
        model.fit(X, y)
        preds = model.predict(X)

        assert np.allclose(preds, y)

    def test_multiple_query_points(self):
        X_train = np.array([[0.0], [10.0]])
        y_train = np.array([0.0, 100.0])

        model = KNNRegressor(k=1)
        model.fit(X_train, y_train)

        preds = model.predict(np.array([[0.5], [9.5]]))
        assert preds.shape == (2,)
        assert preds[0] == pytest.approx(0.0)
        assert preds[1] == pytest.approx(100.0)


class TestKValidation:
    def test_k_zero_raises(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        model = KNNRegressor(k=0)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(np.array([[0.5]]))

    def test_k_negative_raises(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        model = KNNRegressor(k=-1)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(np.array([[0.5]]))

    def test_k_exceeds_n_samples_raises(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        model = KNNRegressor(k=10)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(np.array([[0.5]]))

    def test_k_equals_n_samples_is_valid(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 2.0])
        model = KNNRegressor(k=3)
        model.fit(X, y)
        pred = model.predict(np.array([[0.0]]))
        assert pred[0] == pytest.approx(1.0)


class TestNotFitted:
    def test_predict_before_fit_raises(self):
        model = KNNRegressor(k=3)
        with pytest.raises(NotFittedError):
            model.predict(np.array([[0.0]]))


class TestDistanceMetrics:
    def test_cityblock_changes_nearest_neighbor(self):
        X_train = np.array([[5.0, 0.0], [3.0, 3.0]])
        y_train = np.array([100.0, 200.0])

        model_euclidean = KNNRegressor(k=1, metric="euclidean")
        model_euclidean.fit(X_train, y_train)
        pred_euclidean = model_euclidean.predict(np.array([[0.0, 0.0]]))

        model_cityblock = KNNRegressor(k=1, metric="cityblock")
        model_cityblock.fit(X_train, y_train)
        pred_cityblock = model_cityblock.predict(np.array([[0.0, 0.0]]))

        assert pred_euclidean[0] == pytest.approx(200.0)
        assert pred_cityblock[0] == pytest.approx(100.0)

    def test_chebyshev_uses_max_coordinate_diff(self):
        X_train = np.array([[1.0, 9.0], [5.0, 5.0]])
        y_train = np.array([10.0, 20.0])

        model = KNNRegressor(k=1, metric="chebyshev")
        model.fit(X_train, y_train)
        pred = model.predict(np.array([[0.0, 0.0]]))
        assert pred[0] == pytest.approx(20.0)


class TestWeighting:
    def test_uniform_vs_distance_give_different_results(self):
        X_train = np.array([
            [5.0, 0.0],   # target=100, distance=5
            [-5.0, 0.0],  # target=100, distance=5
            [0.1, 0.0],   # target=1,   distance=0.1
        ])
        y_train = np.array([100.0, 100.0, 1.0])

        model_uniform = KNNRegressor(k=3, weighting="uniform")
        model_uniform.fit(X_train, y_train)
        pred_uniform = model_uniform.predict(np.array([[0.0, 0.0]]))

        model_distance = KNNRegressor(k=3, weighting="distance")
        model_distance.fit(X_train, y_train)
        pred_distance = model_distance.predict(np.array([[0.0, 0.0]]))

        assert pred_uniform[0] == pytest.approx(67.0, abs=1.0)
        assert pred_distance[0] < pred_uniform[0]
        assert pred_distance[0] < 20.0

    def test_distance_weighting_exact_match_dominates(self):
        X_train = np.array([
            [0.0, 0.0],   # target=5.0, distance=0 (exact match)
            [1.0, 0.0],   # target=1000.0
            [2.0, 0.0],   # target=2000.0
        ])
        y_train = np.array([5.0, 1000.0, 2000.0])

        model = KNNRegressor(k=3, weighting="distance")
        model.fit(X_train, y_train)
        pred = model.predict(np.array([[0.0, 0.0]]))
        assert pred[0] == pytest.approx(5.0, abs=1e-6)

    def test_distance_weighted_mean_formula_directly(self):
        X_train = np.array([[0.0], [1.0], [3.0]])
        y_train = np.array([10.0, 20.0, 30.0])

        model = KNNRegressor(k=3, weighting="distance")
        model.fit(X_train, y_train)
        pred = model.predict(np.array([[0.0]]))
        assert pred[0] == pytest.approx(10.0, abs=1e-6)


class TestInputShapes:
    def test_single_1d_query_point_is_reshaped(self):
        X_train = np.array([[0.0], [10.0]])
        y_train = np.array([0.0, 100.0])

        model = KNNRegressor(k=1)
        model.fit(X_train, y_train)

        pred = model.predict(np.array([0.5]))  # 1-D
        assert pred.shape == (1,)
        assert pred[0] == pytest.approx(0.0)

    def test_output_shape_matches_n_queries(self):
        X_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([0.0, 1.0, 2.0])

        model = KNNRegressor(k=2)
        model.fit(X_train, y_train)

        preds = model.predict(np.array([[0.0], [1.0], [2.0], [3.0]]))
        assert preds.shape == (4,)


class TestSanityCheckWithRealDataset:
    def test_diabetes_k1_perfect_on_training_set(self):
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)
        model = KNNRegressor(k=1)
        model.fit(X, y)
        preds = model.predict(X)

        assert np.allclose(preds, y)

    def test_diabetes_reasonable_r2_with_train_test_split(self):
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = KNNRegressor(k=10, weighting="distance")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        baseline_mse = np.mean((y_test - np.mean(y_train)) ** 2)
        model_mse = np.mean((preds - y_test) ** 2)
        assert model_mse < baseline_mse
