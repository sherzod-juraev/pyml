import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from mlkit import KNNRegression
from mlkit.exc import NotFitted


# ─── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def data():
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=0.1,
        random_state=42
    )
    return X, y


@pytest.fixture
def fitted(data):
    X, y = data
    model = KNNRegression(k=5)
    model.fit(X, y)
    return model, X, y


# ════════════════════════════════════════════════════════════
#  Init
# ════════════════════════════════════════════════════════════

class TestInit:

    def test_default_k(self):
        assert KNNRegression().k == 3

    def test_default_metric(self):
        assert KNNRegression().metric == 'euclidean'

    def test_default_weighting(self):
        assert KNNRegression().weighting == 'uniform'

    def test_custom_params(self):
        model = KNNRegression(k=7, metric='cityblock', weighting='distance')
        assert model.k == 7
        assert model.metric == 'cityblock'
        assert model.weighting == 'distance'


# ════════════════════════════════════════════════════════════
#  Fit
# ════════════════════════════════════════════════════════════

class TestFit:

    def test_fit_returns_self(self, data):
        X, y = data
        model = KNNRegression()
        assert model.fit(X, y) is model

    def test_fit_stores_X(self, data):
        X, y = data
        model = KNNRegression()
        model.fit(X, y)
        np.testing.assert_array_equal(model.X_, X)

    def test_fit_stores_y(self, data):
        X, y = data
        model = KNNRegression()
        model.fit(X, y)
        np.testing.assert_array_equal(model.y_, y)

    def test_fit_converts_to_numpy(self, data):
        X, y = data
        model = KNNRegression()
        model.fit(X.tolist(), y.tolist())
        assert isinstance(model.X_, np.ndarray)
        assert isinstance(model.y_, np.ndarray)

    def test_refit_updates_data(self, data):
        X, y = data
        model = KNNRegression()
        model.fit(X, y)
        X2 = np.random.randn(*X.shape)
        y2 = np.random.randn(*y.shape)
        model.fit(X2, y2)
        np.testing.assert_array_equal(model.X_, X2)
        np.testing.assert_array_equal(model.y_, y2)


# ════════════════════════════════════════════════════════════
#  Predict
# ════════════════════════════════════════════════════════════

class TestPredict:

    def test_raises_not_fitted(self):
        model = KNNRegression()
        with pytest.raises(NotFitted):
            model.predict(np.random.randn(5, 3))

    def test_output_shape(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_output_is_float(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert preds.dtype == float or np.issubdtype(preds.dtype, np.floating)

    def test_1d_input_reshaped(self, fitted):
        model, X, y = fitted
        pred = model.predict(X[0])
        assert pred.shape == (1,)

    def test_no_nan_in_output(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))

    def test_no_inf_in_output(self, fitted):
        model, X, y = fitted
        preds = model.predict(X)
        assert not np.any(np.isinf(preds))


# ════════════════════════════════════════════════════════════
#  Correctness
# ════════════════════════════════════════════════════════════

class TestCorrectness:

    def test_perfect_fit_k1(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([10.0, 20.0, 30.0])
        model = KNNRegression(k=1)
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_almost_equal(preds, y)

    def test_uniform_mean_k3(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        model = KNNRegression(k=3, weighting='uniform')
        model.fit(X, y)
        pred = model.predict(np.array([[1.5]]))
        # 3 nearest: 1.0, 2.0, 0.0 → y: 20, 30, 10 → mean = 20
        assert pytest.approx(pred[0], abs=1e-6) == 20.0

    def test_reasonable_r2_score(self, data):
        X, y = data
        model = KNNRegression(k=5)
        model.fit(X, y)
        preds = model.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 >= 0.8


# ════════════════════════════════════════════════════════════
#  Weighting
# ════════════════════════════════════════════════════════════

class TestWeighting:

    @pytest.mark.parametrize('weighting', ['uniform', 'distance'])
    def test_both_weightings_run(self, data, weighting):
        X, y = data
        model = KNNRegression(k=5, weighting=weighting)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_distance_weighting_exact_match(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([5.0, 15.0, 25.0])
        model = KNNRegression(k=2, weighting='distance')
        model.fit(X, y)
        # Query = exact training point → o'sha nuqta dominant bo'ladi
        pred = model.predict(np.array([[0.0]]))
        assert pytest.approx(pred[0], abs=0.5) == 5.0

    def test_distance_weighting_closer_neighbor_dominates(self):
        X_train = np.array([[0.0], [1.0], [10.0]])
        y_train = np.array([100.0, 50.0, 0.0])
        model = KNNRegression(k=2, weighting='distance')
        model.fit(X_train, y_train)
        # Query = 0.1 → 0.0 ga yaqin → 100 ga yaqin natija
        pred_dist = model.predict(np.array([[0.1]]))[0]

        model_uni = KNNRegression(k=2, weighting='uniform')
        model_uni.fit(X_train, y_train)
        pred_uni = model_uni.predict(np.array([[0.1]]))[0]

        assert pred_dist > pred_uni


# ════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════

class TestMetrics:

    @pytest.mark.parametrize('metric', ['euclidean', 'cityblock', 'chebyshev'])
    def test_all_metrics_run(self, data, metric):
        X, y = data
        model = KNNRegression(k=5, metric=metric)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_euclidean_matches_sklearn(self, data):
        X, y = data
        model = KNNRegression(k=5, metric='euclidean', weighting='uniform')
        model.fit(X, y)
        preds = model.predict(X)

        sk = KNeighborsRegressor(n_neighbors=5, metric='euclidean', weights='uniform')
        sk.fit(X, y)
        sk_preds = sk.predict(X)

        np.testing.assert_array_almost_equal(preds, sk_preds, decimal=5)


# ════════════════════════════════════════════════════════════
#  K parameter
# ════════════════════════════════════════════════════════════

class TestKParameter:

    @pytest.mark.parametrize('k', [1, 3, 5, 10])
    def test_various_k_values(self, data, k):
        X, y = data
        model = KNNRegression(k=k)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_k1_memorizes_training(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([10.0, 20.0, 30.0])
        model = KNNRegression(k=1)
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_almost_equal(preds, y)

    def test_larger_k_smoother_predictions(self, data):
        X, y = data
        model_k1 = KNNRegression(k=1).fit(X, y)
        model_k10 = KNNRegression(k=10).fit(X, y)
        preds_k1 = model_k1.predict(X)
        preds_k10 = model_k10.predict(X)
        # k=1 → training data ga yopishadi → variance katta
        assert np.var(preds_k1) >= np.var(preds_k10)


# ════════════════════════════════════════════════════════════
#  Edge cases
# ════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_training_sample(self):
        X = np.array([[1.0, 2.0]])
        y = np.array([5.0])
        model = KNNRegression(k=1)
        model.fit(X, y)
        pred = model.predict(np.array([[1.0, 2.0]]))
        assert pytest.approx(pred[0]) == 5.0

    def test_high_dimensional(self):
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = KNNRegression(k=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_k_equals_n_samples(self):
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        model = KNNRegression(k=10)
        model.fit(X, y)
        preds = model.predict(X)
        # Hammasi bir xil — np.mean(y)
        np.testing.assert_array_almost_equal(preds, np.mean(y))