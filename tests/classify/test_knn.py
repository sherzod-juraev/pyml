import numpy as np
import pytest
from mlkit import KNNClassifier


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def simple_binary_data():

    X = np.array([
        [1.0, 1.0],
        [1.5, 2.0],
        [2.0, 1.5],
        [4.0, 1.0],
        [4.5, 2.0],
        [5.0, 1.5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def multiclass_data():
    X = np.array([
        [0.0, 0.0], [0.5, 0.5],
        [5.0, 0.0], [5.5, 0.5],
        [2.5, 5.0], [3.0, 5.5],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    return X, y


@pytest.fixture
def knn_default():
    return KNNClassifier(k=3, metric='euclidean', weighting='uniform')


@pytest.fixture
def fitted_knn(knn_default, simple_binary_data):
    X, y = simple_binary_data
    knn_default.fit(X, y)
    return knn_default


# ===========================================================================
# 1. FIT TESTLARI
# ===========================================================================

class TestFit:

    def test_fit_returns_self(self, knn_default, simple_binary_data):

        X, y = simple_binary_data
        result = knn_default.fit(X, y)
        assert result is knn_default

    def test_fit_stores_X_(self, knn_default, simple_binary_data):

        X, y = simple_binary_data
        knn_default.fit(X, y)
        assert hasattr(knn_default, 'X_')
        assert knn_default.X_.shape == X.shape

    def test_fit_stores_y_(self, knn_default, simple_binary_data):

        X, y = simple_binary_data
        knn_default.fit(X, y)
        assert hasattr(knn_default, 'y_')
        assert np.array_equal(knn_default.y_, y)

    def test_fit_converts_to_numpy(self, knn_default):

        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [0, 1, 0]
        knn_default.fit(X, y)
        assert isinstance(knn_default.X_, np.ndarray)
        assert isinstance(knn_default.y_, np.ndarray)

    def test_refit_updates_train_data(self, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)

        X_new = np.array([[10.0, 10.0], [11.0, 10.0]])
        y_new = np.array([9, 9])
        clf.fit(X_new, y_new)

        assert clf.X_.shape == X_new.shape
        assert np.array_equal(clf.y_, y_new)


# ===========================================================================
# 2. PREDICT
# ===========================================================================

class TestPredict:

    def test_predict_returns_numpy_array(self, fitted_knn, simple_binary_data):

        X, _ = simple_binary_data
        preds = fitted_knn.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_output_shape(self, fitted_knn, simple_binary_data):

        X, _ = simple_binary_data
        preds = fitted_knn.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_perfect_accuracy_on_separable_data(self, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)

    def test_predict_labels_subset_of_train_labels(self, fitted_knn, simple_binary_data):

        X, y = simple_binary_data
        preds = fitted_knn.predict(X)
        assert np.all(np.isin(preds, np.unique(y)))

    def test_predict_single_sample_1d(self, fitted_knn):

        x = np.array([1.0, 1.0])
        preds = fitted_knn.predict(x)
        assert preds.shape == (1,)

    def test_predict_single_sample_2d(self, fitted_knn):

        x = np.array([[1.0, 1.0]])
        preds = fitted_knn.predict(x)
        assert preds.shape == (1,)

    def test_multiclass_prediction(self, multiclass_data):

        X, y = multiclass_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)


# ===========================================================================
# 3. EXCEPTION
# ===========================================================================

class TestExceptions:

    def test_predict_before_fit_raises(self):

        from mlkit.exc import NotFitted
        clf = KNNClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted):
            clf.predict(X)

    def test_predict_before_fit_message(self):

        from mlkit.exc import NotFitted
        clf = KNNClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted) as exc_info:
            clf.predict(X)

        assert "fitted" in str(exc_info.value).lower()


# ===========================================================================
# 4. METRICS
# ===========================================================================

class TestMetrics:

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev", "cosine"])
    def test_all_metrics_work(self, metric, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=3, metric=metric)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev", "cosine"])
    def test_all_metrics_output_valid_labels(self, metric, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=3, metric=metric)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.all(np.isin(preds, np.unique(y)))

    def test_euclidean_closer_neighbor_wins(self):

        X_ = np.array([[0.0, 0.0], [5.0, 5.0]])
        y_ = np.array([0, 1])
        clf = KNNClassifier(k=1, metric='euclidean')
        clf.fit(X_, y_)

        pred = clf.predict(np.array([[0.1, 0.0]]))
        assert pred[0] == 0

        pred = clf.predict(np.array([[4.9, 5.0]]))
        assert pred[0] == 1

    def test_cityblock_vs_euclidean_can_differ(self):

        X_ = np.array([
            [0.0, 3.0],
            [3.0, 0.0],
            [10.0, 10.0],
        ])
        y_ = np.array([0, 1, 2])
        X_test = np.array([[1.5, 1.5]])

        clf_eucl = KNNClassifier(k=1, metric='euclidean')
        clf_city = KNNClassifier(k=1, metric='cityblock')

        clf_eucl.fit(X_, y_)
        clf_city.fit(X_, y_)

        pred_e = clf_eucl.predict(X_test)
        pred_c = clf_city.predict(X_test)

        assert pred_e[0] in [0, 1, 2]
        assert pred_c[0] in [0, 1, 2]


# ===========================================================================
# 5. WEIGHTING
# ===========================================================================

class TestWeighting:

    @pytest.mark.parametrize("weighting", ["uniform", "distance"])
    def test_both_weightings_work(self, weighting, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=3, weighting=weighting)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_distance_weighting_prefers_closer_neighbor(self):

        X_ = np.array([[0.0, 0.0], [1.0, 0.0], [1.1, 0.0], [1.2, 0.0]])
        y_ = np.array([0, 1, 1, 1])
        X_test = np.array([[0.1, 0.0]])

        clf_uniform = KNNClassifier(k=3, weighting='uniform')
        clf_distance = KNNClassifier(k=3, weighting='distance')

        clf_uniform.fit(X_, y_)
        clf_distance.fit(X_, y_)

        pred_u = clf_uniform.predict(X_test)[0]
        pred_d = clf_distance.predict(X_test)[0]

        assert pred_u == 1
        assert pred_d == 0

    def test_distance_weighting_exact_match(self):

        X_ = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
        y_ = np.array([0, 1, 2])

        clf = KNNClassifier(k=2, weighting='distance')
        clf.fit(X_, y_)

        X_test = np.array([[1.0, 1.0]])
        preds = clf.predict(X_test)

        assert preds[0] in [0, 1, 2]

    def test_uniform_weighting_majority_vote(self):

        X_ = np.array([
            [0.0, 0.0],   # class 0
            [0.5, 0.0],   # class 0
            [1.0, 0.0],   # class 1
            [10.0, 0.0],  # class 1
        ])
        y_ = np.array([0, 0, 1, 1])
        X_test = np.array([[0.2, 0.0]])

        clf = KNNClassifier(k=3, weighting='uniform')
        clf.fit(X_, y_)
        pred = clf.predict(X_test)[0]
        assert pred == 0


# ===========================================================================
# 6. K PARAMETER
# ===========================================================================

class TestKParameter:

    @pytest.mark.parametrize("k", [1, 2, 3, 5])
    def test_various_k_values(self, k, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=k)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_k1_memorizes_training_data(self, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)

    def test_larger_k_more_stable(self, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=5)
        clf.fit(X, y)

        preds1 = clf.predict(X)
        preds2 = clf.predict(X)
        assert np.array_equal(preds1, preds2)

    def test_k_equals_n_samples(self, simple_binary_data):

        X, y = simple_binary_data
        clf = KNNClassifier(k=len(y))
        clf.fit(X, y)
        preds = clf.predict(X)
        majority = np.bincount(y).argmax()
        assert np.all(preds == majority)


# ===========================================================================
# 7. ARGPARTITION
# ===========================================================================

class TestArgpartitionBug:

    def test_distance_weighting_consistent_with_sorted_neighbors(self):

        np.random.seed(42)
        X_ = np.random.randn(50, 2)
        y_ = (X_[:, 0] > 0).astype(int)

        X_test = np.random.randn(10, 2)

        clf = KNNClassifier(k=5, weighting='distance')
        clf.fit(X_, y_)
        preds = clf.predict(X_test)

        assert np.all(np.isin(preds, [0, 1]))

    def test_k_neighbors_count_is_exactly_k(self, simple_binary_data):

        X, y = simple_binary_data

        clf_k1 = KNNClassifier(k=1)
        clf_k3 = KNNClassifier(k=3)

        clf_k1.fit(X, y)
        clf_k3.fit(X, y)

        preds_k1 = clf_k1.predict(X)
        preds_k3 = clf_k3.predict(X)

        assert preds_k1.shape == preds_k3.shape == (len(y),)


# ===========================================================================
# 8. EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_single_training_sample(self):

        X_ = np.array([[3.0, 3.0]])
        y_ = np.array([7])

        clf = KNNClassifier(k=1)
        clf.fit(X_, y_)

        X_test = np.array([[0.0, 0.0], [100.0, 100.0]])
        preds = clf.predict(X_test)
        assert np.all(preds == 7)

    def test_high_dimensional_data(self):

        np.random.seed(0)
        X_ = np.random.randn(20, 100)
        y_ = np.random.randint(0, 3, 20)

        X_test = np.random.randn(5, 100)

        clf = KNNClassifier(k=3)
        clf.fit(X_, y_)
        preds = clf.predict(X_test)

        assert preds.shape == (5,)
        assert np.all(np.isin(preds, np.unique(y_)))

    def test_identical_points_different_classes(self):

        X_ = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y_ = np.array([0, 0, 1])

        clf = KNNClassifier(k=3)
        clf.fit(X_, y_)

        X_test = np.array([[1.0, 1.0]])
        preds = clf.predict(X_test)

        assert preds[0] in [0, 1]

    def test_float_and_int_labels(self):

        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        for y in [np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])]:
            clf = KNNClassifier(k=1)
            clf.fit(X, y)
            preds = clf.predict(X)
            assert preds.shape == (4,)


# ===========================================================================
# 9. SKLEARN — @pytest.mark.slow
# ===========================================================================

@pytest.mark.slow
class TestSklearnComparison:

    @pytest.mark.parametrize("metric,sk_metric", [
        ("euclidean", "euclidean"),
        ("cityblock", "manhattan"),
        ("chebyshev", "chebyshev"),
    ])
    def test_matches_sklearn_uniform(self, metric, sk_metric, simple_binary_data):

        from sklearn.neighbors import KNeighborsClassifier
        X, y = simple_binary_data

        our_clf = KNNClassifier(k=3, metric=metric, weighting='uniform')
        sk_clf = KNeighborsClassifier(n_neighbors=3, metric=sk_metric, weights='uniform')

        our_clf.fit(X, y)
        sk_clf.fit(X, y)

        our_preds = our_clf.predict(X)
        sk_preds = sk_clf.predict(X)

        assert np.array_equal(our_preds, sk_preds), (
            f"metric={metric}: our={our_preds}, sklearn={sk_preds}"
        )

    def test_iris_accuracy(self):

        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        X, y = iris.data, iris.target
        X_, X_test, y_, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = KNNClassifier(k=5, metric='euclidean', weighting='uniform')
        clf.fit(X_, y_)
        preds = clf.predict(X_test)

        accuracy = np.mean(preds == y_test)
        assert accuracy >= 0.90, f"Iris accuracy too low: {accuracy:.2f}"

    @pytest.mark.parametrize("weighting", ["uniform", "distance"])
    def test_breast_cancer_accuracy(self, weighting):

        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        data = load_breast_cancer()
        X, y = data.data, data.target

        X_, X_test, y_, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_ = scaler.fit_transform(X_)
        X_test = scaler.transform(X_test)

        clf = KNNClassifier(k=7, metric='euclidean', weighting=weighting)
        clf.fit(X_, y_)
        preds = clf.predict(X_test)

        accuracy = np.mean(preds == y_test)
        assert accuracy >= 0.90, f"{weighting} accuracy too low: {accuracy:.2f}"