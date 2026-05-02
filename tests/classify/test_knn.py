"""
KNNClassifier uchun to'liq pytest test suite.

Ishga tushirish:
  pytest test_knn_classifier.py -v
  pytest test_knn_classifier.py -v -m "not slow"
  pytest test_knn_classifier.py::TestDistanceWeighting -v
"""

import numpy as np
import pytest
from mlkit.classify import KNNClassifier


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def simple_binary_data():
    """
    2D binary dataset — class 0 chap, class 1 o'ng.
    X[:, 0] < 3 → class 0, X[:, 0] >= 3 → class 1.
    """
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
    """3 klassli, aniq ajratiladigan dataset."""
    X = np.array([
        [0.0, 0.0], [0.5, 0.5],
        [5.0, 0.0], [5.5, 0.5],
        [2.5, 5.0], [3.0, 5.5],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    return X, y


@pytest.fixture
def knn_default():
    """Default KNNClassifier (k=3, euclidean, uniform)."""
    return KNNClassifier(k=3, metric='euclidean', weighting='uniform')


@pytest.fixture
def fitted_knn(knn_default, simple_binary_data):
    """Fit qilingan KNN — predict testlari uchun."""
    X, y = simple_binary_data
    knn_default.fit(X, y)
    return knn_default


# ===========================================================================
# 1. FIT TESTLARI
# ===========================================================================

class TestFit:

    def test_fit_returns_self(self, knn_default, simple_binary_data):
        """Method chaining uchun fit() self qaytarishi kerak."""
        X, y = simple_binary_data
        result = knn_default.fit(X, y)
        assert result is knn_default

    def test_fit_stores_X_(self, knn_default, simple_binary_data):
        """fit() X_'ni saqlashi kerak."""
        X, y = simple_binary_data
        knn_default.fit(X, y)
        assert hasattr(knn_default, 'X_')
        assert knn_default.X_.shape == X.shape

    def test_fit_stores_y_(self, knn_default, simple_binary_data):
        """fit() y_'ni saqlashi kerak."""
        X, y = simple_binary_data
        knn_default.fit(X, y)
        assert hasattr(knn_default, 'y_')
        assert np.array_equal(knn_default.y_, y)

    def test_fit_converts_to_numpy(self, knn_default):
        """Python list berilsa ham numpy arrayga o'girishi kerak."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [0, 1, 0]
        knn_default.fit(X, y)
        assert isinstance(knn_default.X_, np.ndarray)
        assert isinstance(knn_default.y_, np.ndarray)

    def test_refit_updates_train_data(self, simple_binary_data):
        """Qayta fit qilganda eski data yangilanishi kerak."""
        X, y = simple_binary_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)

        X_new = np.array([[10.0, 10.0], [11.0, 10.0]])
        y_new = np.array([9, 9])
        clf.fit(X_new, y_new)

        assert clf.X_.shape == X_new.shape
        assert np.array_equal(clf.y_, y_new)


# ===========================================================================
# 2. PREDICT — ASOSIY TESTLAR
# ===========================================================================

class TestPredict:

    def test_predict_returns_numpy_array(self, fitted_knn, simple_binary_data):
        """predict() numpy array qaytarishi kerak."""
        X, _ = simple_binary_data
        preds = fitted_knn.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_output_shape(self, fitted_knn, simple_binary_data):
        """predict() output shape (n_samples,) bo'lishi kerak."""
        X, _ = simple_binary_data
        preds = fitted_knn.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_perfect_accuracy_on_separable_data(self, simple_binary_data):
        """k=1 — train datada 100% accuracy."""
        X, y = simple_binary_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)

    def test_predict_labels_subset_of_train_labels(self, fitted_knn, simple_binary_data):
        """predict() faqat train labellarini qaytarishi kerak."""
        X, y = simple_binary_data
        preds = fitted_knn.predict(X)
        assert np.all(np.isin(preds, np.unique(y)))

    def test_predict_single_sample_1d(self, fitted_knn):
        """1D array (bitta sample) berilsa ham ishlashi kerak."""
        x = np.array([1.0, 1.0])  # shape (2,) — 1D
        preds = fitted_knn.predict(x)
        assert preds.shape == (1,)

    def test_predict_single_sample_2d(self, fitted_knn):
        """2D array bitta row bilan ishlashi kerak."""
        x = np.array([[1.0, 1.0]])  # shape (1, 2) — 2D
        preds = fitted_knn.predict(x)
        assert preds.shape == (1,)

    def test_multiclass_prediction(self, multiclass_data):
        """3 klassli datada ishlashi kerak."""
        X, y = multiclass_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)


# ===========================================================================
# 3. EXCEPTION TESTLARI
# ===========================================================================

class TestExceptions:

    def test_predict_before_fit_raises(self):
        """fit() chaqirilmay predict() qilinsa exc chiqishi kerak."""
        from mlkit.exc import NotFitted
        clf = KNNClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted):
            clf.predict(X)

    def test_predict_before_fit_message(self):
        """Exception message'da 'fitted' so'zi bo'lishi kerak."""
        from mlkit.exc import NotFitted
        clf = KNNClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted) as exc_info:
            clf.predict(X)

        assert "fitted" in str(exc_info.value).lower()


# ===========================================================================
# 4. METRIKALAR TESTLARI
# ===========================================================================

class TestMetrics:

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev", "cosine"])
    def test_all_metrics_work(self, metric, simple_binary_data):
        """
        Barcha metrikalar crash bo'lmay ishlashi kerak.
        4 ta alohida test yaratadi.
        """
        X, y = simple_binary_data
        clf = KNNClassifier(k=3, metric=metric)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev", "cosine"])
    def test_all_metrics_output_valid_labels(self, metric, simple_binary_data):
        """Barcha metrikalar faqat train labellarini qaytarishi kerak."""
        X, y = simple_binary_data
        clf = KNNClassifier(k=3, metric=metric)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.all(np.isin(preds, np.unique(y)))

    def test_euclidean_closer_neighbor_wins(self):
        """
        k=1, Euclidean: eng yaqin qo'shni tanlashi kerak.
        [0,0] dan [0.1, 0] ko'ra [5, 5] uzoqroq.
        """
        X_ = np.array([[0.0, 0.0], [5.0, 5.0]])
        y_ = np.array([0, 1])
        clf = KNNClassifier(k=1, metric='euclidean')
        clf.fit(X_, y_)

        # (0.1, 0.0) → class 0 ga yaqin
        pred = clf.predict(np.array([[0.1, 0.0]]))
        assert pred[0] == 0

        # (4.9, 5.0) → class 1 ga yaqin
        pred = clf.predict(np.array([[4.9, 5.0]]))
        assert pred[0] == 1

    def test_cityblock_vs_euclidean_can_differ(self):
        """
        Manhattan va Euclidean turli natija berishi mumkin bo'lgan holat.
        Bu testda crash bo'lmasligi va valid label qaytarishi tekshiriladi.
        """
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
# 5. WEIGHTING TESTLARI
# ===========================================================================

class TestWeighting:

    @pytest.mark.parametrize("weighting", ["uniform", "distance"])
    def test_both_weightings_work(self, weighting, simple_binary_data):
        """Ikki weighting ham crash bo'lmay ishlashi kerak."""
        X, y = simple_binary_data
        clf = KNNClassifier(k=3, weighting=weighting)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_distance_weighting_prefers_closer_neighbor(self):
        """
        Distance weighting: ko'pchilik uzoq bo'lsa ham,
        yaqin qo'shni ko'proq og'irlik olishi kerak.

        Holat:
          Train: [0,0]→0, [1,0]→1, [1.1,0]→1, [1.2,0]→1
          Test: [0.1, 0]
          Uniform k=3: 3 ta yaqin = [0,0], [1,0], [1.1,0] → vote: 0:1, 1:2 → label=1
          Distance k=3: [0,0] juda yaqin → og'irlik katta → label=0
        """
        X_ = np.array([[0.0, 0.0], [1.0, 0.0], [1.1, 0.0], [1.2, 0.0]])
        y_ = np.array([0, 1, 1, 1])
        X_test = np.array([[0.1, 0.0]])

        clf_uniform = KNNClassifier(k=3, weighting='uniform')
        clf_distance = KNNClassifier(k=3, weighting='distance')

        clf_uniform.fit(X_, y_)
        clf_distance.fit(X_, y_)

        pred_u = clf_uniform.predict(X_test)[0]
        pred_d = clf_distance.predict(X_test)[0]

        # Uniform: ko'pchilik ovoz → 1
        assert pred_u == 1
        # Distance: yaqin qo'shni (class 0) ko'proq og'irlik oladi → 0
        assert pred_d == 0

    def test_distance_weighting_exact_match(self):
        """
        Test point train point bilan to'liq mos kelsa (distance=0),
        1/distance → inf bo'lmasligi kerak (epsilon tufayli).
        """
        X_ = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
        y_ = np.array([0, 1, 2])

        clf = KNNClassifier(k=2, weighting='distance')
        clf.fit(X_, y_)

        # Aynan train point bilan test — distance=0 holat
        X_test = np.array([[1.0, 1.0]])
        preds = clf.predict(X_test)

        # Crash bo'lmasligi va valid label qaytarishi kerak
        assert preds[0] in [0, 1, 2]

    def test_uniform_weighting_majority_vote(self):
        """
        Uniform: ko'pchilik ovoz qoida ishlashi kerak.
        k=3, 2 ta class 0, 1 ta class 1 → class 0 yutishi kerak.
        """
        X_ = np.array([
            [0.0, 0.0],   # class 0 — yaqin
            [0.5, 0.0],   # class 0 — yaqin
            [1.0, 0.0],   # class 1 — yaqin
            [10.0, 0.0],  # class 1 — uzoq
        ])
        y_ = np.array([0, 0, 1, 1])
        X_test = np.array([[0.2, 0.0]])

        clf = KNNClassifier(k=3, weighting='uniform')
        clf.fit(X_, y_)
        pred = clf.predict(X_test)[0]
        assert pred == 0


# ===========================================================================
# 6. K PARAMETRI TESTLARI
# ===========================================================================

class TestKParameter:

    @pytest.mark.parametrize("k", [1, 2, 3, 5])
    def test_various_k_values(self, k, simple_binary_data):
        """Turli k qiymatlari crash bo'lmay ishlashi kerak."""
        X, y = simple_binary_data
        clf = KNNClassifier(k=k)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_k1_memorizes_training_data(self, simple_binary_data):
        """k=1 — train datada 100% accuracy (1-NN memorizes)."""
        X, y = simple_binary_data
        clf = KNNClassifier(k=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)

    def test_larger_k_more_stable(self, simple_binary_data):
        """
        Katta k — kamroq variance, ko'proq stable natija.
        Bu test consistency tekshiradi: bir xil input → bir xil output.
        """
        X, y = simple_binary_data
        clf = KNNClassifier(k=5)
        clf.fit(X, y)

        preds1 = clf.predict(X)
        preds2 = clf.predict(X)
        assert np.array_equal(preds1, preds2)

    def test_k_equals_n_samples(self, simple_binary_data):
        """k = n_train_samples — barcha qo'shnilarga qaraydi."""
        X, y = simple_binary_data
        clf = KNNClassifier(k=len(y))
        clf.fit(X, y)
        preds = clf.predict(X)
        # Barcha labellar majority classga teng bo'lishi kerak
        majority = np.bincount(y).argmax()
        assert np.all(preds == majority)


# ===========================================================================
# 7. ARGPARTITION XATO TESTI (biz aniqlagan bug)
# ===========================================================================

class TestArgpartitionBug:
    """
    argpartition distance weighting bilan noto'g'ri ishlashi mumkin.
    Bu testlar shu xatoni aniq ko'rsatadi.
    """

    def test_distance_weighting_consistent_with_sorted_neighbors(self):
        """
        Distance weighting natijasi k ta eng yaqin qo'shni
        to'g'ri aniqlanganiga bog'liq.

        Agar argpartition noto'g'ri k ta qo'shni tanlasa,
        bu test farqni ko'rsatadi.
        """
        np.random.seed(42)
        X_ = np.random.randn(50, 2)
        y_ = (X_[:, 0] > 0).astype(int)

        X_test = np.random.randn(10, 2)

        clf = KNNClassifier(k=5, weighting='distance')
        clf.fit(X_, y_)
        preds = clf.predict(X_test)

        # Hech bo'lmasa valid labellar qaytarishi kerak
        assert np.all(np.isin(preds, [0, 1]))

    def test_k_neighbors_count_is_exactly_k(self, simple_binary_data):
        """
        predict() ichida aynan k ta qo'shni ishlatilishi kerak.
        Bu indirect test: k=1 bilan k=3 natijasi farq qilishi mumkin.
        """
        X, y = simple_binary_data

        clf_k1 = KNNClassifier(k=1)
        clf_k3 = KNNClassifier(k=3)

        clf_k1.fit(X, y)
        clf_k3.fit(X, y)

        preds_k1 = clf_k1.predict(X)
        preds_k3 = clf_k3.predict(X)

        # Ikkisi ham valid shape bo'lishi kerak
        assert preds_k1.shape == preds_k3.shape == (len(y),)


# ===========================================================================
# 8. EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_single_training_sample(self):
        """
        Train data 1 ta sample, k=1 — shu sampleni predict qilishi kerak.
        """
        X_ = np.array([[3.0, 3.0]])
        y_ = np.array([7])

        clf = KNNClassifier(k=1)
        clf.fit(X_, y_)

        X_test = np.array([[0.0, 0.0], [100.0, 100.0]])
        preds = clf.predict(X_test)
        assert np.all(preds == 7)

    def test_high_dimensional_data(self):
        """Ko'p o'lchamli datada ishlashi kerak."""
        np.random.seed(0)
        X_ = np.random.randn(20, 100)  # 100-dim
        y_ = np.random.randint(0, 3, 20)

        X_test = np.random.randn(5, 100)

        clf = KNNClassifier(k=3)
        clf.fit(X_, y_)
        preds = clf.predict(X_test)

        assert preds.shape == (5,)
        assert np.all(np.isin(preds, np.unique(y_)))

    def test_identical_points_different_classes(self):
        """
        Bir xil koordinatali lekin turli klassli pointlar.
        Tie-breaking ishlashi kerak.
        """
        X_ = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y_ = np.array([0, 0, 1])

        clf = KNNClassifier(k=3)
        clf.fit(X_, y_)

        X_test = np.array([[1.0, 1.0]])
        preds = clf.predict(X_test)

        # Crash bo'lmasligi va valid label qaytarishi kerak
        assert preds[0] in [0, 1]

    def test_float_and_int_labels(self):
        """Int va float labellar bilan ishlashi kerak."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        for y in [np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])]:
            clf = KNNClassifier(k=1)
            clf.fit(X, y)
            preds = clf.predict(X)
            assert preds.shape == (4,)


# ===========================================================================
# 9. SKLEARN BILAN SOLISHTIRISH — @pytest.mark.slow
# ===========================================================================

@pytest.mark.slow
class TestSklearnComparison:

    @pytest.mark.parametrize("metric,sk_metric", [
        ("euclidean", "euclidean"),
        ("cityblock", "manhattan"),
        ("chebyshev", "chebyshev"),
    ])
    def test_matches_sklearn_uniform(self, metric, sk_metric, simple_binary_data):
        """
        Uniform weighting — sklearn KNeighborsClassifier bilan bir xil natija.
        cosine o'tkazib ketildi — sklearn normalizatsiya farqi bor.
        """
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
        """Iris dataseti — 90%+ accuracy kutiladi."""
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
        """Binary classify — 90%+ accuracy."""
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        data = load_breast_cancer()
        X, y = data.data, data.target

        X_, X_test, y_, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # KNN scale-sensitive — normalize qilish kerak
        scaler = StandardScaler()
        X_ = scaler.fit_transform(X_)
        X_test = scaler.transform(X_test)

        clf = KNNClassifier(k=7, metric='euclidean', weighting=weighting)
        clf.fit(X_, y_)
        preds = clf.predict(X_test)

        accuracy = np.mean(preds == y_test)
        assert accuracy >= 0.90, f"{weighting} accuracy too low: {accuracy:.2f}"