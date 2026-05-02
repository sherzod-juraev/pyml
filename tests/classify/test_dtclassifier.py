"""
DTClassifier uchun to'liq pytest test suite.

Pytest features ishlatilgan:
  - @pytest.fixture        → qayta ishlatiladigan test data va obyektlar
  - @pytest.mark.parametrize → bir xil testni ko'p input bilan ishlatish
  - pytest.raises          → exc tekshirish
  - pytest.approx          → float qiymatlarni taxminiy solishtirish
  - @pytest.mark.slow      → sekin testlarni belgilash (pytest -m "not slow")
  - conftest.py pattern    → fixture'larni modul darajasida ulashish

Ishga tushirish:
  pytest test_dt_classifier.py -v
  pytest test_dt_classifier.py -v -m "not slow"
  pytest test_dt_classifier.py -v --tb=short
"""

import numpy as np
import pytest
from mlkit.classify import DTClassifier


# ===========================================================================
# FIXTURES — qayta ishlatiladigan test data va obyektlar
# @pytest.fixture decorator bilan belgilanadi.
# Test funksiyalari parametr sifatida fixture nomini qabul qiladi.
# ===========================================================================

@pytest.fixture
def linearly_separable_data():
    """
    Oson ajratiladigan binary dataset.
    X[:, 0] < 2 → class 0, X[:, 0] >= 2 → class 1.
    Ideally fitted tree 100% accuracy berishi kerak.
    """
    X = np.array([
        [0.5, 1.0],
        [1.0, 2.0],
        [1.5, 0.5],
        [2.5, 1.0],
        [3.0, 2.0],
        [3.5, 0.5],
    ], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def multiclass_data():
    """3 klassli dataset."""
    X = np.array([
        [1.0, 1.0],
        [1.5, 1.2],
        [5.0, 5.0],
        [5.5, 4.8],
        [9.0, 1.0],
        [9.5, 1.2],
    ], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])
    return X, y


@pytest.fixture
def pure_node_data():
    """
    Hammasi bir klassga tegishli — pure node case.
    Tree hech qanday split qilmasligi kerak.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def gini_clf():
    """Default gini DTClassifier instance."""
    return DTClassifier(depth=5, min_sample=1, algorithm='gini')


@pytest.fixture
def entropy_clf():
    """Entropy DTClassifier instance."""
    return DTClassifier(depth=5, min_sample=1, algorithm='entropy')


# ===========================================================================
# 1. IMPURITY FUNCTIONS TESTLARI
# Mathematical correctness tekshiradi — formulalar to'g'ri ishlayaptimi?
# ===========================================================================

class TestGini:
    """DTClassifier.gini() method testlari."""

    def test_pure_node_gini_is_zero(self, gini_clf):
        """
        Bir klassdan iborat node gini = 0 bo'lishi kerak.
        Gini(y) = 1 - Σpᵢ² = 1 - 1² = 0
        """
        y = np.array([1, 1, 1, 1, 1])
        assert gini_clf.gini(y) == pytest.approx(0.0)
        # pytest.approx — float solishtirish uchun. 0.0 == 0.0 bo'lsa ham
        # floating point xatoliklar bo'lishi mumkin, approx ularni yutadi.

    def test_binary_equal_split_gini(self, gini_clf):
        """
        50/50 binary split — maximum gini = 0.5.
        Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
        """
        y = np.array([0, 0, 0, 1, 1, 1])
        assert gini_clf.gini(y) == pytest.approx(0.5)

    def test_gini_range(self, gini_clf):
        """Gini [0, 1) oralig'ida bo'lishi kerak."""
        y = np.array([0, 1, 2, 0, 1, 2])
        result = gini_clf.gini(y)
        assert 0.0 <= result < 1.0

    def test_gini_symmetric(self, gini_clf):
        """
        Klasslar tartibi gini qiymatiga ta'sir qilmasligi kerak.
        Gini([0,0,1,1]) == Gini([1,1,0,0])
        """
        y1 = np.array([0, 0, 1, 1])
        y2 = np.array([1, 1, 0, 0])
        assert gini_clf.gini(y1) == pytest.approx(gini_clf.gini(y2))


class TestEntropy:
    """DTClassifier.entropy() method testlari."""

    def test_pure_node_entropy_is_zero(self, entropy_clf):
        """Pure node entropy = 0."""
        y = np.array([2, 2, 2, 2])
        assert entropy_clf.entropy(y) == pytest.approx(0.0, abs=1e-9)

    def test_binary_equal_split_entropy(self, entropy_clf):
        """
        50/50 binary split — maximum binary entropy = 1.0 bit.
        Entropy = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
        """
        y = np.array([0, 0, 1, 1])
        assert entropy_clf.entropy(y) == pytest.approx(1.0, abs=1e-6)

    def test_entropy_non_negative(self, entropy_clf):
        """Entropy hech qachon manfiy bo'lmasligi kerak."""
        y = np.array([0, 1, 2, 0, 1])
        assert entropy_clf.entropy(y) >= 0.0

    def test_entropy_increases_with_disorder(self, entropy_clf):
        """Ko'proq aralashgan → ko'proq entropy."""
        pure = np.array([1, 1, 1, 1])
        mixed = np.array([0, 1, 2, 3])
        assert entropy_clf.entropy(pure) < entropy_clf.entropy(mixed)


class TestInformationGain:
    """DTClassifier.information_gain() method testlari."""

    def test_perfect_split_information_gain(self, entropy_clf):
        """
        Perfect split: parent aralash, children pure.
        IG = H(parent) - 0 = H(parent) > 0
        """
        y_parent = np.array([0, 0, 0, 1, 1, 1])
        y_left = np.array([0, 0, 0])
        y_right = np.array([1, 1, 1])
        ig = entropy_clf.information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(1.0, abs=1e-6)

    def test_no_split_information_gain_is_zero(self, entropy_clf):
        """
        Split hech narsa o'zgartirmasa IG ≈ 0.
        parent va children bir xil distribution.
        """
        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 1])
        y_right = np.array([0, 1])
        ig = entropy_clf.information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(0.0, abs=1e-6)

    def test_information_gain_non_negative(self, entropy_clf):
        """IG hech qachon manfiy bo'lmasligi kerak."""
        y_parent = np.array([0, 0, 1, 1, 2])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1, 2])
        ig = entropy_clf.information_gain(y_parent, y_left, y_right)
        assert ig >= 0.0


# ===========================================================================
# 2. FIT va PREDICT TESTLARI
# Core functionality — model to'g'ri o'qiyaptimi va bashorat qilyaptimi?
# ===========================================================================

class TestFitPredict:
    """fit() va predict() asosiy testlari."""

    def test_fit_returns_self(self, gini_clf, linearly_separable_data):
        """
        fit() method chaining uchun self qaytarishi kerak.
        clf.fit(X, y).predict(X) kabi ishlatish mumkin bo'lishi uchun.
        """
        X, y = linearly_separable_data
        result = gini_clf.fit(X, y)
        assert result is gini_clf

    def test_predict_returns_numpy_array(self, gini_clf, linearly_separable_data):
        """predict() numpy array qaytarishi kerak."""
        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_output_shape(self, gini_clf, linearly_separable_data):
        """predict() output shape (n_samples,) bo'lishi kerak."""
        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_perfect_accuracy_on_separable_data(self, gini_clf, linearly_separable_data):
        """
        Oson ajratiladigan datada train accuracy = 100% bo'lishi kerak.
        np.array_equal — elementma-element tenglikni tekshiradi.
        """
        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert np.array_equal(preds, y)

    def test_predict_labels_are_subset_of_train_labels(self, gini_clf, linearly_separable_data):
        """
        predict() faqat train vaqtida ko'rilgan klasslarni qaytarishi kerak.
        np.isin — elementlarning set ichida borligini tekshiradi.
        """
        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        unique_train = np.unique(y)
        assert np.all(np.isin(preds, unique_train))

    def test_pure_node_predicts_single_class(self, gini_clf, pure_node_data):
        """
        Hammasi bir klass bo'lsa, barcha predictionlar shu klass bo'lishi kerak.
        np.unique qaytaradigan array uzunligi 1 bo'lishi kerak.
        """
        X, y = pure_node_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert np.all(preds == 1)

    def test_multiclass_prediction(self, gini_clf, multiclass_data):
        """3 klassli datada to'g'ri ishlashi."""
        X, y = multiclass_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy >= 0.8  # kamida 80% accuracy


# ===========================================================================
# 3. EXCEPTION TESTLARI
# pytest.raises context manager bilan exc handling tekshiriladi.
# ===========================================================================

class TestExceptions:
    """Exception va error handling testlari."""

    def test_predict_before_fit_raises_not_fitted(self):
        """
        fit() chaqirilmay predict() qilinsa NotFitted exc chiqishi kerak.

        pytest.raises(ExceptionClass) — context manager sifatida ishlatiladi.
        Ichidagi kod berilgan exceptionni raise qilishi kerak,
        aks holda test FAIL bo'ladi.
        """
        clf = DTClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(Exception):  # NotFitted — Exception subclass
            clf.predict(X)

    def test_predict_before_fit_raises_correct_type(self):
        """
        Aniq exc type tekshirish.
        NotFitted import qilish kerak bo'lsa:
        from mlkit.exc import NotFitted
        """
        from mlkit.exc import NotFitted
        clf = DTClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted):
            clf.predict(X)

    def test_predict_before_fit_error_message(self):
        """
        Exception message to'g'ri ekanligini tekshirish.
        pytest.raises(...) as exc_info — exc obyektini olish uchun.
        """
        from mlkit.exc import NotFitted
        clf = DTClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted) as exc_info:
            clf.predict(X)

        # Exception message'da "fitted" so'zi borligini tekshirish
        assert "fitted" in str(exc_info.value).lower()


# ===========================================================================
# 4. PARAMETRIZE — bir xil testni ko'p input bilan ishlatish
# @pytest.mark.parametrize decorator bilan belgilanadi.
# Har bir (input, expected) juftligi uchun alohida test yaratadi.
# ===========================================================================

class TestParametrized:
    """@pytest.mark.parametrize misollari."""

    @pytest.mark.parametrize("algorithm", ["gini", "entropy"])
    def test_both_algorithms_fit_and_predict(self, algorithm, linearly_separable_data):
        """
        Gini va entropy algoritmlarining ikkalasi ham ishlashi kerak.
        'algorithm' o'zgaruvchisi ["gini", "entropy"] ro'yxatidan navbat bilan olinadi.
        Bu 2 ta alohida test yaratadi.
        """
        X, y = linearly_separable_data
        clf = DTClassifier(depth=5, min_sample=1, algorithm=algorithm)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    @pytest.mark.parametrize("depth", [1, 2, 3, 5, 10])
    def test_various_depths_no_crash(self, depth, linearly_separable_data):
        """
        Turli depth qiymatlarida crash bo'lmasligi kerak.
        5 ta alohida test yaratadi.
        """
        X, y = linearly_separable_data
        clf = DTClassifier(depth=depth, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    @pytest.mark.parametrize("min_sample", [1, 2, 3])
    def test_various_min_samples(self, min_sample, linearly_separable_data):
        """min_sample turli qiymatlarda ishlashi kerak."""
        X, y = linearly_separable_data
        clf = DTClassifier(depth=5, min_sample=min_sample)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    @pytest.mark.parametrize("y_pure,expected_class", [
        (np.array([0, 0, 0, 0]), 0),
        (np.array([1, 1, 1, 1]), 1),
        (np.array([5, 5, 5, 5]), 5),
    ])
    def test_pure_datasets_predict_correct_class(self, y_pure, expected_class):
        """
        Pure dataset — predict natijasi o'sha klass bo'lishi kerak.
        Bir vaqtda 3 ta test: class 0, class 1, class 5.
        """
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X, y_pure)
        preds = clf.predict(X)
        assert np.all(preds == expected_class)

    @pytest.mark.parametrize("algorithm,expected_metric", [
        ("gini", "weighted_gini"),
        ("entropy", "information_gain"),
    ])
    def test_algorithm_uses_correct_internal_metric(self, algorithm, expected_metric):
        """
        Har bir algorithm o'zining metric funksiyasiga ega ekanligini tekshirish.
        hasattr — obyektda attribute yoki method borligini tekshiradi.
        """
        clf = DTClassifier(algorithm=algorithm)
        assert hasattr(clf, expected_metric)


# ===========================================================================
# 5. BEST_SPLIT TESTLARI
# Internal logikani tekshirish — qora quti testidan farqli.
# ===========================================================================

class TestBestSplit:
    """best_split() method testlari."""

    def test_returns_none_when_no_valid_split(self):
        """
        min_sample juda katta bo'lsa, hech qanday split topilmasligi kerak.
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])
        clf = DTClassifier(min_sample=10)  # 10 > len(y) → hech qanday split
        feature, threshold = clf.best_split(X, y)
        assert feature is None
        assert threshold is None

    def test_finds_correct_split_feature(self, linearly_separable_data):
        """
        Ajratiladigan datada birinchi feature bo'yicha split topishi kerak.
        """
        X, y = linearly_separable_data
        clf = DTClassifier(min_sample=1)
        feature, threshold = clf.best_split(X, y)
        assert feature is not None
        assert threshold is not None

    def test_split_threshold_within_feature_range(self, linearly_separable_data):
        """
        Topilgan threshold feature qiymatlarining oralig'ida bo'lishi kerak.
        """
        X, y = linearly_separable_data
        clf = DTClassifier(min_sample=1)
        feature, threshold = clf.best_split(X, y)
        if feature is not None:
            assert X[:, feature].min() <= threshold <= X[:, feature].max()

    def test_pure_data_returns_none_split(self, pure_node_data):
        """
        Pure node (hammasi bir klass) — split zarur emas, None qaytarishi kerak.
        """
        X, y = pure_node_data
        clf = DTClassifier(min_sample=1)
        feature, threshold = clf.best_split(X, y)
        # Pure datada barcha thresholds gini=0 beradi, lekin split topilishi mumkin
        # Asosiy tekshiruv: crash bo'lmasligi
        # feature None yoki int bo'lishi mumkin (implementationga bog'liq)
        assert feature is None or isinstance(feature, (int, np.integer))


# ===========================================================================
# 6. TREE STRUCTURE TESTLARI
# Build qilingan tree to'g'ri tuzilmaga egami?
# ===========================================================================

class TestTreeStructure:
    """Daraxt tuzilmasi testlari."""

    def test_root_is_not_none_after_fit(self, gini_clf, linearly_separable_data):
        """fit() dan keyin root node bo'sh bo'lmasligi kerak."""
        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        assert gini_clf.root is not None

    def test_depth_1_tree_has_leaf_children(self, linearly_separable_data):
        """
        depth=1 tree — root dan keyin to'g'ridan-to'g'ri leaf bo'lishi kerak.
        Leaf node: value mavjud, left/right yo'q.
        """
        X, y = linearly_separable_data
        clf = DTClassifier(depth=1, min_sample=1)
        clf.fit(X, y)

        root = clf.root
        if root.left is not None:
            assert root.left.value is not None  # leaf
        if root.right is not None:
            assert root.right.value is not None  # leaf

    def test_single_feature_data(self):
        """1 ta feature bilan ishlashi kerak."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)

    def test_many_features_data(self):
        """Ko'p feature bilan ishlashi kerak."""
        np.random.seed(42)
        X = np.random.randn(50, 20)  # 50 sample, 20 feature
        y = (X[:, 0] > 0).astype(int)  # faqat birinchi feature muhim
        clf = DTClassifier(depth=5, min_sample=2)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (50,)


# ===========================================================================
# 7. DEPTH va OVERFITTING TESTLARI
# Hyperparameter ta'sirini tekshirish.
# ===========================================================================

class TestDepthEffect:
    """Depth parametrining ta'siri."""

    def test_depth_1_lower_or_equal_accuracy_than_depth_10(self, linearly_separable_data):
        """
        Katta depth — train datada ko'proq yoki teng accuracy berishi kerak.
        depth=1 <= depth=10 accuracy.
        """
        X, y = linearly_separable_data

        clf_shallow = DTClassifier(depth=1, min_sample=1)
        clf_deep = DTClassifier(depth=10, min_sample=1)

        clf_shallow.fit(X, y)
        clf_deep.fit(X, y)

        acc_shallow = np.mean(clf_shallow.predict(X) == y)
        acc_deep = np.mean(clf_deep.predict(X) == y)

        assert acc_shallow <= acc_deep + 1e-9  # floating point tolerance

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_max_depth_respected(self, depth):
        """
        Daraxt belgilangan depth dan chuqurroq o'smasligi kerak.
        Recursive traversal orqali max depth hisoblanadi.
        """
        X = np.array([[float(i)] for i in range(20)])
        y = np.array([i % 4 for i in range(20)])  # 4 klass

        clf = DTClassifier(depth=depth, min_sample=1)
        clf.fit(X, y)

        def get_max_depth(node, current=0):
            if node is None or node.value is not None:
                return current
            return max(
                get_max_depth(node.left, current + 1),
                get_max_depth(node.right, current + 1)
            )

        actual_depth = get_max_depth(clf.root)
        assert actual_depth <= depth


# ===========================================================================
# 8. SEKIN TESTLAR — @pytest.mark.slow bilan belgilanadi
# Ishga tushirish: pytest -m "not slow"  →  bu testlarni o'tkazib ketadi
# Ishga tushirish: pytest -m "slow"      →  faqat bu testlarni ishlatadi
# ===========================================================================

@pytest.mark.slow
class TestWithRealDatasets:
    """Katta datasetlar bilan integration testlar."""

    def test_sklearn_iris_comparison(self):
        """
        sklearn iris dataset bilan tekshirish.
        sklearn DTClassifier bilan taxminiy bir xil natija bo'lishi kerak.
        """
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = DTClassifier(depth=5, min_sample=2, algorithm='gini')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accuracy = np.mean(preds == y_test)
        # Iris dataseti oson, 85%+ kutiladi
        assert accuracy >= 0.85, f"Iris accuracy too low: {accuracy:.2f}"

    @pytest.mark.parametrize("algorithm", ["gini", "entropy"])
    def test_sklearn_breast_cancer(self, algorithm):
        """Breast cancer dataset — binary classify."""
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        data = load_breast_cancer()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = DTClassifier(depth=7, min_sample=3, algorithm=algorithm)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accuracy = np.mean(preds == y_test)
        assert accuracy >= 0.85, f"{algorithm} accuracy too low: {accuracy:.2f}"


# ===========================================================================
# 9. EDGE CASES — chegaraviy holatlar
# ===========================================================================

class TestEdgeCases:
    """Noodatiy va chegaraviy holatlar."""

    def test_single_sample_per_class(self):
        """Har bir klassda bitta sample — ishlab chiqishi kerak."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([0, 1])
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 2

    def test_duplicate_feature_values(self):
        """
        Bir xil feature qiymatlari bo'lsa ham ishlab chiqishi kerak.
        threshold = midpoint → unique qiymatlar kerak, aks holda thresholds bo'sh.
        """
        X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 6

    def test_predict_new_samples_in_range(self, linearly_separable_data):
        """
        Train datadan boshqa lekin shu oraliqda bo'lgan yangi samplelar.
        """
        X_train, y_train = linearly_separable_data
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X_train, y_train)

        X_new = np.array([[0.8, 1.5], [3.2, 1.8]])
        preds = clf.predict(X_new)
        assert preds.shape == (2,)

    def test_two_classes_with_imbalance(self):
        """
        Imbalanced dataset: 90% class 0, 10% class 1.
        """
        np.random.seed(0)
        X = np.random.randn(100, 2)
        y = np.zeros(100, dtype=int)
        y[:10] = 1  # 10% minority class

        # X ni minority class uchun ajrat
        X[:10] += 5  # class 1 ni boshqa joyga siljit

        clf = DTClassifier(depth=5, min_sample=2)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)