import numpy as np
import pytest
from mlkit import DTClassifier


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def linearly_separable_data():
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
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def gini_clf():
    return DTClassifier(depth=5, min_sample=1, algorithm='gini')


@pytest.fixture
def entropy_clf():
    return DTClassifier(depth=5, min_sample=1, algorithm='entropy')


# ===========================================================================
# 1. IMPURITY FUNCTIONS
# ===========================================================================

class TestGini:

    def test_pure_node_gini_is_zero(self, gini_clf):
        """
        Gini(y) = 1 - Σpᵢ² = 1 - 1² = 0
        """
        y = np.array([1, 1, 1, 1, 1])
        assert gini_clf.gini(y) == pytest.approx(0.0)

    def test_binary_equal_split_gini(self, gini_clf):
        """
        50/50 binary split — maximum gini = 0.5.
        Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
        """
        y = np.array([0, 0, 0, 1, 1, 1])
        assert gini_clf.gini(y) == pytest.approx(0.5)

    def test_gini_range(self, gini_clf):
        """Gini [0, 1)"""
        y = np.array([0, 1, 2, 0, 1, 2])
        result = gini_clf.gini(y)
        assert 0.0 <= result < 1.0

    def test_gini_symmetric(self, gini_clf):
        """
        Gini([0,0,1,1]) == Gini([1,1,0,0])
        """
        y1 = np.array([0, 0, 1, 1])
        y2 = np.array([1, 1, 0, 0])
        assert gini_clf.gini(y1) == pytest.approx(gini_clf.gini(y2))


class TestEntropy:

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
        y = np.array([0, 1, 2, 0, 1])
        assert entropy_clf.entropy(y) >= 0.0

    def test_entropy_increases_with_disorder(self, entropy_clf):
        pure = np.array([1, 1, 1, 1])
        mixed = np.array([0, 1, 2, 3])
        assert entropy_clf.entropy(pure) < entropy_clf.entropy(mixed)


class TestInformationGain:

    def test_perfect_split_information_gain(self, entropy_clf):
        """
        IG = H(parent) - 0 = H(parent) > 0
        """
        y_parent = np.array([0, 0, 0, 1, 1, 1])
        y_left = np.array([0, 0, 0])
        y_right = np.array([1, 1, 1])
        ig = entropy_clf.information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(1.0, abs=1e-6)

    def test_no_split_information_gain_is_zero(self, entropy_clf):

        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 1])
        y_right = np.array([0, 1])
        ig = entropy_clf.information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(0.0, abs=1e-6)

    def test_information_gain_non_negative(self, entropy_clf):

        y_parent = np.array([0, 0, 1, 1, 2])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1, 2])
        ig = entropy_clf.information_gain(y_parent, y_left, y_right)
        assert ig >= 0.0


# ===========================================================================
# 2. FIT and PREDICT
# ===========================================================================

class TestFitPredict:

    def test_fit_returns_self(self, gini_clf, linearly_separable_data):

        X, y = linearly_separable_data
        result = gini_clf.fit(X, y)
        assert result is gini_clf

    def test_predict_returns_numpy_array(self, gini_clf, linearly_separable_data):

        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_output_shape(self, gini_clf, linearly_separable_data):

        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_perfect_accuracy_on_separable_data(self, gini_clf, linearly_separable_data):

        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert np.array_equal(preds, y)

    def test_predict_labels_are_subset_of_train_labels(self, gini_clf, linearly_separable_data):

        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        unique_train = np.unique(y)
        assert np.all(np.isin(preds, unique_train))

    def test_pure_node_predicts_single_class(self, gini_clf, pure_node_data):

        X, y = pure_node_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        assert np.all(preds == 1)

    def test_multiclass_prediction(self, gini_clf, multiclass_data):

        X, y = multiclass_data
        gini_clf.fit(X, y)
        preds = gini_clf.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy >= 0.8


# ===========================================================================
# 3. EXCEPTION
# ===========================================================================

class TestExceptions:

    def test_predict_before_fit_raises_not_fitted(self):

        clf = DTClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(Exception):
            clf.predict(X)

    def test_predict_before_fit_raises_correct_type(self):

        from mlkit.exc import NotFitted
        clf = DTClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted):
            clf.predict(X)

    def test_predict_before_fit_error_message(self):

        from mlkit.exc import NotFitted
        clf = DTClassifier()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFitted) as exc_info:
            clf.predict(X)

        assert "fitted" in str(exc_info.value).lower()


# ===========================================================================
# 4. PARAMETRIZE
# ===========================================================================

class TestParametrized:

    @pytest.mark.parametrize("algorithm", ["gini", "entropy"])
    def test_both_algorithms_fit_and_predict(self, algorithm, linearly_separable_data):

        X, y = linearly_separable_data
        clf = DTClassifier(depth=5, min_sample=1, algorithm=algorithm)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    @pytest.mark.parametrize("depth", [1, 2, 3, 5, 10])
    def test_various_depths_no_crash(self, depth, linearly_separable_data):

        X, y = linearly_separable_data
        clf = DTClassifier(depth=depth, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    @pytest.mark.parametrize("min_sample", [1, 2, 3])
    def test_various_min_samples(self, min_sample, linearly_separable_data):

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

        clf = DTClassifier(algorithm=algorithm)
        assert hasattr(clf, expected_metric)


# ===========================================================================
# 5. BEST_SPLIT
# ===========================================================================

class TestBestSplit:

    def test_returns_none_when_no_valid_split(self):

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])
        clf = DTClassifier(min_sample=10)
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

        X, y = linearly_separable_data
        clf = DTClassifier(min_sample=1)
        feature, threshold = clf.best_split(X, y)
        if feature is not None:
            assert X[:, feature].min() <= threshold <= X[:, feature].max()

    def test_pure_data_returns_none_split(self, pure_node_data):

        X, y = pure_node_data
        clf = DTClassifier(min_sample=1)
        feature, threshold = clf.best_split(X, y)
        assert feature is None or isinstance(feature, (int, np.integer))


# ===========================================================================
# 6. TREE STRUCTURE
# ===========================================================================

class TestTreeStructure:

    def test_root_is_not_none_after_fit(self, gini_clf, linearly_separable_data):

        X, y = linearly_separable_data
        gini_clf.fit(X, y)
        assert gini_clf.root is not None

    def test_depth_1_tree_has_leaf_children(self, linearly_separable_data):

        X, y = linearly_separable_data
        clf = DTClassifier(depth=1, min_sample=1)
        clf.fit(X, y)

        root = clf.root
        if root.left is not None:
            assert root.left.value is not None  # leaf
        if root.right is not None:
            assert root.right.value is not None  # leaf

    def test_single_feature_data(self):

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.array_equal(preds, y)

    def test_many_features_data(self):
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = (X[:, 0] > 0).astype(int)
        clf = DTClassifier(depth=5, min_sample=2)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (50,)


# ===========================================================================
# 7. DEPTH and OVERFITTING
# ===========================================================================

class TestDepthEffect:

    def test_depth_1_lower_or_equal_accuracy_than_depth_10(self, linearly_separable_data):

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


@pytest.mark.slow
class TestWithRealDatasets:

    def test_sklearn_iris_comparison(self):

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
# 9. EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_single_sample_per_class(self):

        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([0, 1])
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 2

    def test_duplicate_feature_values(self):

        X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        clf = DTClassifier(depth=3, min_sample=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 6

    def test_predict_new_samples_in_range(self, linearly_separable_data):

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
        y[:10] = 1

        X[:10] += 5

        clf = DTClassifier(depth=5, min_sample=2)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)