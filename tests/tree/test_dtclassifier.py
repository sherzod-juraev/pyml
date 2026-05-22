import numpy as np
import pytest
from sklearn.datasets import make_classification

from pyml.tree.decision_tree import DTClassifier


class TestDTClassifier:
    """Decision Tree Classifier."""

    @pytest.fixture
    def data(self):
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        X, y = make_classification(n_samples=150, n_features=5, n_classes=3,
                                    n_clusters_per_class=1, random_state=42)
        return X, y

    def test_fit_creates_root(self, data):
        X, y = data
        model = DTClassifier(depth=3).fit(X, y)
        assert model.root is not None

    def test_predict_shape(self, data):
        X, y = data
        model = DTClassifier(depth=3).fit(X, y)
        X_test = np.random.randn(20, 5).astype(np.float64)
        pred = model.predict(X_test)
        assert pred.shape == (20,)

    def test_single_sample(self, data):
        X, y = data
        model = DTClassifier(depth=3).fit(X, y)
        pred = model.predict(X[:1])
        assert pred.shape == (1,)

    def test_binary_output(self, data):
        X, y = data
        model = DTClassifier(depth=3).fit(X, y)
        pred = model.predict(X)
        assert set(np.unique(pred)).issubset(set(np.unique(y)))

    def test_multiclass_output(self, multiclass_data):
        X, y = multiclass_data
        model = DTClassifier(depth=5).fit(X, y)
        pred = model.predict(X)
        assert set(np.unique(pred)).issubset({0, 1, 2})

    def test_accuracy_above_chance(self, data):
        X, y = data
        model = DTClassifier(depth=5).fit(X, y)
        pred = model.predict(X)
        acc = np.mean(pred == y)
        assert acc > 0.6

    def test_gini_vs_entropy(self, data):
        X, y = data
        gini = DTClassifier(depth=3, algorithm='gini').fit(X, y)
        entropy = DTClassifier(depth=3, algorithm='entropy').fit(X, y)
        assert gini.root is not None
        assert entropy.root is not None

    def test_different_depths(self, data):
        X, y = data
        for depth in [1, 3, 5, 10]:
            model = DTClassifier(depth=depth).fit(X, y)
            pred = model.predict(X)
            assert pred.shape == (100,)

    def test_min_sample_parameter(self, data):
        X, y = data
        model = DTClassifier(depth=5, min_sample=10).fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (100,)

    def test_not_fitted_raises(self):
        model = DTClassifier(depth=3)
        X = np.random.randn(10, 3).astype(np.float64)
        with pytest.raises(Exception):
            model.predict(X)

    def test_constant_target(self):
        X = np.random.randn(50, 3).astype(np.float64)
        y = np.ones(50, dtype=np.int64)
        model = DTClassifier(depth=3).fit(X, y)
        pred = model.predict(X)
        assert np.all(pred == 1)

    def test_fit_multiple_times(self, data):
        X, y = data
        model = DTClassifier(depth=3)
        model.fit(X, y)
        pred1 = model.predict(X)
        model.fit(X, y)
        pred2 = model.predict(X)
        assert np.array_equal(pred1, pred2)

    def test_reproducibility(self, data):
        X, y = data
        model1 = DTClassifier(depth=3).fit(X, y)
        model2 = DTClassifier(depth=3).fit(X, y)
        assert np.array_equal(model1.predict(X), model2.predict(X))

    def test_single_feature(self):
        X = np.random.randn(50, 1).astype(np.float64)
        y = np.random.randint(0, 2, 50)
        model = DTClassifier(depth=3).fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (50,)
