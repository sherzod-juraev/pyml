import numpy as np
import pytest

from pyml import DTClassifier
from pyml.exc import NotFittedError


class TestDTInit:
    def test_init_defaults(self):
        model = DTClassifier()

        assert model.max_depth == 5
        assert model.min_sample == 5
        assert model.algorithm == "gini"
        assert model.root is None

    def test_init_custom(self):
        model = DTClassifier(depth=3, min_sample=2, algorithm="entropy")

        assert model.max_depth == 3
        assert model.min_sample == 2
        assert model.algorithm == "entropy"


class TestDTImpurity:

    def test_gini_pure(self):
        y = np.array([1, 1, 1])
        model = DTClassifier()

        assert model.gini(y) == 0.0

    def test_gini_mixed(self):
        y = np.array([0, 1])
        model = DTClassifier()

        assert np.isclose(model.gini(y), 0.5)

    def test_entropy_pure(self):
        y = np.array([1, 1, 1])
        model = DTClassifier()

        assert np.isclose(model.entropy(y), 0.0)

    def test_entropy_mixed(self):
        y = np.array([0, 1])
        model = DTClassifier()

        assert np.isclose(model.entropy(y), 1.0, atol=1e-6)


class TestDTWeighted:

    def test_weighted_gini(self):
        model = DTClassifier()

        y_left = np.array([0, 0])
        y_right = np.array([1, 1])

        score = model.weighted_gini(y_left, y_right)

        assert np.isclose(score, 0.0)

    def test_weighted_entropy(self):
        model = DTClassifier()

        y_left = np.array([0, 0])
        y_right = np.array([1, 1])

        score = model.weighted_entropy(y_left, y_right)

        assert np.isclose(score, 0.0)


class TestDTInfoGain:

    def test_information_gain_perfect_split(self):
        model = DTClassifier()

        y_parent = np.array([0, 1, 0, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])

        ig = model.information_gain(y_parent, y_left, y_right)

        assert ig > 0


class TestDTBestSplit:

    def test_best_split_simple(self):
        X = np.array([
            [1.],
            [2.],
            [3.],
            [10.],
            [11.],
            [12.]
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = DTClassifier(depth=2, min_sample=1, algorithm="gini")

        feature, threshold = model.best_split(X, y)

        assert feature == 0
        assert threshold is not None


class TestDTFit:

    def test_fit_sets_root(self):
        X = np.array([
            [1., 1.],
            [2., 2.],
            [10., 10.],
            [11., 11.]
        ])
        y = np.array([0, 0, 1, 1])

        model = DTClassifier(depth=3, min_sample=1)
        model.fit(X, y)

        assert model.root is not None


class TestDTPredict:

    def test_predict_simple(self):
        X = np.array([
            [1., 1.],
            [2., 2.],
            [10., 10.],
            [11., 11.]
        ])
        y = np.array([0, 0, 1, 1])

        model = DTClassifier(depth=3, min_sample=1)
        model.fit(X, y)

        preds = model.predict(X)

        assert preds.shape == (4,)
        assert set(preds) <= {0, 1}

    def test_single_prediction(self):
        X = np.array([
            [1., 1.],
            [10., 10.]
        ])
        y = np.array([0, 1])

        model = DTClassifier(depth=1, min_sample=1)
        model.fit(X, y)

        assert isinstance(model.determine_label(X[0]), int)


class TestDTNotFitted:

    def test_predict_before_fit(self):
        model = DTClassifier()

        X = np.array([[1., 2.]])

        with pytest.raises(NotFittedError):
            model.predict(X)


class TestDTOutput:

    def test_output_dtype(self):
        X = np.array([
            [1., 2.],
            [3., 4.]
        ])
        y = np.array([0, 1])

        model = DTClassifier(depth=2, min_sample=1)
        model.fit(X, y)

        preds = model.predict(X)

        assert np.issubdtype(preds.dtype, np.integer)
