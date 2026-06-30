import numpy as np
import pytest

from pyml.tree import DTRegressor
from pyml.exc import NotFittedError


class TestDTRegressorInit:

    def test_init_defaults(self):
        model = DTRegressor()

        assert model.max_depth == 5
        assert model.min_split == 5
        assert model.min_leaf == 5
        assert model.min_impurity_decrease == 1e-2
        assert model.root is None

    def test_init_custom(self):
        model = DTRegressor(
            max_depth=3,
            min_split=2,
            min_leaf=1,
            min_impurity_decrease=0.0
        )

        assert model.max_depth == 3
        assert model.min_split == 2
        assert model.min_leaf == 1


class TestDTRegressorMetrics:

    def test_variance_zero_case(self):
        y = np.array([5., 5., 5.])

        model = DTRegressor()

        assert np.isclose(np.var(y), 0.0)

    def test_best_split_no_valid(self):
        X = np.array([[1.], [1.], [1.], [1.]])
        y = np.array([1., 2., 3., 4.])

        model = DTRegressor(min_leaf=10)

        delta, feat, thr = model.best_split(X, y)

        assert feat is None
        assert thr is None
        assert delta == -np.inf


class TestDTRegressorBestSplit:

    def test_best_split_linear(self):
        X = np.array([[1.], [2.], [3.], [10.], [11.], [12.]])
        y = np.array([1., 2., 3., 10., 11., 12.])

        model = DTRegressor(min_leaf=1, min_split=2)

        delta, feat, thr = model.best_split(X, y)

        assert feat == 0
        assert thr is not None
        assert delta > 0


class TestDTRegressorFit:

    def test_fit_sets_root(self):
        X = np.array([[1.], [2.], [3.], [10.], [11.], [12.]])
        y = np.array([1., 2., 3., 10., 11., 12.])

        model = DTRegressor(max_depth=3, min_leaf=1, min_split=2)

        model.fit(X, y)

        assert model.root is not None


class TestDTRegressorPredict:

    def test_predict_simple_regression(self):
        X = np.array([[1.], [2.], [3.], [10.], [11.], [12.]])
        y = np.array([1., 2., 3., 10., 11., 12.])

        model = DTRegressor(max_depth=3, min_leaf=1, min_split=2)
        model.fit(X, y)

        preds = model.predict(X)

        assert preds.shape == (6,)
        assert np.allclose(preds, y, atol=1e-1)  # approximate


class TestDTRegressorConstant:

    def test_constant_output(self):
        X = np.array([[1.], [2.], [3.], [4.]])
        y = np.array([5., 5., 5., 5.])

        model = DTRegressor()
        model.fit(X, y)

        preds = model.predict(X)

        assert np.allclose(preds, 5.0)


class TestDTRegressorErrors:

    def test_predict_before_fit(self):
        model = DTRegressor()

        X = np.array([[1.], [2.]])

        with pytest.raises(NotFittedError):
            model.predict(X)


class TestDTRegressorEdgeCases:

    def test_min_split_blocks_tree(self):
        X = np.array([[1.], [2.], [3.]])
        y = np.array([1., 2., 3.])

        model = DTRegressor(min_split=100)

        model.fit(X, y)

        assert model.root.value is not None

    def test_single_sample(self):
        X = np.array([[1.]])
        y = np.array([10.])

        model = DTRegressor()

        model.fit(X, y)

        assert model.predict(X)[0] == 10.0


class TestDTRegressorOutput:

    def test_dtype(self):
        X = np.array([[1.], [2.], [3.]])
        y = np.array([1., 2., 3.])

        model = DTRegressor(max_depth=2, min_leaf=1, min_split=2)
        model.fit(X, y)

        preds = model.predict(X)

        assert preds.dtype == np.float64
