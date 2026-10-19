"""Behavior every public pyml estimator must share, checked across all of them."""

from typing import Any

import numpy as np
import pytest

from pyml.cluster import DBSCAN, KMeans
from pyml.core.exceptions import NotFittedError, ShapeMismatchError
from pyml.linear_model import (
    ElasticNet,
    ElasticNetClassifier,
    Lasso,
    LassoClassifier,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from pyml.naive_bayes import GaussianNB, MultinomialNB
from pyml.neighbors import (
    KNNClassifier,
    KNNRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from pyml.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pyml.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFIERS = [
    ElasticNetClassifier,
    LassoClassifier,
    LogisticRegression,
    RidgeClassifier,
    GaussianNB,
    MultinomialNB,
    lambda: KNNClassifier(n_neighbors=3),
    lambda: RadiusNeighborsClassifier(radius=50.0),
    lambda: DecisionTreeClassifier(max_depth=3),
]
REGRESSORS = [
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    lambda: KNNRegressor(n_neighbors=3),
    lambda: RadiusNeighborsRegressor(radius=50.0),
    lambda: DecisionTreeRegressor(max_depth=3),
]
TRANSFORMERS = [MinMaxScaler, RobustScaler, StandardScaler]
PREDICTABLE_CLUSTERERS = [lambda: KMeans(n_clusters=2, random_state=42)]
ALL_CLUSTERERS = [*PREDICTABLE_CLUSTERERS, lambda: DBSCAN(eps=3.0, min_samples=3)]


def _ids(factories: list[Any]) -> list[str]:
    return [f.__name__ if isinstance(f, type) else f().__class__.__name__ for f in factories]


@pytest.fixture
def X(rng):
    return rng.uniform(1, 10, size=(60, 3))


@pytest.fixture
def y_class():
    return np.array([0, 1] * 30)


@pytest.fixture
def y_reg(rng):
    return rng.uniform(0, 10, size=60)


class TestSupervisedContract:
    @pytest.mark.parametrize("make", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_classifier_fit_returns_self(self, make, X, y_class):
        model = make()
        assert model.fit(X, y_class) is model

    @pytest.mark.parametrize("make", REGRESSORS, ids=_ids(REGRESSORS))
    def test_regressor_fit_returns_self(self, make, X, y_reg):
        model = make()
        assert model.fit(X, y_reg) is model

    @pytest.mark.parametrize("make", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_classifier_predict_before_fit_raises(self, make, X):
        with pytest.raises(NotFittedError):
            make().predict(X)

    @pytest.mark.parametrize("make", REGRESSORS, ids=_ids(REGRESSORS))
    def test_regressor_predict_before_fit_raises(self, make, X):
        with pytest.raises(NotFittedError):
            make().predict(X)

    @pytest.mark.parametrize("make", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_classifier_rejects_wrong_feature_count(self, make, X, y_class):
        model = make().fit(X, y_class)
        with pytest.raises(ShapeMismatchError):
            model.predict(X[:, :2])

    @pytest.mark.parametrize("make", REGRESSORS, ids=_ids(REGRESSORS))
    def test_regressor_rejects_wrong_feature_count(self, make, X, y_reg):
        model = make().fit(X, y_reg)
        with pytest.raises(ShapeMismatchError):
            model.predict(X[:, :2])

    @pytest.mark.parametrize("make", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_classifier_fit_does_not_modify_inputs(self, make, X, y_class):
        X_before, y_before = X.copy(), y_class.copy()
        make().fit(X, y_class)
        assert np.array_equal(X, X_before)
        assert np.array_equal(y_class, y_before)

    @pytest.mark.parametrize("make", REGRESSORS, ids=_ids(REGRESSORS))
    def test_regressor_fit_does_not_modify_inputs(self, make, X, y_reg):
        X_before, y_before = X.copy(), y_reg.copy()
        make().fit(X, y_reg)
        assert np.array_equal(X, X_before)
        assert np.array_equal(y_reg, y_before)

    @pytest.mark.parametrize("make", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_classifier_predict_does_not_modify_input(self, make, X, y_class):
        model = make().fit(X, y_class)
        X_before = X.copy()
        model.predict(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("make", REGRESSORS, ids=_ids(REGRESSORS))
    def test_regressor_predict_does_not_modify_input(self, make, X, y_reg):
        model = make().fit(X, y_reg)
        X_before = X.copy()
        model.predict(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("make", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_classifier_refit_on_new_feature_count_replaces_old_state(self, make, X, y_class):
        model = make().fit(X, y_class)
        model.fit(X[:, :2], y_class)
        assert model.predict(X[:5, :2]).shape == (5,)

    @pytest.mark.parametrize("make", REGRESSORS, ids=_ids(REGRESSORS))
    def test_regressor_refit_on_new_feature_count_replaces_old_state(self, make, X, y_reg):
        model = make().fit(X, y_reg)
        model.fit(X[:, :2], y_reg)
        assert model.predict(X[:5, :2]).shape == (5,)

    @pytest.mark.parametrize("make", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    def test_classifier_fitting_twice_gives_identical_predictions(self, make, X, y_class):
        first = make().fit(X, y_class).predict(X)
        second = make().fit(X, y_class).predict(X)
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("make", REGRESSORS, ids=_ids(REGRESSORS))
    def test_regressor_fitting_twice_gives_identical_predictions(self, make, X, y_reg):
        first = make().fit(X, y_reg).predict(X)
        second = make().fit(X, y_reg).predict(X)
        assert np.array_equal(first, second)


class TestTransformerContract:
    @pytest.mark.parametrize("make", TRANSFORMERS, ids=_ids(TRANSFORMERS))
    def test_fit_returns_self(self, make, X):
        model = make()
        assert model.fit(X) is model

    @pytest.mark.parametrize("make", TRANSFORMERS, ids=_ids(TRANSFORMERS))
    def test_transform_before_fit_raises(self, make, X):
        with pytest.raises(NotFittedError):
            make().transform(X)

    @pytest.mark.parametrize("make", TRANSFORMERS, ids=_ids(TRANSFORMERS))
    def test_rejects_wrong_feature_count(self, make, X):
        model = make().fit(X)
        with pytest.raises(ShapeMismatchError):
            model.transform(X[:, :2])

    @pytest.mark.parametrize("make", TRANSFORMERS, ids=_ids(TRANSFORMERS))
    def test_transform_does_not_modify_input(self, make, X):
        model = make().fit(X)
        X_before = X.copy()
        model.transform(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("make", TRANSFORMERS, ids=_ids(TRANSFORMERS))
    def test_inverse_transform_round_trips(self, make, X):
        model = make().fit(X)
        assert np.allclose(model.inverse_transform(model.transform(X)), X)

    @pytest.mark.parametrize("make", TRANSFORMERS, ids=_ids(TRANSFORMERS))
    def test_fit_transform_matches_fit_then_transform(self, make, X):
        assert np.allclose(make().fit_transform(X), make().fit(X).transform(X))


class TestClustererContract:
    @pytest.mark.parametrize("make", ALL_CLUSTERERS, ids=_ids(ALL_CLUSTERERS))
    def test_fit_returns_self(self, make, X):
        model = make()
        assert model.fit(X) is model

    @pytest.mark.parametrize("make", ALL_CLUSTERERS, ids=_ids(ALL_CLUSTERERS))
    def test_fit_does_not_modify_input(self, make, X):
        X_before = X.copy()
        make().fit(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("make", ALL_CLUSTERERS, ids=_ids(ALL_CLUSTERERS))
    def test_one_label_per_sample(self, make, X):
        assert make().fit_predict(X).shape == (X.shape[0],)

    @pytest.mark.parametrize("make", ALL_CLUSTERERS, ids=_ids(ALL_CLUSTERERS))
    def test_fitting_twice_gives_identical_labels(self, make, X):
        assert np.array_equal(make().fit_predict(X), make().fit_predict(X))

    @pytest.mark.parametrize("make", PREDICTABLE_CLUSTERERS, ids=_ids(PREDICTABLE_CLUSTERERS))
    def test_predict_before_fit_raises(self, make, X):
        with pytest.raises(NotFittedError):
            make().predict(X)

    @pytest.mark.parametrize("make", PREDICTABLE_CLUSTERERS, ids=_ids(PREDICTABLE_CLUSTERERS))
    def test_rejects_wrong_feature_count(self, make, X):
        model = make().fit(X)
        with pytest.raises(ShapeMismatchError):
            model.predict(X[:, :2])


ALL_ESTIMATORS = [*CLASSIFIERS, *REGRESSORS, *TRANSFORMERS, *ALL_CLUSTERERS]


class TestParamsContract:
    @pytest.mark.parametrize("make", ALL_ESTIMATORS, ids=_ids(ALL_ESTIMATORS))
    def test_get_params_round_trips_through_the_constructor(self, make):
        model = make()
        clone = type(model)(**model.get_params())
        assert clone.get_params() == model.get_params()

    @pytest.mark.parametrize("make", ALL_ESTIMATORS, ids=_ids(ALL_ESTIMATORS))
    def test_set_params_with_own_params_is_a_no_op(self, make):
        model = make()
        params = model.get_params()
        assert model.set_params(**params).get_params() == params

    @pytest.mark.parametrize("make", ALL_ESTIMATORS, ids=_ids(ALL_ESTIMATORS))
    def test_new_estimator_is_not_fitted(self, make):
        assert make().is_fitted_ is False
