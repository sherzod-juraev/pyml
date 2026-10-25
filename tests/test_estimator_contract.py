"""Behavior every public pyml estimator must share, checked across all of them.

The estimators come from ``tests/registry.py``. A completeness test
fails when a public estimator is missing from the registry, so a new
model cannot skip these checks.
"""

import copy
import importlib
import inspect
import pickle
from typing import Any

import numpy as np
import pytest

import pyml
from pyml.core.base.estimator import BaseEstimator
from pyml.core.base.validation import DataValidatorMixin
from pyml.core.exceptions import InvalidParameterError, NotFittedError, ShapeMismatchError
from tests.registry import ENTRIES, Entry, by_family, ids

CLASSIFIERS = by_family("classifier")
REGRESSORS = by_family("regressor")
TRANSFORMERS = by_family("transformer")
ALL_CLUSTERERS = by_family("clusterer")
PREDICTABLE_CLUSTERERS = by_family("clusterer", predictable=True)
INVALID_PARAMS = [(entry, params) for entry in ENTRIES for params in entry.invalid]
INVALID_PARAM_IDS = [f"{entry.name}-{params}" for entry, params in INVALID_PARAMS]


@pytest.fixture
def X(rng):
    return rng.uniform(1, 10, size=(60, 3))


@pytest.fixture
def y_class():
    return np.array([0, 1] * 30)


@pytest.fixture
def y_reg(rng):
    return rng.uniform(0, 10, size=60)


def _fit_args(entry: Entry, X, y_class, y_reg) -> tuple[Any, ...]:
    if entry.family == "classifier":
        return (X, y_class)
    if entry.family == "regressor":
        return (X, y_reg)
    return (X,)


def _fitted_state(model: BaseEstimator) -> bytes:
    state = {key: value for key, value in vars(model).items() if key.endswith("_")}
    return pickle.dumps(state)


def _output(entry: Entry, model: Any, X) -> Any:
    if entry.family == "transformer":
        return model.transform(X)
    if entry.family == "clusterer":
        return model.labels_
    return model.predict(X)


def _public_estimator_classes() -> set[type[BaseEstimator]]:
    classes: set[type[BaseEstimator]] = set()
    for package_name in vars(pyml)["__all__"]:
        package = importlib.import_module(f"pyml.{package_name}")
        for name in getattr(package, "__all__", []):
            member = getattr(package, name)
            if (
                inspect.isclass(member)
                and issubclass(member, BaseEstimator)
                and not inspect.isabstract(member)
            ):
                classes.add(member)
    return classes


class TestRegistry:
    def test_every_public_estimator_is_registered(self):
        registered = {entry.cls for entry in ENTRIES}

        assert _public_estimator_classes() == registered

    def test_registered_entries_are_unique(self):
        assert len({entry.cls for entry in ENTRIES}) == len(ENTRIES)


class TestSupervisedContract:
    @pytest.mark.parametrize("entry", CLASSIFIERS, ids=ids(CLASSIFIERS))
    def test_classifier_fit_returns_self(self, entry, X, y_class):
        model = entry.build()
        assert model.fit(X, y_class) is model

    @pytest.mark.parametrize("entry", REGRESSORS, ids=ids(REGRESSORS))
    def test_regressor_fit_returns_self(self, entry, X, y_reg):
        model = entry.build()
        assert model.fit(X, y_reg) is model

    @pytest.mark.parametrize("entry", CLASSIFIERS, ids=ids(CLASSIFIERS))
    def test_classifier_predict_before_fit_raises(self, entry, X):
        with pytest.raises(NotFittedError):
            entry.build().predict(X)

    @pytest.mark.parametrize("entry", REGRESSORS, ids=ids(REGRESSORS))
    def test_regressor_predict_before_fit_raises(self, entry, X):
        with pytest.raises(NotFittedError):
            entry.build().predict(X)

    @pytest.mark.parametrize("entry", CLASSIFIERS, ids=ids(CLASSIFIERS))
    def test_classifier_rejects_wrong_feature_count(self, entry, X, y_class):
        model = entry.build().fit(X, y_class)
        with pytest.raises(ShapeMismatchError):
            model.predict(X[:, :2])

    @pytest.mark.parametrize("entry", REGRESSORS, ids=ids(REGRESSORS))
    def test_regressor_rejects_wrong_feature_count(self, entry, X, y_reg):
        model = entry.build().fit(X, y_reg)
        with pytest.raises(ShapeMismatchError):
            model.predict(X[:, :2])

    @pytest.mark.parametrize("entry", CLASSIFIERS, ids=ids(CLASSIFIERS))
    def test_classifier_fit_does_not_modify_inputs(self, entry, X, y_class):
        X_before, y_before = X.copy(), y_class.copy()
        entry.build().fit(X, y_class)
        assert np.array_equal(X, X_before)
        assert np.array_equal(y_class, y_before)

    @pytest.mark.parametrize("entry", REGRESSORS, ids=ids(REGRESSORS))
    def test_regressor_fit_does_not_modify_inputs(self, entry, X, y_reg):
        X_before, y_before = X.copy(), y_reg.copy()
        entry.build().fit(X, y_reg)
        assert np.array_equal(X, X_before)
        assert np.array_equal(y_reg, y_before)

    @pytest.mark.parametrize("entry", CLASSIFIERS, ids=ids(CLASSIFIERS))
    def test_classifier_predict_does_not_modify_input(self, entry, X, y_class):
        model = entry.build().fit(X, y_class)
        X_before = X.copy()
        model.predict(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("entry", REGRESSORS, ids=ids(REGRESSORS))
    def test_regressor_predict_does_not_modify_input(self, entry, X, y_reg):
        model = entry.build().fit(X, y_reg)
        X_before = X.copy()
        model.predict(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("entry", CLASSIFIERS, ids=ids(CLASSIFIERS))
    def test_classifier_refit_on_new_feature_count_replaces_old_state(self, entry, X, y_class):
        model = entry.build().fit(X, y_class)
        model.fit(X[:, :2], y_class)
        assert model.predict(X[:5, :2]).shape == (5,)

    @pytest.mark.parametrize("entry", REGRESSORS, ids=ids(REGRESSORS))
    def test_regressor_refit_on_new_feature_count_replaces_old_state(self, entry, X, y_reg):
        model = entry.build().fit(X, y_reg)
        model.fit(X[:, :2], y_reg)
        assert model.predict(X[:5, :2]).shape == (5,)

    @pytest.mark.parametrize("entry", CLASSIFIERS, ids=ids(CLASSIFIERS))
    def test_classifier_fitting_twice_gives_identical_predictions(self, entry, X, y_class):
        first = entry.build().fit(X, y_class).predict(X)
        second = entry.build().fit(X, y_class).predict(X)
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("entry", REGRESSORS, ids=ids(REGRESSORS))
    def test_regressor_fitting_twice_gives_identical_predictions(self, entry, X, y_reg):
        first = entry.build().fit(X, y_reg).predict(X)
        second = entry.build().fit(X, y_reg).predict(X)
        assert np.array_equal(first, second)


class TestTransformerContract:
    @pytest.mark.parametrize("entry", TRANSFORMERS, ids=ids(TRANSFORMERS))
    def test_fit_returns_self(self, entry, X):
        model = entry.build()
        assert model.fit(X) is model

    @pytest.mark.parametrize("entry", TRANSFORMERS, ids=ids(TRANSFORMERS))
    def test_transform_before_fit_raises(self, entry, X):
        with pytest.raises(NotFittedError):
            entry.build().transform(X)

    @pytest.mark.parametrize("entry", TRANSFORMERS, ids=ids(TRANSFORMERS))
    def test_rejects_wrong_feature_count(self, entry, X):
        model = entry.build().fit(X)
        with pytest.raises(ShapeMismatchError):
            model.transform(X[:, :2])

    @pytest.mark.parametrize("entry", TRANSFORMERS, ids=ids(TRANSFORMERS))
    def test_transform_does_not_modify_input(self, entry, X):
        model = entry.build().fit(X)
        X_before = X.copy()
        model.transform(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("entry", TRANSFORMERS, ids=ids(TRANSFORMERS))
    def test_inverse_transform_round_trips(self, entry, X):
        model = entry.build().fit(X)
        assert np.allclose(model.inverse_transform(model.transform(X)), X)

    @pytest.mark.parametrize("entry", TRANSFORMERS, ids=ids(TRANSFORMERS))
    def test_fit_transform_matches_fit_then_transform(self, entry, X):
        assert np.allclose(entry.build().fit_transform(X), entry.build().fit(X).transform(X))


class TestClustererContract:
    @pytest.mark.parametrize("entry", ALL_CLUSTERERS, ids=ids(ALL_CLUSTERERS))
    def test_fit_returns_self(self, entry, X):
        model = entry.build()
        assert model.fit(X) is model

    @pytest.mark.parametrize("entry", ALL_CLUSTERERS, ids=ids(ALL_CLUSTERERS))
    def test_fit_does_not_modify_input(self, entry, X):
        X_before = X.copy()
        entry.build().fit(X)
        assert np.array_equal(X, X_before)

    @pytest.mark.parametrize("entry", ALL_CLUSTERERS, ids=ids(ALL_CLUSTERERS))
    def test_one_label_per_sample(self, entry, X):
        assert entry.build().fit_predict(X).shape == (X.shape[0],)

    @pytest.mark.parametrize("entry", ALL_CLUSTERERS, ids=ids(ALL_CLUSTERERS))
    def test_fitting_twice_gives_identical_labels(self, entry, X):
        assert np.array_equal(entry.build().fit_predict(X), entry.build().fit_predict(X))

    @pytest.mark.parametrize("entry", PREDICTABLE_CLUSTERERS, ids=ids(PREDICTABLE_CLUSTERERS))
    def test_predict_before_fit_raises(self, entry, X):
        with pytest.raises(NotFittedError):
            entry.build().predict(X)

    @pytest.mark.parametrize("entry", PREDICTABLE_CLUSTERERS, ids=ids(PREDICTABLE_CLUSTERERS))
    def test_rejects_wrong_feature_count(self, entry, X):
        model = entry.build().fit(X)
        with pytest.raises(ShapeMismatchError):
            model.predict(X[:, :2])


class TestParamsContract:
    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_get_params_round_trips_through_the_constructor(self, entry):
        model = entry.build()
        clone = type(model)(**model.get_params())
        assert clone.get_params() == model.get_params()

    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_set_params_with_own_params_is_a_no_op(self, entry):
        model = entry.build()
        params = model.get_params()
        assert model.set_params(**params).get_params() == params

    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_new_estimator_is_not_fitted(self, entry):
        assert entry.build().is_fitted_ is False

    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_estimator_with_hyperparameters_overrides_the_validation_hook(self, entry):
        has_params = bool(entry.build().get_params())
        overrides = entry.cls._validate_params is not DataValidatorMixin._validate_params

        assert overrides == has_params


class TestHyperparameterValidation:
    @pytest.mark.parametrize(("entry", "params"), INVALID_PARAMS, ids=INVALID_PARAM_IDS)
    def test_invalid_hyperparameters_are_rejected_on_fit(self, entry, params, X, y_class, y_reg):
        model = entry.build(**params)

        with pytest.raises(InvalidParameterError):
            model.fit(*_fit_args(entry, X, y_class, y_reg))

    @pytest.mark.parametrize(("entry", "params"), INVALID_PARAMS, ids=INVALID_PARAM_IDS)
    def test_hyperparameters_are_checked_before_the_data(self, entry, params, rng, y_class, y_reg):
        X_1d = rng.uniform(1, 10, size=60)
        model = entry.build(**params)

        with pytest.raises(InvalidParameterError):
            model.fit(*_fit_args(entry, X_1d, y_class, y_reg))

    @pytest.mark.parametrize(("entry", "params"), INVALID_PARAMS, ids=INVALID_PARAM_IDS)
    def test_failed_refit_keeps_the_previous_fit(self, entry, params, X, y_class, y_reg):
        model = entry.build().fit(*_fit_args(entry, X, y_class, y_reg))
        state_before = _fitted_state(model)
        output_before = _output(entry, model, X)

        valid_params = {name: model.get_params()[name] for name in params}
        model.set_params(**params)
        with pytest.raises(InvalidParameterError):
            model.fit(*_fit_args(entry, X[:, :2], y_class, y_reg))

        assert model.n_features_in_ == X.shape[1]
        assert _fitted_state(model) == state_before
        model.set_params(**valid_params)
        assert np.array_equal(_output(entry, model, X), output_before)


class TestCopyContract:
    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_deepcopy_of_a_fitted_estimator_gives_the_same_output(self, entry, X, y_class, y_reg):
        model = entry.build().fit(*_fit_args(entry, X, y_class, y_reg))

        clone = copy.deepcopy(model)

        assert np.array_equal(_output(entry, clone, X), _output(entry, model, X))

    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_pickle_round_trip_of_a_fitted_estimator_gives_the_same_output(
        self, entry, X, y_class, y_reg
    ):
        model = entry.build().fit(*_fit_args(entry, X, y_class, y_reg))

        clone = pickle.loads(pickle.dumps(model))

        assert np.array_equal(_output(entry, clone, X), _output(entry, model, X))
