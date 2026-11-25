"""Tests for Pipeline.

The estimators inside are tested on their own, so these tests check what the
pipeline adds: the validation of ``steps``, the order in which the data flows
through the steps, and the guards on the last step. The expected results are
computed by chaining the same steps by hand.
"""

from typing import Any

import numpy as np
import pytest

from pyml.cluster import DBSCAN, KMeans
from pyml.core.exceptions import (
    InvalidParameterError,
    NotFittedError,
    ShapeMismatchError,
    ValidationError,
)
from pyml.linear_model import LogisticRegression, Ridge
from pyml.pipeline import Pipeline
from pyml.preprocessing import MinMaxScaler, StandardScaler


@pytest.fixture
def regression_data(linear_data):
    X, y = linear_data(n_samples=60)
    return X * np.array([100.0, 0.01]) + 50.0, y


class TestValidateSteps:
    @pytest.mark.parametrize(
        "steps",
        [
            lambda: [("scaler", StandardScaler()), ("model", Ridge())],
            lambda: [("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=2))],
            lambda: [("model", Ridge())],
            lambda: [("dbscan", DBSCAN())],
        ],
    )
    def test_accepts_valid_steps(self, steps, linear_data):
        X, y = linear_data(n_samples=60)
        pipeline = Pipeline(steps())
        supervised = isinstance(pipeline.steps[-1][1], Ridge)

        pipeline.fit(X, y if supervised else None)

        assert pipeline.is_fitted_

    @pytest.mark.parametrize(
        ("steps", "message"),
        [
            ([], "non-empty list"),
            ((("model", Ridge()),), "non-empty list"),
            ([Ridge()], r"steps\[0\] must be a \(name, estimator\) tuple"),
            ([("model", Ridge(), 1)], r"steps\[0\] must be a \(name, estimator\) tuple"),
            ([(1, Ridge())], r"steps\[0\] must be a \(name, estimator\) tuple"),
            ([("scaler", StandardScaler()), "model"], r"steps\[1\] must be a"),
            ([("", Ridge())], "non-empty"),
            ([("a__b", Ridge())], "'__'"),
            ([("x", StandardScaler()), ("x", Ridge())], "unique"),
            ([("model", Ridge()), ("scaler", StandardScaler())], "'model' \\(Ridge\\)"),
            ([("kmeans", KMeans(n_clusters=2)), ("model", Ridge())], "'kmeans' \\(KMeans\\)"),
            ([("scaler", StandardScaler())], "'scaler' \\(StandardScaler\\)"),
            ([("scaler", StandardScaler()), ("other", MinMaxScaler())], "'other'"),
            ([("model", object())], "'model' \\(object\\)"),
        ],
    )
    def test_rejects_invalid_steps(self, steps, message, regression_data):
        X, y = regression_data

        with pytest.raises(InvalidParameterError, match=message):
            Pipeline(steps).fit(X, y)

    def test_error_names_the_step(self, regression_data):
        X, y = regression_data
        steps: list[tuple[str, Any]] = [
            ("scaler", StandardScaler()),
            ("model", Ridge()),
            ("late", Ridge()),
        ]

        with pytest.raises(InvalidParameterError, match="'model'"):
            Pipeline(steps).fit(X, y)

    @pytest.mark.parametrize("method", ["predict", "predict_proba", "score"])
    def test_steps_replaced_after_fit_are_checked_again(self, method, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())]).fit(X, y)

        pipeline.set_params(steps=[])

        with pytest.raises(InvalidParameterError, match="non-empty list"):
            getattr(pipeline, method)(X, y) if method == "score" else getattr(pipeline, method)(X)


class TestFit:
    def test_returns_self(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        assert pipeline.fit(X, y) is pipeline

    def test_is_fitted_flag(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        before = pipeline.is_fitted_
        pipeline.fit(X, y)

        assert before is False
        assert pipeline.is_fitted_ is True

    def test_fits_the_steps_in_place(self, regression_data):
        X, y = regression_data
        scaler, model = StandardScaler(), Ridge()

        Pipeline([("scaler", scaler), ("model", model)]).fit(X, y)

        assert np.allclose(scaler.mean_, X.mean(axis=0))
        assert model.is_fitted_

    def test_each_step_gets_the_output_of_the_one_before(self, regression_data):
        X, y = regression_data
        first, second = MinMaxScaler(), StandardScaler()

        Pipeline([("first", first), ("second", second), ("model", Ridge())]).fit(X, y)

        expected = MinMaxScaler().fit_transform(X)
        assert np.allclose(second.mean_, expected.mean(axis=0))

    def test_last_step_is_fitted_on_the_transformed_data(self, regression_data):
        X, y = regression_data
        model = Ridge()
        Pipeline([("scaler", StandardScaler()), ("model", model)]).fit(X, y)
        Xt = StandardScaler().fit_transform(X)

        reference = Ridge().fit(Xt, y)

        assert np.allclose(model.coef_, reference.coef_)

    def test_supervised_last_step_requires_y(self, regression_data):
        X, _ = regression_data

        with pytest.raises(ValidationError, match="y is required"):
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]).fit(X)

    def test_clusterer_last_step_rejects_y(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=2))])

        with pytest.raises(ValidationError, match="y must be None"):
            pipeline.fit(X, y)

    def test_failed_validation_does_not_mark_the_pipeline_fitted(self, regression_data):
        X, _ = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        with pytest.raises(ValidationError):
            pipeline.fit(X)

        assert pipeline.is_fitted_ is False

    def test_data_errors_come_from_the_step(self, regression_data):
        X, y = regression_data
        X = X.copy()
        X[0, 0] = np.nan
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        with pytest.raises(ValueError, match="NaN"):
            pipeline.fit(X, y)


class TestPredict:
    def test_matches_the_steps_chained_by_hand(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())]).fit(X, y)
        scaler = StandardScaler().fit(X)
        model = Ridge().fit(scaler.transform(X), y)

        result = pipeline.predict(X)

        assert np.allclose(result, model.predict(scaler.transform(X)))

    def test_uses_transform_not_a_new_fit(self, regression_data):
        X, y = regression_data
        scaler = StandardScaler()
        pipeline = Pipeline([("scaler", scaler), ("model", Ridge())]).fit(X, y)
        mean = scaler.mean_.copy()

        pipeline.predict(X[:5] + 1000.0)

        assert np.allclose(scaler.mean_, mean)

    def test_clusterer_as_last_step(self, two_blobs):
        X, _ = two_blobs()
        pipeline = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(2, random_state=0))])

        pipeline.fit(X)
        scaler = StandardScaler().fit(X)
        kmeans = KMeans(2, random_state=0).fit(scaler.transform(X))

        assert np.array_equal(pipeline.predict(X), kmeans.predict(scaler.transform(X)))

    def test_clusterer_without_predict_is_rejected(self, two_blobs):
        X, _ = two_blobs()
        pipeline = Pipeline([("scaler", StandardScaler()), ("dbscan", DBSCAN())]).fit(X)

        with pytest.raises(InvalidParameterError, match="DBSCAN has no predict method"):
            pipeline.predict(X)

    def test_before_fit_raises(self, regression_data):
        X, _ = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        with pytest.raises(NotFittedError, match="Pipeline"):
            pipeline.predict(X)

    def test_wrong_number_of_features_is_reported_by_the_first_step(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())]).fit(X, y)

        with pytest.raises(ShapeMismatchError):
            pipeline.predict(X[:, :1])


class TestPredictProba:
    def test_matches_the_steps_chained_by_hand(self, two_blobs):
        X, y = two_blobs()
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
        pipeline.fit(X, y)
        scaler = StandardScaler().fit(X)
        model = LogisticRegression().fit(scaler.transform(X), y)

        result = pipeline.predict_proba(X)

        assert np.allclose(result, model.predict_proba(scaler.transform(X)))

    def test_model_without_predict_proba_is_rejected(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())]).fit(X, y)

        with pytest.raises(InvalidParameterError, match="Ridge has no predict_proba"):
            pipeline.predict_proba(X)

    def test_before_fit_raises(self, two_blobs):
        X, _ = two_blobs()
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])

        with pytest.raises(NotFittedError, match="Pipeline"):
            pipeline.predict_proba(X)


class TestScore:
    def test_regressor_matches_the_steps_chained_by_hand(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())]).fit(X, y)
        scaler = StandardScaler().fit(X)
        model = Ridge().fit(scaler.transform(X), y)

        assert pipeline.score(X, y) == pytest.approx(model.score(scaler.transform(X), y))

    def test_classifier_matches_the_steps_chained_by_hand(self, two_blobs):
        X, y = two_blobs()
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
        pipeline.fit(X, y)
        scaler = StandardScaler().fit(X)
        model = LogisticRegression().fit(scaler.transform(X), y)

        assert pipeline.score(X, y) == pytest.approx(model.score(scaler.transform(X), y))

    def test_clusterer_has_no_score(self, two_blobs):
        X, y = two_blobs()
        pipeline = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(2, random_state=0))])
        pipeline.fit(X)

        with pytest.raises(InvalidParameterError, match="KMeans has no score"):
            pipeline.score(X, y)

    def test_before_fit_raises(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        with pytest.raises(NotFittedError, match="Pipeline"):
            pipeline.score(X, y)


class TestParams:
    def test_get_params_returns_the_steps(self):
        steps: list[tuple[str, Any]] = [("scaler", StandardScaler()), ("model", Ridge())]

        assert Pipeline(steps).get_params() == {"steps": steps}

    def test_set_params_replaces_the_steps(self, regression_data):
        X, y = regression_data
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        pipeline.set_params(steps=[("scaler", MinMaxScaler()), ("model", Ridge())]).fit(X, y)

        assert isinstance(pipeline.steps[0][1], MinMaxScaler)
        assert pipeline.is_fitted_


class TestGetItem:
    @pytest.fixture
    def parts(self):
        return StandardScaler(), MinMaxScaler(), Ridge()

    @pytest.fixture
    def pipeline(self, parts):
        scaler, minmax, model = parts
        return Pipeline([("scaler", scaler), ("minmax", minmax), ("model", model)])

    def test_position_returns_the_estimator_itself(self, pipeline, parts):
        assert pipeline[0] is parts[0]
        assert pipeline[2] is parts[2]

    def test_name_returns_the_estimator_itself(self, pipeline, parts):
        assert pipeline["minmax"] is parts[1]

    def test_negative_position_counts_from_the_end(self, pipeline, parts):
        assert pipeline[-1] is parts[2]
        assert pipeline[-3] is parts[0]

    @pytest.mark.parametrize("position", [3, 100, -4, -100])
    def test_position_outside_the_steps_raises_index_error(self, pipeline, position):
        with pytest.raises(IndexError, match="out of range for a pipeline with 3 steps"):
            pipeline[position]

    def test_unknown_name_raises_key_error_with_the_available_names(self, pipeline):
        with pytest.raises(KeyError, match="'missing'.*'scaler', 'minmax', 'model'"):
            pipeline["missing"]

    @pytest.mark.parametrize("item", [True, False, 1.0, None, slice(0, 2), np.int64(1)])
    def test_other_types_raise_type_error(self, pipeline, item):
        with pytest.raises(TypeError, match="must be str or int"):
            pipeline[item]

    def test_invalid_steps_are_reported_first(self):
        with pytest.raises(InvalidParameterError, match="non-empty list"):
            Pipeline([])[0]

    def test_works_before_fit(self, pipeline):
        assert isinstance(pipeline["scaler"], StandardScaler)


class TestLen:
    def test_counts_the_steps(self):
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        assert len(pipeline) == 2

    def test_follows_set_params(self):
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        pipeline.set_params(steps=[("model", Ridge())])

        assert len(pipeline) == 1
