"""Tests that pyml's estimators match its protocols.

The protocols are checked by mypy, not at run time: assigning an estimator to a
variable typed with a protocol fails mypy when the estimator lacks one of the
methods. These tests also call the methods through the protocol type, so they
run as well. A case that must not match carries ``# type: ignore[assignment]``,
which mypy reports as an error as soon as the match starts to work, so it guards
against a protocol that has become too loose. The protocols of the estimator
roles are also checked with ``isinstance``.
"""

from pyml.cluster import DBSCAN, KMeans
from pyml.core.protocols import (
    ClustererLike,
    PredictableClustererLike,
    SupervisedLike,
    SupportsParams,
    SupportsPredictProba,
    TransformerLike,
)
from pyml.ensemble import RandomForestClassifier, RandomForestRegressor
from pyml.linear_model import LinearRegression, LogisticRegression, Ridge
from pyml.neighbors import KNNClassifier, KNNRegressor
from pyml.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pyml.tree import DecisionTreeClassifier, DecisionTreeRegressor


class _WrongSignatures:
    """Has every method of a transformer by name, none with a usable signature."""

    def get_params(self):
        return {}

    def set_params(self):
        return self

    def fit_transform(self):
        return None

    def transform(self):
        return None


class TestSupportsParams:
    def test_every_kind_of_estimator_has_the_parameter_methods(self):
        estimators: list[SupportsParams] = [
            StandardScaler(),
            Ridge(),
            LogisticRegression(),
            KMeans(n_clusters=2),
            DBSCAN(),
            DecisionTreeClassifier(max_depth=3),
        ]

        for estimator in estimators:
            assert isinstance(estimator.get_params(), dict)
            assert estimator.set_params() is estimator


class TestTransformerLike:
    def test_scalers_transform_through_the_protocol(self, rng):
        X = rng.random((20, 3))
        transformers: list[TransformerLike] = [StandardScaler(), MinMaxScaler(), RobustScaler()]

        for transformer in transformers:
            assert transformer.fit_transform(X).shape == X.shape
            assert transformer.transform(X).shape == X.shape


class TestSupervisedLike:
    def test_regressors_predict_and_score_through_the_protocol(self, linear_data):
        X, y = linear_data(n_samples=60)
        models: list[SupervisedLike] = [
            Ridge(),
            LinearRegression(),
            KNNRegressor(),
            DecisionTreeRegressor(max_depth=3),
            RandomForestRegressor(max_depth=3, n_estimators=5, random_state=0),
        ]

        for model in models:
            model.fit(X, y)

            assert model.predict(X).shape == (60,)
            assert isinstance(model.score(X, y), float)

    def test_classifiers_predict_and_score_through_the_protocol(self, two_blobs):
        X, y = two_blobs(n_per_class=20)
        models: list[SupervisedLike] = [
            LogisticRegression(),
            KNNClassifier(),
            DecisionTreeClassifier(max_depth=3),
            RandomForestClassifier(max_depth=3, n_estimators=5, random_state=0),
        ]

        for model in models:
            model.fit(X, y)

            assert model.predict(X).shape == (40,)
            assert isinstance(model.score(X, y), float)


class TestClustererLike:
    def test_clusterers_fit_predict_through_the_protocol(self, two_blobs):
        X, _ = two_blobs(n_per_class=20)
        clusterers: list[ClustererLike] = [KMeans(n_clusters=2, random_state=0), DBSCAN()]

        for clusterer in clusterers:
            assert clusterer.fit_predict(X).shape == (40,)


class TestPredictableClustererLike:
    def test_kmeans_assigns_new_samples_through_the_protocol(self, two_blobs):
        X, _ = two_blobs(n_per_class=20)
        clusterer: PredictableClustererLike = KMeans(n_clusters=2, random_state=0)

        clusterer.fit(X)

        assert clusterer.predict(X).shape == (40,)


class TestSupportsPredictProba:
    def test_probabilistic_classifiers_predict_probabilities_through_the_protocol(self, two_blobs):
        X, y = two_blobs(n_per_class=20)
        models: list[SupportsPredictProba] = [
            LogisticRegression().fit(X, y),
            DecisionTreeClassifier(max_depth=3).fit(X, y),
            RandomForestClassifier(max_depth=3, n_estimators=5, random_state=0).fit(X, y),
        ]

        for model in models:
            assert model.predict_proba(X).shape[0] == 40


class TestMismatches:
    def test_knn_classifier_does_not_match_supports_predict_proba(self):
        model: SupportsPredictProba = KNNClassifier()  # type: ignore[assignment]

        assert not hasattr(model, "predict_proba")

    def test_scaler_does_not_match_supervised_like(self):
        model: SupervisedLike = StandardScaler()  # type: ignore[assignment]

        assert not hasattr(model, "predict")

    def test_regressor_does_not_match_transformer_like(self):
        model: TransformerLike = Ridge()  # type: ignore[assignment]

        assert not hasattr(model, "transform")

    def test_dbscan_does_not_match_predictable_clusterer_like(self):
        model: PredictableClustererLike = DBSCAN()  # type: ignore[assignment]

        assert not hasattr(model, "predict")


class TestRuntimeCheck:
    def test_isinstance_accepts_estimators_that_have_the_methods(self):
        assert isinstance(StandardScaler(), TransformerLike)
        assert isinstance(Ridge(), SupervisedLike)
        assert isinstance(KMeans(n_clusters=2), ClustererLike)
        assert isinstance(KMeans(n_clusters=2), PredictableClustererLike)
        assert isinstance(DBSCAN(), ClustererLike)
        assert isinstance(LogisticRegression(), SupportsPredictProba)

    def test_every_estimator_matches_supports_params(self):
        estimators = [
            StandardScaler(),
            Ridge(),
            LogisticRegression(),
            KMeans(n_clusters=2),
            DBSCAN(),
        ]

        assert all(isinstance(estimator, SupportsParams) for estimator in estimators)

    def test_isinstance_rejects_estimators_without_a_method(self):
        assert not isinstance(Ridge(), TransformerLike)
        assert not isinstance(StandardScaler(), SupervisedLike)
        assert not isinstance(StandardScaler(), ClustererLike)
        assert not isinstance(DBSCAN(), PredictableClustererLike)
        assert not isinstance(KNNClassifier(), SupportsPredictProba)

    def test_an_object_without_the_methods_matches_no_protocol(self):
        assert not isinstance(object(), SupportsParams)
        assert not isinstance([], TransformerLike)

    def test_isinstance_checks_method_names_not_signatures(self):
        assert isinstance(_WrongSignatures(), TransformerLike)
