"""Registry of every public estimator, shared by the cross-estimator tests.

Each entry describes one estimator: its family, the hyper-parameters
used to build a small, fast instance, and the invalid hyper-parameter
combinations it must reject. The contract tests are parametrized over
this registry, and a completeness test fails when a public estimator
is missing from it.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from pyml.cluster import DBSCAN, KMeans
from pyml.core.base.estimator import BaseEstimator
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

Family = Literal["classifier", "regressor", "transformer", "clusterer"]

_NAN = float("nan")
_INF = float("inf")


@dataclass(frozen=True)
class Entry:
    """One registered estimator."""

    cls: type[BaseEstimator]
    family: Family
    defaults: dict[str, Any] = field(default_factory=dict)
    invalid: tuple[dict[str, Any], ...] = ()
    predictable: bool = False

    @property
    def name(self) -> str:
        return self.cls.__name__

    def build(self, **overrides: Any) -> Any:
        return self.cls(**{**self.defaults, **overrides})


_SOLVER_INVALID: tuple[dict[str, Any], ...] = (
    {"learning_rate": 0.0},
    {"learning_rate": -1.0},
    {"learning_rate": _INF},
    {"learning_rate": _NAN},
    {"max_iter": 0},
    {"tol": -1e-9},
    {"tol": _NAN},
)
_ALPHA_INVALID: tuple[dict[str, Any], ...] = ({"alpha": -0.1}, {"alpha": _NAN}, {"alpha": _INF})
_L1_RATIO_INVALID: tuple[dict[str, Any], ...] = (
    {"l1_ratio": -0.1},
    {"l1_ratio": 1.1},
    {"l1_ratio": _NAN},
)
_NEIGHBORS_INVALID: tuple[dict[str, Any], ...] = (
    {"weights": "distnce"},
    {"metric": "euclidian"},
)
_TREE_INVALID: tuple[dict[str, Any], ...] = (
    {"max_depth": -1},
    {"max_depth": _NAN},
    {"min_samples": 0},
    {"min_samples": _NAN},
    {"min_impurity_decrease": -1.0},
    {"min_impurity_decrease": _NAN},
)

ENTRIES: tuple[Entry, ...] = (
    # --- classifiers ---
    Entry(LogisticRegression, "classifier", invalid=_SOLVER_INVALID),
    Entry(RidgeClassifier, "classifier", invalid=(*_SOLVER_INVALID, *_ALPHA_INVALID)),
    Entry(LassoClassifier, "classifier", invalid=(*_SOLVER_INVALID, *_ALPHA_INVALID)),
    Entry(
        ElasticNetClassifier,
        "classifier",
        invalid=(*_SOLVER_INVALID, *_ALPHA_INVALID, *_L1_RATIO_INVALID),
    ),
    Entry(GaussianNB, "classifier"),
    Entry(MultinomialNB, "classifier", invalid=_ALPHA_INVALID),
    Entry(
        KNNClassifier,
        "classifier",
        defaults={"n_neighbors": 3},
        invalid=(
            {"n_neighbors": 0},
            {"n_neighbors": -1},
            {"n_neighbors": _NAN},
            *_NEIGHBORS_INVALID,
        ),
    ),
    Entry(
        RadiusNeighborsClassifier,
        "classifier",
        defaults={"radius": 50.0},
        invalid=(
            {"radius": 0.0},
            {"radius": _NAN},
            {"radius": _INF},
            *_NEIGHBORS_INVALID,
        ),
    ),
    Entry(
        DecisionTreeClassifier,
        "classifier",
        defaults={"max_depth": 3},
        invalid=_TREE_INVALID,
    ),
    # --- regressors ---
    Entry(LinearRegression, "regressor", invalid=_SOLVER_INVALID),
    Entry(Ridge, "regressor", invalid=(*_SOLVER_INVALID, *_ALPHA_INVALID)),
    Entry(Lasso, "regressor", invalid=(*_SOLVER_INVALID, *_ALPHA_INVALID)),
    Entry(
        ElasticNet,
        "regressor",
        invalid=(*_SOLVER_INVALID, *_ALPHA_INVALID, *_L1_RATIO_INVALID),
    ),
    Entry(
        KNNRegressor,
        "regressor",
        defaults={"n_neighbors": 3},
        invalid=(
            {"n_neighbors": 0},
            {"n_neighbors": -1},
            {"n_neighbors": _NAN},
            *_NEIGHBORS_INVALID,
        ),
    ),
    Entry(
        RadiusNeighborsRegressor,
        "regressor",
        defaults={"radius": 50.0},
        invalid=(
            {"radius": 0.0},
            {"radius": _NAN},
            {"radius": _INF},
            *_NEIGHBORS_INVALID,
        ),
    ),
    Entry(
        DecisionTreeRegressor,
        "regressor",
        defaults={"max_depth": 3},
        invalid=_TREE_INVALID,
    ),
    # --- transformers ---
    Entry(MinMaxScaler, "transformer"),
    Entry(RobustScaler, "transformer"),
    Entry(StandardScaler, "transformer"),
    # --- clusterers ---
    Entry(
        KMeans,
        "clusterer",
        defaults={"n_clusters": 2, "random_state": 42},
        invalid=(
            {"n_clusters": 0},
            {"n_clusters": -3},
            {"n_clusters": _NAN},
            {"max_iter": 0},
            {"tol": -1e-9},
            {"tol": _NAN},
            {"random_state": -1},
        ),
        predictable=True,
    ),
    Entry(
        DBSCAN,
        "clusterer",
        defaults={"eps": 3.0, "min_samples": 3},
        invalid=(
            {"eps": 0.0},
            {"eps": _NAN},
            {"eps": _INF},
            {"min_samples": 0},
            {"metric": "euclidian"},
        ),
    ),
)


def by_family(family: Family, *, predictable: bool | None = None) -> list[Entry]:
    entries = [entry for entry in ENTRIES if entry.family == family]
    if predictable is not None:
        entries = [entry for entry in entries if entry.predictable is predictable]
    return entries


def ids(entries: list[Entry]) -> list[str]:
    return [entry.name for entry in entries]
