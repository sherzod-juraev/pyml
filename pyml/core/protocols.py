"""Protocols describing what one part of pyml needs from another.

A protocol names the methods a part relies on, so the types that provide them do
not have to inherit from it.

The protocols of the estimator roles also work with ``isinstance``. That check
only looks for the methods by name and does not compare their signatures, so
mypy stays the judge of the signatures.
"""

from typing import Any, Protocol, Self, runtime_checkable

import numpy.typing as npt

from .dtypes import FeatureMatrix


class _ForestTree(Protocol):
    """What a forest needs from each of its trees.

    The forest only grows its trees, checks their parameters and asks them for
    predictions, so any tree with these methods fits. Because this is a
    protocol, the decision trees do not have to inherit from it.
    """

    def fit(self, X: FeatureMatrix, y: npt.NDArray[Any], /) -> Self:
        """Fit the tree on the given samples.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        Self
            The fitted tree.
        """

    def predict(self, X: FeatureMatrix, /) -> npt.NDArray[Any]:
        """Predict the target of each sample.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predictions of shape (n_samples,).
        """

    def _validate_params(self) -> None:
        """Check the hyper-parameters of the tree.

        Raises
        ------
        InvalidParameterError
            If a hyper-parameter of the tree is invalid.
        """


@runtime_checkable
class SupportsParams(Protocol):
    """An estimator whose constructor parameters can be read and changed.

    .. versionadded:: 0.6.0

    Every pyml estimator provides these two methods through ``BaseEstimator``.
    Code that works with an estimator from the outside, such as a pipeline that
    reaches a parameter of one of its steps, relies on them.
    """

    def get_params(self) -> dict[str, object]:
        """Return the constructor parameters and their values.

        Returns
        -------
        dict[str, object]
            Mapping of parameter name to its current value.
        """

    def set_params(self, **params: object) -> Self:
        """Set one or more constructor parameters.

        Parameters
        ----------
        **params : object
            Parameter names and the values to set.

        Returns
        -------
        Self
            The estimator itself, for method chaining.
        """


@runtime_checkable
class TransformerLike(SupportsParams, Protocol):
    """A transformer: an estimator that learns from data and reshapes it.

    .. versionadded:: 0.6.0

    Matches pyml's ``Transformer`` by its methods, without inheriting from it.
    """

    def fit_transform(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Fit the transformer to X, then transform X.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Transformed data with one row per sample.
        """

    def transform(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Apply the learned transformation to X.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Transformed data with one row per sample.
        """


@runtime_checkable
class SupervisedLike(SupportsParams, Protocol):
    """An estimator that learns from features and targets and predicts targets.

    .. versionadded:: 0.6.0

    Matches pyml's ``Regressor`` and ``Classifier`` by their methods, without
    inheriting from them.
    """

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any], /) -> Self:
        """Fit the estimator on training data.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        Self
            The fitted estimator.
        """

    def predict(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Predict the target of each sample.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predictions of shape (n_samples,).
        """

    def score(self, X: npt.NDArray[Any], y: npt.NDArray[Any], /) -> float:
        """Score the predictions for X against the true targets y.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        y : numpy.ndarray
            True target values of shape (n_samples,).

        Returns
        -------
        float
            Score of the estimator, where a higher score is better. The kind of
            score depends on the estimator, :math:`R^2` for a regressor and the
            accuracy for a classifier.
        """


@runtime_checkable
class ClustererLike(SupportsParams, Protocol):
    """A clusterer: an estimator that groups samples without any targets.

    .. versionadded:: 0.6.0

    Matches pyml's ``Clusterer`` by its methods, without inheriting from it.
    """

    def fit(self, X: npt.NDArray[Any], /) -> Self:
        """Fit the clusterer to X.

        Parameters
        ----------
        X : numpy.ndarray
            Data to cluster, of shape (n_samples, n_features).

        Returns
        -------
        Self
            The fitted clusterer.
        """

    def fit_predict(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Fit the clusterer to X and return the cluster labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data to cluster, of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Cluster label of each sample, of shape (n_samples,).
        """


@runtime_checkable
class PredictableClustererLike(ClustererLike, Protocol):
    """A clusterer that can also assign clusters to new samples.

    .. versionadded:: 0.6.0

    Matches pyml's ``PredictableClusterer``. ``KMeans`` can do this, ``DBSCAN``
    cannot.
    """

    def predict(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Assign a cluster label to each new sample.

        Parameters
        ----------
        X : numpy.ndarray
            New data of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Cluster label of each sample, of shape (n_samples,).
        """


@runtime_checkable
class SupportsPredictProba(SupportsParams, Protocol):
    """An estimator that can predict probabilities.

    .. versionadded:: 0.6.0

    Only some classifiers can, for example the decision tree, the random forest
    and the logistic models, so the base classes do not require it.
    """

    def predict_proba(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Predict the probabilities for each sample.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted probabilities, with one entry or row per sample. The shape
            depends on the estimator: a decision tree returns one column per
            class, while a logistic model returns only the probability of the
            positive class.
        """
