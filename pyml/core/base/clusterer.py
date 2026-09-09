"""Abstract base classes for clustering estimators.

Defines the fit/fit_predict lifecycle shared by all clusterers,
including input validation and fit-state tracking. Concrete
clusterers implement _fit and _get_labels with their own algorithm.
PredictableClusterer additionally supports assigning cluster labels
to new, previously unseen samples via predict — not all clustering
algorithms support this (e.g. DBSCAN does not).
"""

from abc import ABC, abstractmethod
from typing import Self

from ..dtypes import ClusterLabels, FeatureMatrix
from .estimator import BaseEstimator
from .validation import DataValidatorMixin


class Clusterer(ABC, BaseEstimator, DataValidatorMixin):
    """Abstract base class for all pyml clusterers.

    Subclasses implement _fit (the clustering algorithm) and _get_labels
    (returning the cluster assignments computed during fit). The public
    fit/fit_predict methods handle input validation and fit-state
    tracking automatically.
    """

    @abstractmethod
    def _fit(self, X: FeatureMatrix, /) -> None:
        """Fit the clusterer to the given data.

        Subclasses implement the model-specific clustering logic here,
        typically storing per-sample cluster assignments for later retrieval
        via _get_labels. Input is already validated by the time this is
        called.

        Parameters
        ----------
        X : FeatureMatrix
            Data to cluster, of shape (n_samples, n_features).
        """

    def fit(self, X: FeatureMatrix, /) -> Self:
        """Validate input, fit the clusterer, and mark it as fitted.

        Parameters
        ----------
        X : FeatureMatrix
            Data to cluster, of shape (n_samples, n_features).

        Returns
        -------
        Self
            The fitted estimator instance.

        Raises
        ------
        ValueError
            If X fails shape/dtype/finiteness validation.
        TypeError
            If X has an invalid dtype.
        """
        self._validate_X(X)
        self._fit(X)
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _get_labels(self) -> ClusterLabels:
        """Return the cluster labels computed during fit.

        Subclasses implement this to expose whatever cluster assignments
        were computed and stored during _fit.

        Returns
        -------
        ClusterLabels
            Cluster index assigned to each sample seen during fit.
        """

    def fit_predict(self, X: FeatureMatrix, /) -> ClusterLabels:
        """Fit the clusterer to X and return the resulting cluster labels.

        Equivalent to calling fit(X) followed by retrieving the labels
        computed during fitting.

        Parameters
        ----------
        X : FeatureMatrix
            Data to cluster, of shape (n_samples, n_features).

        Returns
        -------
        ClusterLabels
            Cluster index assigned to each sample in X.
        """
        return self.fit(X)._get_labels()


class PredictableClusterer(Clusterer):
    """Base class for clusterers that can assign labels to new samples.

    Extends Clusterer with predict, for algorithms where a fitted model
    can meaningfully assign a cluster to previously unseen data (e.g.
    KMeans, via nearest centroid). Not suitable for algorithms without
    this notion (e.g. DBSCAN).
    """

    @abstractmethod
    def _predict(self, X: FeatureMatrix, /) -> ClusterLabels:
        """Assign a cluster label to new, previously unseen samples.

        Subclasses implement the model-specific assignment logic here.

        Parameters
        ----------
        X : FeatureMatrix
            New data of shape (n_samples, n_features).

        Returns
        -------
        ClusterLabels
            Cluster index assigned to each sample in X.
        """

    def predict(self, X: FeatureMatrix, /) -> ClusterLabels:
        """Validate input and assign cluster labels to new samples.

        Parameters
        ----------
        X : FeatureMatrix
            New data of shape (n_samples, n_features).

        Returns
        -------
        ClusterLabels
            Cluster index assigned to each sample in X.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If X fails shape/dtype/finiteness validation.
        """
        self._check_is_fitted()
        self._validate_X(X)
        return self._predict(X)
