"""Abstract base class for Naive Bayes classifiers.

Centralizes the shared Bayes' theorem machinery (class priors,
log-space posterior computation, and MAP prediction) for Naive Bayes
models. Subclasses supply their own per-class parameter estimation
and likelihood computation, allowing different feature-distribution
assumptions (Gaussian, multinomial, Bernoulli) to reuse the same
prediction logic.
"""

from abc import abstractmethod
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ..core.base import Classifier
from ..core.dtypes import ClassificationTarget, FeatureMatrix

_FloatArray = npt.NDArray[np.floating[Any]]


class _NaiveBayesBase(Classifier):
    r"""Abstract base class for Naive Bayes classifiers.

    All Naive Bayes models share the same prediction rule, derived
    from Bayes' theorem under the conditional independence ("naive")
    assumption between features:

    .. math::
        P(c \mid X) \propto P(c) \prod_{j=1}^{n} P(x_j \mid c)

    Computed in log-space to avoid numerical underflow from
    multiplying many small probabilities:

    .. math::
        \log P(c \mid X) \propto \log P(c) + \sum_{j=1}^{n} \log P(x_j \mid c)

    Prediction selects the class with the highest log-posterior:

    .. math::
        \hat{y} = \arg\max_{c} \left[ \log P(c) + \log P(X \mid c) \right]

    Subclasses differ only in how :math:`P(x_j \mid c)` is estimated
    and computed, which lets them assume different feature
    distributions without duplicating the prior/posterior/prediction
    logic.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during `fit`.
    class_log_prior_ : ndarray of shape (n_classes,)
        Log prior probability of each class in `classes_`.


    .. note::
        Subclasses must implement `_fit_class_params` and
        `_log_likelihood` to define how the per-class likelihood is
        estimated and computed.
    """

    def _prior(self, y: ClassificationTarget, /) -> _FloatArray:
        r"""Compute the log prior probability of each class.

        .. math::
            \log P(c) = \log \left( \frac{n_c}{n} \right)

        where :math:`n_c` is the number of training samples belonging
        to class :math:`c` and :math:`n` is the total number of
        samples.

        Parameters
        ----------
        y : ClassificationTarget
            Class labels.

        Returns
        -------
        FloatArray
            Log prior probability for each unique class in `y`,
            ordered to match `numpy.unique(y)`.
        """
        _, counts = np.unique(y, return_counts=True)
        n = y.shape[0]
        class_prior = counts / n
        return cast(_FloatArray, np.log(class_prior))

    @abstractmethod
    def _fit_class_params(self, X_c: FeatureMatrix, /) -> Any:
        """Estimate per-class parameters needed to compute the likelihood.

        Subclasses implement this to compute whatever statistics their
        assumed feature distribution requires (e.g. mean and variance for
        a Gaussian assumption, feature counts for a multinomial one).

        Parameters
        ----------
        X_c : ndarray of shape (n_samples_c, n_features)
            Training samples belonging to a single class.

        Returns
        -------
        Any
            Distribution-specific parameters for this class, later consumed
            by `_log_likelihood`.
        """

    @abstractmethod
    def _log_likelihood(self, X: FeatureMatrix, /) -> _FloatArray:
        r"""Compute the log-likelihood of each sample under each class.

        Subclasses implement this using the parameters stored in
        `_class_params` during `_fit`, according to their assumed feature
        distribution.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Log-likelihood :math:`\log P(X \mid c)` of each sample under
            each class, ordered to match `classes_`.
        """

    def _fit(self, X: FeatureMatrix, y: ClassificationTarget, /) -> None:
        """Fit the Naive Bayes model according to the training data.

        Computes class priors and delegates per-class parameter estimation
        to `_fit_class_params`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training samples.
        y : ndarray of shape (n_samples,)
            Target class labels.
        """
        self.classes_ = np.unique(y)
        self.class_log_prior_ = self._prior(y)
        self._class_params = [self._fit_class_params(X[y == c]) for c in self.classes_]

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        r"""Predict class labels via maximum a posteriori (MAP) estimation.

        .. math::
            \hat{y} = \arg\max_{c} \left[ \log P(c) + \log P(X \mid c) \right]

        Parameters
        ----------
        X : FeatureMatrix
            Samples to classify.

        Returns
        -------
        ClassificationTarget
            Predicted class label for each sample, of shape
            (n_samples,).
        """
        log_likelihood = self._log_likelihood(X)
        log_posterior = log_likelihood + self.class_log_prior_
        return self.classes_[np.argmax(log_posterior, axis=1)]
