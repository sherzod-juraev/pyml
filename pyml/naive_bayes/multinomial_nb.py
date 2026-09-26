"""Multinomial Naive Bayes classifier."""

from typing import Any, cast

import numpy as np

from ..core.dtypes import FeatureMatrix
from ._base import _FloatArray, _NaiveBayesBase


class MultinomialNB(_NaiveBayesBase):
    r"""Multinomial Naive Bayes classifier.

    Assumes that features represent counts (e.g. word frequencies in
    a bag-of-words representation), and that, within each class
    :math:`c`, the probability of a feature :math:`x_j` is estimated
    as its relative frequency among that class's training samples:

    .. math::
        P(x_j \mid c) = \frac{\text{count}(x_j, c) + \alpha}
            {\text{total}(c) + \alpha \cdot n}

    where :math:`\text{count}(x_j, c)` is the total count of feature
    :math:`j` across all training samples of class :math:`c`,
    :math:`\text{total}(c)` is the total count of all features in
    class :math:`c`, :math:`n` is the number of features, and
    :math:`\alpha` (Laplace/additive smoothing) prevents a zero
    probability for a feature that never appeared in a class.

    Combined with the shared machinery in `_NaiveBayesBase`, the full
    prediction rule is:

    .. math::
        \hat{y} = \arg\max_{c} \left[ \log P(c) +
            \sum_{j=1}^{n} x_j \log P(x_j \mid c) \right]

    Unlike GaussianNB, the per-sample feature values :math:`x_j` act
    as weights on the log-probabilities (a feature occurring 3 times
    contributes 3 times its log-probability), so the log-likelihood
    reduces to a single matrix multiplication rather than an
    elementwise formula evaluated per sample.

    Parameters
    ----------
    alpha : float, optional
        Additive (Laplace) smoothing parameter. Larger values assign
        more probability mass to unseen feature/class combinations.
        ``alpha=0`` disables smoothing entirely, which can produce
        ``-inf`` log-probabilities for features never observed in a
        class.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during `fit`.
    class_log_prior_ : ndarray of shape (n_classes,)
        Log prior probability of each class in `classes_`.


    .. note::
        Fitting is a single closed-form pass over the data (counting
        feature occurrences per class) rather than iterative
        optimization — there is no `learning_rate`, `max_iter`, or
        convergence check.

    .. note::
        Designed for count data (e.g. word counts). Passing negative
        or non-count values as features doesn't raise an error, but
        the underlying probability model no longer has a meaningful
        interpretation.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.naive_bayes import MultinomialNB

        rng = np.random.default_rng(42)
        # class 0: word A frequent, word B rare
        X0 = rng.poisson(lam=[8, 1], size=(50, 2)).astype(np.float64)
        # class 1: word A rare, word B frequent
        X1 = rng.poisson(lam=[1, 8], size=(50, 2)).astype(np.float64)
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = MultinomialNB().fit(X, y)

        xx, yy = np.meshgrid(
            np.linspace(0, 15, 200),
            np.linspace(0, 15, 200),
        )
        grid_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, grid_pred.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
        ax.set_xlabel("Count of word A")
        ax.set_ylabel("Count of word B")
        ax.set_title("MultinomialNB decision boundary")
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize this classifier's smoothing hyperparameter.

        Parameters
        ----------
        alpha : float, optional
            Additive (Laplace) smoothing parameter; must be
            non-negative. Defaults to 1.0.
        """
        super().__init__()
        self.alpha: float = alpha

    def _fit_class_params(self, X_c: FeatureMatrix, /) -> Any:
        """Compute the smoothed log feature probabilities for one class.

        Parameters
        ----------
        X_c : FeatureMatrix
            Training samples belonging to a single class.

        Returns
        -------
        FloatArray
            Log probability of each feature within this class, of
            shape (n_features,).
        """
        feature_counts = np.sum(X_c, axis=0)
        total_count = np.sum(feature_counts)
        feature_prob = (feature_counts + self.alpha) / (total_count + self.alpha * X_c.shape[1])
        return np.log(feature_prob)

    def _log_likelihood(self, X: FeatureMatrix, /) -> _FloatArray:
        r"""Compute the log-likelihood of each sample under each class.

        .. math::
            \log P(X \mid c) = \sum_{j=1}^{n} x_j \log P(x_j \mid c)

        Parameters
        ----------
        X : FeatureMatrix
            Samples to evaluate (feature counts).

        Returns
        -------
        FloatArray
            Log-likelihood of each sample under each class, of shape
            (n_samples, n_classes).
        """
        class_params = cast(list[_FloatArray], self._class_params)
        feature_log_prob = np.array(class_params)
        log_likelihood = X @ feature_log_prob.T
        return log_likelihood
