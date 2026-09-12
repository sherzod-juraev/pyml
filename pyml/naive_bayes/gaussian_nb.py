"""Gaussian Naive Bayes classifier."""

from typing import Any, cast

import numpy as np

from ..core.dtypes import FeatureMatrix
from ._base import _FloatArray, _NaiveBayesBase


class GaussianNB(_NaiveBayesBase):
    r"""Gaussian Naive Bayes classifier.

    Assumes that, within each class :math:`c`, each feature
    :math:`x_j` is normally distributed:

    .. math::
        P(x_j \mid c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}}
            \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}\right)

    where :math:`\mu_{c,j}` and :math:`\sigma_{c,j}^2` are the mean
    and variance of feature :math:`j` among training samples of class
    :math:`c`. A small constant is added to the variance to avoid
    division by zero when a feature is (near-)constant within a
    class.

    Combined with the shared machinery in `_NaiveBayesBase`, the full
    prediction rule is:

    .. math::
        \hat{y} = \arg\max_{c} \left[ \log P(c) +
            \sum_{j=1}^{n} \log P(x_j \mid c) \right]

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during `fit`.
    class_log_prior_ : ndarray of shape (n_classes,)
        Log prior probability of each class in `classes_`.


    .. note::
        Unlike the linear models in this library, fitting is a single
        closed-form pass over the data (computing per-class mean and
        variance) rather than iterative optimization — there is no
        `learning_rate`, `max_iter`, or convergence check.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.naive_bayes import GaussianNB

        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(50, 2))
        X1 = rng.normal(loc=(2, 2), scale=1.0, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = GaussianNB().fit(X, y)

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
        )
        grid_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, grid_pred.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("GaussianNB decision boundary")
    """

    def _fit_class_params(self, X_c: FeatureMatrix, /) -> Any:
        """Compute the per-feature mean and variance for one class.

        Parameters
        ----------
        X_c : FeatureMatrix
            Training samples belonging to a single class.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            Mean and variance of each feature among `X_c`, each of
            shape (n_features,).
        """
        mean = np.mean(X_c, axis=0)
        variance = np.var(X_c, axis=0)
        return mean, variance

    def _log_likelihood(self, X: FeatureMatrix, /) -> _FloatArray:
        r"""Compute the Gaussian log-likelihood of each sample under each class.

        .. math::
            \log P(X \mid c) = \sum_{j=1}^{n} \left[
                -\tfrac{1}{2}\log(2\pi\sigma_{c,j}^2)
                - \frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right]

        Parameters
        ----------
        X : FeatureMatrix
            Samples to evaluate.

        Returns
        -------
        FloatArray
            Log-likelihood of each sample under each class, of shape
            (n_samples, n_classes).
        """
        class_params = cast(list[tuple[_FloatArray, _FloatArray]], self._class_params)
        class_params_arr = np.array(class_params)
        means = class_params_arr[:, 0, :]
        variances = class_params_arr[:, 1, :]
        X_expanded = X[:, np.newaxis, :]
        means_expanded = means[np.newaxis, :, :]
        variances_expanded = variances[np.newaxis, :, :]
        var_stable = variances_expanded + 1e-10
        term1 = -0.5 * np.log(2 * np.pi)
        term2 = -0.5 * np.log(var_stable)
        term3 = -((X_expanded - means_expanded) ** 2) / (2 * var_stable)
        log_likelihood = np.sum(term1 + term2 + term3, axis=2)
        return log_likelihood
