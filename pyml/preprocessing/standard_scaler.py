"""Standardization by removing the mean and scaling to unit variance."""

import numpy as np

from ..core.base import Transformer
from ..core.dtypes import FeatureMatrix


class StandardScaler(Transformer):
    r"""Standardize features by removing the mean and scaling to unit variance.

    Transforms each feature independently:

    .. math::
        z = \frac{x - \mu}{\sigma}

    where :math:`\mu` and :math:`\sigma` are the mean and standard
    deviation of that feature, computed from the training data.

    Attributes
    ----------
    mean_ : FeatureMatrix
        Per-feature mean computed during fit, of shape (n_features,).
    std_ : FeatureMatrix
        Per-feature standard deviation computed during fit, of shape
        (n_features,).


    .. note::
        A small constant is added to the standard deviation before
        dividing, so a feature with exactly zero variance (a constant
        column) does not raise a division-by-zero error.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        X = rng.normal(loc=50, scale=10, size=(200, 1))

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.hist(X, bins=20)
        ax1.set_title("Original")
        ax2.hist(X_scaled, bins=20)
        ax2.set_title("Standardized")
    """

    def __init__(self) -> None:
        """Initialize this transformer with no configurable hyperparameters."""
        super().__init__()

    def _fit(self, X: FeatureMatrix, /) -> None:
        """Compute the per-feature mean and standard deviation.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def _transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Standardize the given data using the mean and std from fit.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Standardized data, of the same shape as X.
        """
        return (X - self.mean_) / (self.std_ + 1e-10)

    def _inverse_transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Reverse standardization, mapping data back to its original scale.

        Parameters
        ----------
        X : FeatureMatrix
            Standardized data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Data mapped back to its original scale, of the same shape as X.
        """
        return X * (self.std_ + 1e-10) + self.mean_
