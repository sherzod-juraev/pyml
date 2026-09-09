"""Min-max scaling to a fixed [0, 1] range."""

import numpy as np

from ..core.base import Transformer
from ..core.dtypes import FeatureMatrix


class MinMaxScaler(Transformer):
    r"""Scale features to a fixed [0, 1] range.

    Transforms each feature independently:

    .. math::
        z = \frac{x - x_{min}}{x_{max} - x_{min}}

    where :math:`x_{min}` and :math:`x_{max}` are the minimum and
    maximum of that feature, computed from the training data.

    Attributes
    ----------
    min_ : FeatureMatrix
        Per-feature minimum computed during fit, of shape (n_features,).
    range_ : FeatureMatrix
        Per-feature range (max - min) computed during fit, of shape
        (n_features,).


    .. note::
        A small constant is added to the range before dividing, so a
        feature with exactly zero range (a constant column) does not
        raise a division-by-zero error.

    .. note::
        Unlike StandardScaler, MinMaxScaler is sensitive to outliers: a
        single extreme value stretches the range and compresses every
        other value toward one end of [0, 1].

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.preprocessing import MinMaxScaler

        rng = np.random.default_rng(42)
        X = rng.normal(loc=50, scale=10, size=(200, 1))

        scaler = MinMaxScaler().fit(X)
        X_scaled = scaler.transform(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.hist(X, bins=20)
        ax1.set_title("Original")
        ax2.hist(X_scaled, bins=20)
        ax2.set_title("Min-max scaled")
    """

    def __init__(self) -> None:
        """Initialize this transformer with no configurable hyperparameters."""
        super().__init__()

    def _fit(self, X: FeatureMatrix, /) -> None:
        """Compute the per-feature minimum and range.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        """
        self.min_ = np.min(X, axis=0)
        self.range_ = np.max(X, axis=0) - self.min_

    def _transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Scale the given data to [0, 1] using the min and range from fit.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Scaled data, of the same shape as X.
        """
        return (X - self.min_) / (self.range_ + 1e-10)

    def _inverse_transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Reverse min-max scaling, mapping data back to its original scale.

        Parameters
        ----------
        X : FeatureMatrix
            Scaled data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Data mapped back to its original scale, of the same shape as X.
        """
        return X * (self.range_ + 1e-10) + self.min_
