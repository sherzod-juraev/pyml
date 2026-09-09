"""Robust scaling using the median and interquartile range."""

import numpy as np

from ..core.base import Transformer
from ..core.dtypes import FeatureMatrix


class RobustScaler(Transformer):
    r"""Scale features using statistics that are robust to outliers.

    Transforms each feature independently:

    .. math::
        z = \frac{x - Q_2}{Q_3 - Q_1}

    where :math:`Q_2` is the median and :math:`Q_3 - Q_1` is the
    interquartile range (IQR), both computed from the training data.

    Attributes
    ----------
    q2_ : FeatureMatrix
        Per-feature median computed during fit, of shape (n_features,).
    iqr_ : FeatureMatrix
        Per-feature interquartile range computed during fit, of shape
        (n_features,).


    .. note::
        A small constant is added to the IQR before dividing, so a
        feature with exactly zero IQR (e.g. more than half its values
        identical) does not raise a division-by-zero error.

    .. note::
        Unlike StandardScaler and MinMaxScaler, the median and IQR are
        barely affected by extreme values, since both depend only on
        the middle portion of the data's distribution. This makes
        RobustScaler a better choice when your data contains outliers.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.preprocessing import RobustScaler

        rng = np.random.default_rng(42)
        X = rng.normal(loc=50, scale=10, size=(200, 1))
        X[0] = 500.0  # a single extreme outlier

        scaler = RobustScaler().fit(X)
        X_scaled = scaler.transform(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
        ax1.hist(X, bins=20)
        ax1.set_title("Original")
        ax2.hist(X_scaled, bins=20)
        ax2.set_title("Robust scaled")
        fig.tight_layout()
    """

    def __init__(self) -> None:
        """Initialize this transformer with no configurable hyperparameters."""
        super().__init__()

    def _fit(self, X: FeatureMatrix, /) -> None:
        """Compute the per-feature median and interquartile range.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        """
        self.q2_ = np.median(X, axis=0)
        self.iqr_ = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)

    def _transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Scale the given data using the median and IQR from fit.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Scaled data, of the same shape as X.
        """
        return (X - self.q2_) / (self.iqr_ + 1e-10)

    def _inverse_transform(self, X: FeatureMatrix, /) -> FeatureMatrix:
        """Reverse robust scaling, mapping data back to its original scale.

        Parameters
        ----------
        X : FeatureMatrix
            Scaled data of shape (n_samples, n_features).

        Returns
        -------
        FeatureMatrix
            Data mapped back to its original scale, of the same shape as X.
        """
        return X * (self.iqr_ + 1e-10) + self.q2_
