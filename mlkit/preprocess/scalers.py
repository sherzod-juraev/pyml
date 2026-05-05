import numpy as np
from typing import Self
from ..exc import NotFitted


class MinMaxScaler:
    """
    Scale features to a fixed range [0, 1] using min-max normalization.

    Each feature is scaled independently based on the minimum and maximum
    values observed during fitting:

    .. math::

        X_{scaled} = \\frac{X - X_{min}}{X_{max} - X_{min}}

    Constant features (where :math:`X_{min} == X_{max}`) are handled gracefully by
    incrementing :math:`X_{max}` by 1, preserving the true minimum and
    resulting in a zero-filled column after transformation.

    Attributes
    ----------
    min_ : np.ndarray of shape (n_features,)
        Per-feature minimum values observed during fit.
    max_ : np.ndarray of shape (n_features,)
        Per-feature maximum values observed during fit.

    Notes
    -----
    Sensitive to outliers. If the dataset contains extreme values,
    consider using RobustScaler instead.

    Examples
    --------
    >>> scaler = MinMaxScaler()
    >>> X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
    >>> X_original = scaler.inverse_transform(X_scaled)
    """

    def __init__(self) -> None:
        self.min_ = None
        self.max_ = None
        self.__fitted = False

    def fit(self, X: np.ndarray) -> Self:
        """
        Compute per-feature minimum and maximum values from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to compute :math:`X_{min}` and :math:`X_{max}` per feature.

        Returns
        -------
        self : MinMaxScaler
            Fitted scaler instance.
        """

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.__fitted = True
        diff = self.max_ - self.min_
        ind = diff == 0
        if np.any(ind):
            self.max_[ind] += 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features of X using the fitted min and max values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to scale. Must have the same number of features as
            the data used during fit.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data with values in range [0, 1].

        Raises
        ------
        NotFitted
            If called before fitting the scaler.
        """

        if not self.__fitted:
            raise NotFitted(self, before='transform()')
        return (X - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling and return data to its original space.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Scaled data to inverse transform.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Data in the original feature space.

        Raises
        ------
        NotFitted
            If called before fitting the scaler.
        """

        if not self.__fitted:
            raise NotFitted(self, before='inverse_transform()')
        return X * (self.max_ - self.min_) + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data and return the scaled result in one step.

        Equivalent to calling fit(X).transform(X).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to fit and scale.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data with values in range [0, 1].
        """

        return self.fit(X).transform(X)


class StandardScaler:
    """
    Standardize features by removing the :math:`\\mu` and scaling to unit variance.

    Each feature is standardized independently using statistics computed
    during fitting:

    .. math::

        X_{scaled} = \\frac{X - \\mu}{\\sigma}

    Also known as Z-score normalization. The resulting distribution of each
    feature has :math:`\\mu` 0 and :math:`\\sigma` 1.

    Constant features (where :math:`\\sigma == 0``) are handled gracefully by
    setting std to 1 while preserving the true mean, resulting in a
    zero-filled column after transformation.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature :math:`\\mu` values computed during fit.
    std_ : np.ndarray of shape (n_features,)
        Per-feature standard deviations computed during fit.

    Notes
    -----
    Assumes data is approximately normally distributed. If the dataset
    contains significant outliers, consider using RobustScaler instead.

    Examples
    --------
    >>> scaler = StandardScaler()
    >>> X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
    >>> X_original = scaler.inverse_transform(X_scaled)
    """

    def __init__(self) -> None:

        self.mean_ = None
        self.std_ = None
        self.__fitted = False

    def fit(self, X: np.ndarray) -> Self:
        """
        Compute per-feature mean and standard deviation from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to compute mean and std per feature.

        Returns
        -------
        self : StandardScaler
            Fitted scaler instance.
        """

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        ind = self.std_ == 0
        if np.any(ind):
            self.std_[ind] = 1
        self.__fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features of X using the fitted mean and std.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to standardize. Must have the same number of features
            as the data used during fit.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Standardized data with mean 0 and standard deviation 1
            per feature.

        Raises
        ------
        NotFitted
            If called before fitting the scaler.
        """

        if not self.__fitted:
            raise NotFitted(self, before='transform()')
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the standardization and return data to its original space.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Standardized data to inverse transform.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Data in the original feature space.

        Raises
        ------
        NotFitted
            If called before fitting the scaler.
        """

        if not self.__fitted:
            raise NotFitted(self, before='inverse_transform()')
        return (X * self.std_) + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data and return the standardized result in one step.

        Equivalent to calling fit(X).transform(X).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to fit and standardize.

        Returns
        -------
        :math:`X_{scaled}` : np.ndarray of shape (n_samples, n_features)
            Standardized data with mean 0 and standard deviation 1
            per feature.
        """

        return self.fit(X).transform(X)


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers.

    Each feature is scaled independently using the median and
    interquartile range (IQR) computed during fitting:

    .. math::

        X_{scaled} = \\frac{X - median}{IQR}

    where IQR = Q3 - Q1 (75th percentile minus 25th percentile).

    Because the median and IQR are not influenced by extreme values,
    this scaler is significantly more robust to outliers than
    MinMaxScaler or StandardScaler.

    Constant features (where :math:`IQR == 0`) are handled gracefully by
    setting IQR to 1 while preserving the true median, resulting in a
    zero-filled column after transformation.

    Attributes
    ----------
    median_ : np.ndarray of shape (n_features,)
        Per-feature median values computed during fit.
    iqr_ : np.ndarray of shape (n_features,)
        Per-feature interquartile range (Q3 - Q1) computed during fit.

    Notes
    -----
    Does not scale data to a fixed range. The output range depends on
    the spread of the data within the IQR. Recommended when the dataset
    contains significant outliers.

    Examples
    --------
    >>> scaler = RobustScaler()
    >>> X_train = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 6.0]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
    >>> X_original = scaler.inverse_transform(X_scaled)
    """

    def __init__(self) -> None:

        self.median_ = None
        self.iqr_ = None
        self.__fitted = False

    def fit(self, X: np.ndarray) -> Self:
        """
        Compute per-feature median and IQR from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to compute median and IQR per feature.

        Returns
        -------
        self : RobustScaler
            Fitted scaler instance.
        """

        self.median_ = np.percentile(X, 50, axis=0)
        self.iqr_ = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        ind = self.iqr_ == 0
        if np.any(ind):
            self.iqr_[ind] = 1
        self.__fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features of X using the fitted median and IQR.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to scale. Must have the same number of features as
            the data used during fit.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data centered around 0 with unit IQR per feature.

        Raises
        ------
        NotFitted
            If called before fitting the scaler.
        """

        if not self.__fitted:
            raise NotFitted(self, before='transform()')
        return (X - self.median_) / self.iqr_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling and return data to its original space.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Scaled data to inverse transform.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Data in the original feature space.

        Raises
        ------
        NotFitted
            If called before fitting the scaler.
        """

        if not self.__fitted:
            raise NotFitted(self, before='inverse_transform()')
        return (X * self.iqr_) + self.median_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data and return the scaled result in one step.

        Equivalent to calling fit(X).transform(X).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to fit and scale.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data centered around 0 with unit IQR per feature.
        """

        return self.fit(X).transform(X)