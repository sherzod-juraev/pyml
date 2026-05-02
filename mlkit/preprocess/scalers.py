import numpy as np
from mlkit.exc import NotFitted


class MinMaxScaler:

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.__fitted = False

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        ind = self.min_ == self.max_
        if np.any(ind):
            self.min_[ind] = 0
            self.max_[ind] = 1
        self.__fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:

        if not self.__fitted:
            raise NotFitted(f"{type(self).__name__} not fitted yet")
        return (X - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:

        if not self.__fitted:
            raise NotFitted(f"{type(self).__name__} not fitted yet")
        return X * (self.max_ - self.min_) + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        return self.fit(X).transform(X)