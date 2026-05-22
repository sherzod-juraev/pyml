from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from ..exc import NotFittedError


class BasicLinearModel(ABC):
    """Abstract base for all linear models."""
    def __init__(self) -> None:
        self._fitted = False

    def _check_fitted(self) -> None:
        """Raise if model not fitted."""
        if not self._fitted:
            raise NotFittedError(self)

    @abstractmethod
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[Any]) -> Self:
        """Fit the model"""

    @abstractmethod
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
        """Predict using the model"""
