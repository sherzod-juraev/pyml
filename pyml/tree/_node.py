from typing import Self

from numpy import float64


class Node:
    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None | int = None,
        left: Self | None = None,
        right: Self | None = None,
        value: int | None | float64 = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
