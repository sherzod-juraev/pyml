from typing import Optional, Self

from numpy import float64


class Node:

    def __init__(
            self,
            feature: Optional[int] = None,
            threshold: Optional[float] | int = None,
            left: Optional[Self] = None,
            right: Optional[Self] = None,
            value: Optional[int] | float64 = None
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
