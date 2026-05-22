from typing import Optional, Self


class Node:

    def __init__(
            self,
            feature: Optional[int] = None,
            threshold: Optional[float] | int = None,
            left: Optional[Self] = None,
            right: Optional[Self] = None,
            value: Optional[int] = None
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
