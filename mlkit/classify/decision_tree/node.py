class Node:
    def __init__(
            self,
            feature: int | None = None,
            threshold: float | int | None = None,
            left = None,
            right = None,
            value: int | None = None
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value