from mlkit.exc.base import MLKitException


class NotFitted(MLKitException):
    """
    Raised when an estimator is used before calling fit().

    Parameters
    -----------

    estimator : object
        The unfitted estimator instance
    before: str
        Default predict()
    """

    def __init__(
            self,
            estimator: object,
            before: str = 'predict()'
    ) -> None:
        name = type(estimator).__name__
        super().__init__(
            f"{name} is not fitted yet. Call fit() before using {before}."
        )