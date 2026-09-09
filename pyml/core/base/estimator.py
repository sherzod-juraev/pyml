"""Base estimator providing shared parameter management and fit-state tracking.

Supplies scikit-learn-style get_params/set_params via constructor
signature introspection, along with the is_fitted_ flag and a
_check_is_fitted helper used by subclasses to guard predict/transform
calls. Not part of the public API — subclasses (Regressor, Classifier,
Transformer, Clusterer) provide the actual fit/predict contracts.
"""

import inspect
from typing import Self

from ..exceptions import InvalidParameterError, NotFittedError


class BaseEstimator:
    """Internal base class providing parameter introspection and fit-state tracking.

    Not exposed as part of pyml's public API. Subclasses are responsible
    for calling super().__init__() and for implementing their own
    fit/_fit contract.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the estimator's fit method has been called.
    """

    def __init__(self) -> None:
        """Initialize the estimator with is_fitted_ set to False."""
        self.is_fitted_: bool = False

    def _check_is_fitted(self) -> None:
        """Raise NotFittedError if the estimator has not been fitted yet.

        Intended to be called at the start of predict/transform methods in
        subclasses, before accessing any attributes learned during fit.

        Raises
        ------
        NotFittedError
            If is_fitted_ is False.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def get_params(self) -> dict[str, object]:
        """Return the estimator's constructor parameters and values.

        Parameter names are discovered via introspection of __init__, so
        subclasses do not need to override this method as long as
        constructor arguments are stored as same-named attributes.

        Returns
        -------
        dict[str, object]
            Mapping of parameter name to its current value.

        Notes
        -----
        Accessing self.__init__ for signature introspection is flagged by
        mypy as unsound in the general case (a subclass could theoretically
        override __init__ with an incompatible signature). This is
        suppressed here because the method only inspects the signature and
        never calls it.
        """
        signature = inspect.signature(self.__init__)  # type: ignore[misc]
        params = {}
        for p in signature.parameters.values():
            if p.name != "self" and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL:
                params[p.name] = getattr(self, p.name)
        return params

    def set_params(self, **params: object) -> Self:
        """Set one or more constructor parameters on this estimator.

        Parameters
        ----------
        **params : object
            Parameter names and values to set. Each name must match an
            existing constructor parameter.

        Returns
        -------
        Self
            This estimator instance, for method chaining.

        Raises
        ------
        InvalidParameterError
            If a given parameter name does not match any of the estimator's
            constructor arguments.
        """
        if not params:
            return self
        accepted_params = self.get_params()
        for key, val in params.items():
            if key not in accepted_params:
                raise InvalidParameterError(
                    f"Invalid parameter {key} for {type(self).__name__}. "
                    f"Valid parameters are: {', '.join(accepted_params.keys())}"
                )
            setattr(self, key, val)
        return self
