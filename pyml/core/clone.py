"""Copy an estimator without what it has learned from data."""

from copy import deepcopy

from .protocols import SupportsParams


def clone[P: SupportsParams](estimator: P) -> P:
    """Return a new, unfitted estimator with the same parameters.

    .. versionadded:: 0.7.0

    The new estimator is built by calling the class of ``estimator`` with the
    values from ``estimator.get_params()``. ``fit`` has not run on it, so the
    attributes that ``fit`` sets, such as ``coef_``, are absent, whether or not
    ``estimator`` was fitted. A meta-estimator is cloned as a whole: the
    estimators inside its parameters are cloned too, so the copy shares no
    estimator with the original.

    Parameters
    ----------
    estimator : SupportsParams
        The estimator to copy. It must be an instance, not a class.

    Returns
    -------
    SupportsParams
        A new estimator of the same class, with copied parameters.

    Raises
    ------
    TypeError
        If ``estimator`` is a class, or does not have ``get_params`` and
        ``set_params``, or if a parameter, at any depth, is an instance of a
        subclass of ``list``, ``tuple`` or ``dict``, such as a named tuple or
        an ``OrderedDict``.

    Notes
    -----
    A parameter value is copied by the first rule that fits:

    - an estimator is cloned with ``clone``;
    - a plain ``list``, ``tuple`` or ``dict`` is rebuilt, and its items are
      copied by the same rules;
    - any other value, such as a number, a string, an array or a ``set``, is
      copied with ``copy.deepcopy``.

    A subclass of ``list``, ``tuple`` or ``dict`` is rejected because it cannot
    be rebuilt safely, and a deep copy of it would also copy the fitted state
    of an estimator inside it. For the same reason, an estimator held in a
    ``set`` or ``frozenset`` is deep-copied together with its fitted state, so
    put estimators in a ``list``, ``tuple`` or ``dict``.

    The class of ``estimator`` must accept every key of ``get_params()`` as a
    keyword argument, which holds for every pyml estimator.
    """
    if isinstance(estimator, type):
        raise TypeError(
            f"clone expects an estimator instance, not the class {estimator.__name__}. "
            f"Pass {estimator.__name__}(...) instead."
        )
    if not isinstance(estimator, SupportsParams):
        raise TypeError(f"clone expects an estimator, not {type(estimator).__name__}.")

    params = estimator.get_params()
    # Pass each parameter value to the internal recursive helper.
    cloned_params = {k: _clone_value(v) for k, v in params.items()}

    # Return a new instance of the same class with cloned params.
    return type(estimator)(**cloned_params)


def _clone_value(value: object) -> object:
    """Copy one parameter value by the rules listed in the notes of ``clone``."""
    # 1. If the value is an estimator, call the public `clone`.
    if isinstance(value, SupportsParams) and not isinstance(value, type):
        return clone(value)

    # 2. If it's a container, recursively clone its elements.
    elif type(value) is dict:
        return {k: _clone_value(v) for k, v in value.items()}
    elif type(value) is list:
        return [_clone_value(v) for v in value]
    elif type(value) is tuple:
        return tuple(_clone_value(v) for v in value)
    elif isinstance(value, dict | list | tuple):
        # A subclass (namedtuple, OrderedDict, ...) cannot be rebuilt safely, and a deep copy
        # would copy the fitted state of an estimator inside it, so it is rejected.
        raise TypeError(
            f"clone supports only plain list, tuple and dict parameters, not "
            f"{type(value).__name__}. Use a plain list, tuple or dict instead."
        )

    # 3. For any other type (int, str, None, etc.), deepcopy is enough.
    return deepcopy(value)
