"""Public exception classes for pyml.

Exposes the full exception hierarchy used throughout the library:

- :class:`PymlError` — root exception
- :class:`FittingError`, :class:`NotFittedError` 'NoNeighborsError' — fit/predict lifecycle errors
- :class:`ValidationError`, :class:`InvalidParameterError`, :class: `ShapeMismatchError` —
input/config errors
"""

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "PymlError",
    "FittingError",
    "NotFittedError",
    "NoNeighborsError",
    "ValidationError",
    "InvalidParameterError",
    "ShapeMismatchError",
]

# Mapping: symbol → (module, name)
_LAZY_IMPORTS = {
    "PymlError": ("base", "PymlError"),
    "FittingError": ("fitting", "FittingError"),
    "NotFittedError": ("fitting", "NotFittedError"),
    "NoNeighborsError": ("fitting", "NoNeighborsError"),
    "ValidationError": ("validation", "ValidationError"),
    "InvalidParameterError": ("validation", "InvalidParameterError"),
    "ShapeMismatchError": ("validation", "ShapeMismatchError"),
}


def __dir__() -> list[str]:
    """Return list of public attributes and submodules."""
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    """Lazy-load symbols and submodules on demand.

    Parameters
    ----------
    name : str
        The attribute name being accessed.

    Returns
    -------
    Any
        The requested module, class, or function.

    Raises
    ------
    AttributeError
        If the attribute does not exist.
    """
    # Check for lazy symbol import
    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, symbol_name)

    # Attribute not found
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. Available: {', '.join(sorted(__all__))}"
    )
