"""Public API for pyml's core infrastructure.

Lazily exposes the base, exceptions, and dtypes subpackages so that
importing pyml.core does not eagerly load their contents.
"""

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "base",
    "dtypes",
    "exceptions",
]


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
    # Check for submodule access
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)

    # Attribute not found
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. Available: {', '.join(sorted(__all__))}"
    )
