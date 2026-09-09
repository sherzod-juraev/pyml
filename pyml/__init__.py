"""pyml: a from-scratch machine learning library built on NumPy and SciPy.

Lazily exposes pyml's subpackages (core, linear_model, metrics,
neighbors, preprocessing, model_selection, cluster), so that
importing pyml does not eagerly load their contents. Each subpackage
is imported directly, e.g. ``from pyml.linear_model import Ridge`` —
pyml itself never re-exports individual classes or functions.
"""

__version__ = "0.2.0"

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "cluster",
    "core",
    "linear_model",
    "metrics",
    "model_selection",
    "neighbors",
    "preprocessing",
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
