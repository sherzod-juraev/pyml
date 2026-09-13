"""Public API for pyml's clustering algorithms.

Lazily exposes unsupervised clustering models (KMeans, DBSCAN), so
that importing pyml.cluster does not eagerly load their contents.
"""

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "DBSCAN",
    "KMeans",
]


# Mapping: symbol → (module, name)
_LAZY_IMPORTS = {
    "KMeans": ("kmeans", "KMeans"),
    "DBSCAN": ("dbscan", "DBSCAN"),
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
