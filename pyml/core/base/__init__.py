"""Public base classes for building pyml estimators.

Exposes the abstract base classes that concrete models (linear
models, KNN, scalers, KMeans, DBSCAN, etc.) inherit from: Regressor,
Classifier, Transformer, Clusterer, and PredictableClusterer. Symbols
are lazy-loaded on first access to keep import time low.
"""

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "Classifier",
    "Clusterer",
    "PredictableClusterer",
    "Regressor",
    "Transformer",
]

# Mapping: symbol → (module, name)
_LAZY_IMPORTS = {
    "Classifier": ("classifier", "Classifier"),
    "Clusterer": ("clusterer", "Clusterer"),
    "PredictableClusterer": ("clusterer", "PredictableClusterer"),
    "Regressor": ("regressor", "Regressor"),
    "Transformer": ("transformer", "Transformer"),
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
