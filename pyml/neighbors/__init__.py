"""Public API for pyml's neighbor-based models.

Lazily exposes distance-based regression and classification models:
KNNClassifier/KNNRegressor (fixed number of nearest neighbors) and
RadiusNeighborsClassifier/RadiusNeighborsRegressor (all neighbors
within a fixed radius), so that importing pyml.neighbors does not
eagerly load their contents.
"""

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "KNNClassifier",
    "KNNRegressor",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
]

# Mapping: symbol → (module, name)
_LAZY_IMPORTS = {
    "KNNClassifier": ("knn_classifier", "KNNClassifier"),
    "KNNRegressor": ("knn_regressor", "KNNRegressor"),
    "RadiusNeighborsClassifier": ("radius_classifier", "RadiusNeighborsClassifier"),
    "RadiusNeighborsRegressor": ("radius_regressor", "RadiusNeighborsRegressor"),
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
