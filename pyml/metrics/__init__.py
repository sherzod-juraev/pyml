"""Public API for pyml's evaluation metrics.

Lazily exposes regression metrics (mean_squared_error,
root_mean_squared_error, mean_absolute_error, r2_score) and
classification metrics (accuracy_score, precision_score,
recall_score, f1_score) so that importing pyml.metrics does not
eagerly load their contents.
"""

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "accuracy_score",
    "f1_score",
    "precision_score",
    "recall_score",
]

# Mapping: symbol → (module, name)
_LAZY_IMPORTS = {
    "mean_squared_error": ("regression", "mean_squared_error"),
    "mean_absolute_error": ("regression", "mean_absolute_error"),
    "root_mean_squared_error": ("regression", "root_mean_squared_error"),
    "r2_score": ("regression", "r2_score"),
    "accuracy_score": ("classification", "accuracy_score"),
    "f1_score": ("classification", "f1_score"),
    "precision_score": ("classification", "precision_score"),
    "recall_score": ("classification", "recall_score"),
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
