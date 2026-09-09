"""Public API for pyml's preprocessing transformers.

Lazily exposes feature-scaling transformers (StandardScaler,
MinMaxScaler, RobustScaler), so that importing pyml.preprocessing
does not eagerly load their contents.
"""

import importlib
from typing import Any

# Lazy-load all public API
__all__ = [
    "MinMaxScaler",
    "RobustScaler",
    "StandardScaler",
]

# Mapping: symbol → (module, name)
_LAZY_IMPORTS = {
    "MinMaxScaler": ("min_max_scaler", "MinMaxScaler"),
    "RobustScaler": ("robust_scaler", "RobustScaler"),
    "StandardScaler": ("standard_scaler", "StandardScaler"),
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
