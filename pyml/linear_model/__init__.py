"""Public API for pyml's linear and logistic regression models.

Lazily exposes gradient-descent-based linear models (LinearRegression,
Ridge, Lasso, ElasticNet) and logistic regression classifiers
(LogisticRegression, RidgeClassifier, LassoClassifier,
ElasticNetClassifier), so that importing pyml.linear_model does not
eagerly load their contents.
"""

import importlib
from typing import Any

__all__ = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "RidgeClassifier",
    "LassoClassifier",
    "ElasticNetClassifier",
]


# Mapping: symbol → (module, name)
_LAZY_IMPORTS = {
    "LinearRegression": ("linear_regression", "LinearRegression"),
    "Ridge": ("ridge", "Ridge"),
    "Lasso": ("lasso", "Lasso"),
    "ElasticNet": ("elastic_net", "ElasticNet"),
    "LogisticRegression": ("logistic_regression", "LogisticRegression"),
    "RidgeClassifier": ("ridge_classifier", "RidgeClassifier"),
    "LassoClassifier": ("lasso_classifier", "LassoClassifier"),
    "ElasticNetClassifier": ("elastic_net_classifier", "ElasticNetClassifier"),
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
