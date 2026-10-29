"""Sphinx extension: hide internal base classes from autodoc "Bases:" lines.

Autodoc lists every direct base class of a documented class. For pyml that
would expose implementation details (``_LinearModelBase``, ``BaseEstimator``,
``DataValidatorMixin``) that users never need to see. This extension hooks
into autodoc's ``autodoc-process-bases`` event and replaces each hidden base
with its own immediate parents (one level only).

A base is hidden if either:

* its name starts with an underscore (``_LinearModelBase``), or
* its name is listed in the ``pyml_hidden_bases`` config value in ``conf.py``
  (public-named classes that are internal details).

Usage in ``conf.py``::

    sys.path.append(str(Path(__file__).parent / "_ext"))
    extensions = [..., "pyml_hide_bases"]
    pyml_hidden_bases = {"BaseEstimator", "DataValidatorMixin"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata


def should_hide_base(base: type, hidden_names: frozenset[str]) -> bool:
    """Return True if ``base`` should be hidden from autodoc output.

    Parameters
    ----------
    base : type
        A direct base class of the class being documented.
    hidden_names : frozenset of str
        Public class names that are internal implementation details
        (from the ``pyml_hidden_bases`` config value).
    """
    name = getattr(base, "__name__", "")
    return name.startswith("_") or name in hidden_names


def unwrap_hidden_bases(bases: list[type], hidden_names: frozenset[str]) -> list[type]:
    """Replace each hidden base with its immediate parents (1-level unwrapping).

    ``object`` is never added, so a hidden base that only inherits from
    ``object`` simply disappears.
    """
    new_bases: list[type] = []
    for base in bases:
        if should_hide_base(base, hidden_names):
            new_bases.extend(
                parent for parent in getattr(base, "__bases__", ()) if parent is not object
            )
        else:
            new_bases.append(base)
    return new_bases


def _on_process_bases(
    app: Sphinx,
    name: str,
    obj: Any,
    options: Any,
    bases: list[type],
) -> None:
    """Rewrite the bases list for autodoc's ``autodoc-process-bases`` event.

    Sphinx expects ``bases`` to be changed in place; the return value is
    ignored.
    """
    hidden_names = frozenset(app.config.pyml_hidden_bases)
    bases[:] = unwrap_hidden_bases(bases, hidden_names)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the config value and connect the autodoc hook."""
    app.setup_extension("sphinx.ext.autodoc")  # event must exist before connect
    app.add_config_value(
        "pyml_hidden_bases",
        frozenset(),
        "env",
        types=frozenset({set, frozenset, list, tuple}),
    )
    # Priority below the default 500: sphinx_autodoc_typehints also listens to this
    # event and replaces the class objects in ``bases`` with rendered strings. We
    # must see the classes, so we run first (lower number = earlier).
    app.connect("autodoc-process-bases", _on_process_bases, priority=400)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
