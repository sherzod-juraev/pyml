"""Sphinx extension: ``.. dataset-info:: <name>`` describes a bundled dataset.

The directive asks ``pyml_datasets`` about the dataset while the docs are built and writes
a short table, the notes of the dataset, and a collapsible block with the citation and the
changes made to the data. Nothing about a dataset is typed by hand, so the page cannot go
out of date when ``pyml-datasets`` changes. If the name does not exist, the build fails
with the list of available names (the docs build runs with ``-W``).

Why the citation and the changes are always shown: the datasets are CC BY 4.0 (or CC0), and
CC BY 4.0 asks for attribution and for a statement of the changes made to the data.
``DatasetInfo`` keeps both, this directive prints them.

Usage, in an ``.rst`` file or in a text block of a sphinx-gallery example::

    .. dataset-info:: wine

The extension has no configuration.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from docutils import nodes
    from pyml_datasets import Dataset, DatasetInfo
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata

# Long feature lists (digits has 64 pixels) are shortened to this many names.
_MAX_FEATURES = 8
_URL = re.compile(r"https?://[^\s<>]+")
_SPECIAL = re.compile(r"([\\`*_|\[\]<>])")


def _escape(text: str) -> str:
    """Escape the characters that reStructuredText treats as inline markup."""
    return _SPECIAL.sub(r"\\\1", text)


def _with_links(text: str) -> str:
    """Escape ``text`` and turn the web addresses in it into links."""
    parts: list[str] = []
    position = 0
    for match in _URL.finditer(text):
        url = match.group().rstrip(".,;)")
        end = match.start() + len(url)
        parts.append(_escape(text[position : match.start()]))
        parts.append(f"`{url} <{url}>`__")
        position = end
    parts.append(_escape(text[position:]))
    return "".join(parts)


def _features(names: tuple[str, ...]) -> str:
    """List the feature names, shortened when there are many."""
    if len(names) <= _MAX_FEATURES:
        return ", ".join(_escape(name) for name in names)
    shown = ", ".join(_escape(name) for name in names[:_MAX_FEATURES])
    return f"{shown}, ... ({len(names)} in total)"


def _classes(data: Dataset) -> str | None:
    """Describe the classes with their sizes, or return ``None`` if not applicable."""
    if data.info.task != "classification" or data.y.ndim != 1:
        return None
    labels, counts = np.unique(data.y, return_counts=True)
    names = data.target_names
    parts = []
    for label, count in zip(labels, counts, strict=True):
        name = names[int(label)] if names is not None else str(label)
        parts.append(f"{_escape(name)} ({count})")
    return ", ".join(parts)


def _categorical(info: DatasetInfo) -> str:
    """Describe the categorical features and their categories."""
    if not info.categorical:
        return "none"
    return "; ".join(
        f"{_escape(feature)}: {', '.join(_escape(c) for c in categories)}"
        for feature, categories in info.categorical.items()
    )


def _rows(data: Dataset) -> list[tuple[str, str]]:
    """Return the (label, text) rows of the summary table."""
    info = data.info
    n_samples, n_features = data.X.shape
    rows = [
        ("Task", info.task.capitalize()),
        ("Samples × features", f"{n_samples} × {n_features}"),
        ("Features", _features(data.feature_names)),
    ]
    classes = _classes(data)
    if classes is not None:
        rows.append(("Classes", classes))
    rows += [
        ("Categorical features", _categorical(info)),
        ("Missing values", "yes, stored as NaN" if data.has_missing else "no"),
        ("License", f"`{_escape(info.license)} <{info.license_url}>`__"),
        ("Source", f"`{_escape(info.source)} <{info.source_url}>`__"),
    ]
    return rows


def _build(data: Dataset) -> list[str]:
    """Write the reStructuredText of the whole block, one string per line."""
    info = data.info
    lines = [_escape(info.summary), ""]
    lines += [".. list-table::", "   :widths: 28 72", "   :class: pyml-dataset-info", ""]
    for label, text in _rows(data):
        lines += [f"   * - **{label}**", f"     - {text}"]
    lines.append("")
    for note in info.notes:
        lines += [".. note::", "", f"   {_escape(note)}", ""]

    lines += [".. dropdown:: Citation, credits and changes made to the data", ""]
    lines += [f"   **Citation.** {_with_links(info.citation)}", ""]
    lines += [f"   **Created by** {_escape(', '.join(info.creators))}.", ""]
    if info.doi is not None:
        doi_url = f"https://doi.org/{info.doi}"
        lines += [f"   **DOI.** `{info.doi} <{doi_url}>`__", ""]
    lines += [f"   **Obtained from.** {_with_links(info.obtained_from)}", ""]
    lines += ["   **Changes made to the data.**", ""]
    lines += [f"   * {_with_links(change)}" for change in info.changes]
    lines += ["", f"   The license was last checked at the source on {info.verified_on}.", ""]
    return lines


class DatasetInfoDirective(SphinxDirective):
    """``.. dataset-info:: <name>``: summary table and credits of a bundled dataset."""

    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def run(self) -> list[nodes.Node]:
        """Load the dataset, write its description and parse it into nodes."""
        from pyml_datasets import load_dataset

        try:
            data = load_dataset(self.arguments[0])
        except ValueError as error:
            raise self.error(f"dataset-info: {error}") from error
        return self.parse_text_to_nodes("\n".join(_build(data)))


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the ``dataset-info`` directive."""
    # The directive writes sphinx-design's ``dropdown``.
    app.setup_extension("sphinx_design")
    app.add_directive("dataset-info", DatasetInfoDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
