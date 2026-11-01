"""Sphinx extension: the cards of an examples page, in the style of the other card pages.

sphinx-gallery writes one ``index.rst`` per family (``examples/<family>/index.rst``) with its own
thumbnail cards. This extension reads that page when Sphinx loads it (the ``source-read``
event) and replaces the cards with the ``sphinx-design`` cards that the rest of the site uses:
the thumbnail on top, the title, one sentence about the model and an "Open ..." button. The
sentence is the first one of the docstring of the example. The files and the URLs of the
examples do not change, only the index page.

The cards can also be put in groups, each with a heading and a sentence of its own. The groups
are set in ``conf.py``::

    pyml_gallery_groups = {
        "linear-model": [
            ("Regression", "Predict a number.", ["linear-regression", "ridge"]),
            ("Classification", "Predict a class.", ["logistic-regression"]),
        ],
    }

Each group is ``(heading, text, [script names without .py])``. A family that is not in the
dictionary gets one grid under the heading "Examples". The build warns when an example of a
grouped family is in no group, when a group names an example that does not exist, or when the
index page does not have the shape this extension expects (sphinx-gallery wrote it in another
format), so a page never breaks without notice. In the last case the page of sphinx-gallery is
left as it is.

The order of the cards inside a group follows the list. The order of the pages in the sidebar
is ``EXAMPLE_ORDER`` in ``pyml_gallery_scraper.py``, keep the two in agreement.
"""

from __future__ import annotations

import html
import re
from textwrap import indent
from typing import TYPE_CHECKING

from sphinx.util import logging

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata

logger = logging.getLogger(__name__)

_START = "sphx-glr-thumbnails"
_OPEN = ".. thumbnail-parent-div-open\n"
_CLOSE = ".. thumbnail-parent-div-close\n"
_WRAP = '.. raw:: html\n\n    <div class="sphx-glr-thumbnails">'
# One card: a raw block that opens the container, the image and the link, and the raw block with
# the title that closes the container.
_CARD = re.compile(
    r'\.\. raw:: html\n\n    <div class="sphx-glr-thumbcontainer".*?'
    r'<div class="sphx-glr-thumbnail-title">.*?</div>\n    </div>\n',
    re.DOTALL,
)
_TOOLTIP = re.compile(r'tooltip="(?P<text>.*?)">', re.DOTALL)
_TITLE = re.compile(r'<div class="sphx-glr-thumbnail-title">(?P<text>.*?)</div>', re.DOTALL)
_IMAGE = re.compile(r"\.\. image:: (?P<path>\S+)")
_LINK = re.compile(r":doc:`(?P<path>/examples/(?P<family>[^/`]+)/(?P<name>[^/`]+))`")
_SENTENCE = re.compile(r"^.*?[.!?](?=\s|$)", re.DOTALL)


def _escape(text: str) -> str:
    """Escape what reStructuredText would read as markup in a plain sentence."""
    return re.sub(r"([*`|\\]|_(?=\W|$))", r"\\\1", text)


def _card(link: re.Match[str], block: str) -> str | None:
    """Return the ``sphinx-design`` card for one thumbnail block of sphinx-gallery."""
    title, image, tooltip = _TITLE.search(block), _IMAGE.search(block), _TOOLTIP.search(block)
    if title is None or image is None or tooltip is None:
        return None
    name = html.unescape(title.group("text")).strip()
    text = " ".join(html.unescape(tooltip.group("text")).split())
    first = _SENTENCE.match(text)
    sentence = _escape(first.group(0) if first else text)
    return (
        ".. grid-item-card::\n"
        "    :text-align: center\n"
        "    :shadow: md\n"
        f"    :img-top: {image.group('path')}\n"
        "    :img-alt:\n\n"
        f"    {_escape(name)}\n"
        f"    {'^' * len(name)}\n"
        f"    {sentence}\n\n"
        "    +++\n"
        f"    .. button-ref:: {link.group('path')}\n"
        "        :ref-type: doc\n"
        "        :color: primary\n"
        "        :expand:\n\n"
        "        Open example\n"
    )


def _grid(cards: list[str]) -> str:
    """Return a grid of cards, two or three in a row, so that no row is left with one card."""
    columns = "1 2 2 3" if len(cards) % 3 == 0 else "1 2 2 2"
    items = "\n".join(indent(card, "    ") for card in cards)
    return (
        ".. only:: html\n\n"
        f"    .. grid:: {columns}\n"
        "        :gutter: 3\n"
        "        :class-container: sd-mb-4\n\n"
        f"{indent(items, '    ')}\n\n"
    )


def _regroup(source: str, family: str, groups: list[tuple[str, str, list[str]]]) -> str | None:
    """Return the index page of ``family`` with the new cards, or None if it cannot."""
    start, end = source.find(_OPEN), source.find(_CLOSE)
    first_wrap = source.rfind(_WRAP, 0, start)
    if start < 0 or end < start or first_wrap < 0:
        return None
    cards: dict[str, str] = {}
    for match in _CARD.finditer(source[start:end]):
        link = _LINK.search(match.group(0))
        card = _card(link, match.group(0)) if link is not None else None
        if link is None or card is None or link.group("family") != family:
            return None
        cards[link.group("name")] = card
    if not cards:
        return None

    # Everything before the grid (the title and the text of GALLERY_HEADER.rst) stays, and so
    # does everything after it (the hidden toctree).
    tail = source[end + len(_CLOSE) :].removeprefix("\n.. raw:: html\n\n    </div>\n")
    parts = [source[:first_wrap]]
    if not groups:
        parts.append(f"Examples\n--------\n\n{_grid(list(cards.values()))}")
    else:
        listed = {name for _, _, names in groups for name in names}
        for name in sorted(listed - cards.keys()):
            _warn("gallery group of %s names %r, which is not an example", family, name)
        for name in sorted(cards.keys() - listed):
            _warn("example %r of %s is in no group of pyml_gallery_groups", name, family)
        for heading, text, names in groups:
            found = [cards[name] for name in names if name in cards]
            if found:
                parts.append(f"{heading}\n{'-' * len(heading)}\n\n{text}\n\n{_grid(found)}")
        # Cards that no group claims are not lost, they follow the groups without a heading.
        rest = [card for name, card in cards.items() if name not in listed]
        if rest:
            parts.append(_grid(rest))
    parts.append(tail)
    return "".join(parts)


def _warn(message: str, *args: object) -> None:
    """Warn through Sphinx, so that Read the Docs fails on it."""
    logger.warning(message, *args, type="pyml_gallery", subtype="groups")


def _on_source_read(app: Sphinx, docname: str, source: list[str]) -> None:
    """Replace the cards of a generated family page, in groups when ``conf.py`` has them."""
    parts = docname.split("/")
    if len(parts) != 3 or parts[0] != "examples" or parts[2] != "index":
        return
    if _START not in source[0]:
        return
    groups = app.config.pyml_gallery_groups.get(parts[1], [])
    page = _regroup(source[0], parts[1], groups)
    if page is None:
        _warn(
            "cannot rebuild the cards of %s, the index page of sphinx-gallery has another shape",
            docname,
        )
        return
    source[0] = page


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the setting and the handler."""
    app.add_config_value("pyml_gallery_groups", {}, "env", types=[dict])
    app.connect("source-read", _on_source_read)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
