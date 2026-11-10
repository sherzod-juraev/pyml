"""Sphinx extension: light and dark images for sphinx-gallery examples.

Plots of the ``.. plot::`` directive are drawn once per colour theme by
``pyml_themed_plot``. The examples of the gallery are plain scripts that are executed once, so
this extension does the same job differently:

* ``reset_matplotlib`` runs before every example and applies ``plot_rcparams`` and the light
  style, so the figures of an example are drawn in the light theme,
* ``themed_scraper`` saves every figure as it is (class ``only-light``) and then saves a copy
  with the dark colours (class ``only-dark``). Each group of images is put in a container
  with that class. The theme's CSS shows the image that matches the
  visitor's mode.

The colours come from the same two files as in ``pyml_themed_plot``
(``pyml_plot_styles``, ``_plot_styles/light.mplstyle`` and ``dark.mplstyle``), so the palette
is defined once. Examples need no special code.

How the dark copy is made: the figure is pickled before the light image is saved, and the
copy is recoloured with the ``text.color``, ``axes.edgecolor``, ``xtick.color``,
``grid.color`` and ``legend.edgecolor`` entries of the dark style. Colours of the data
(lines, points, bars, filled regions) are not changed.

Usage in ``conf.py``::

    sphinx_gallery_conf = {
        "image_scrapers": ("pyml_gallery_scraper.themed_scraper",),
        "reset_modules": ("matplotlib", "pyml_gallery_scraper.reset_matplotlib"),
    }

Limitations:

* A text with its own colour (``ax.text(..., color="red")``) is recoloured in the dark copy.
* Images for high-resolution screens (``image_srcset``) are not made for the dark copy.
* Builders other than HTML (LaTeX) get the light image only. The dark images are inside
  ``.. only:: html``, so this also holds when a LaTeX build reads pages that an HTML build
  wrote (sphinx-gallery does not write the page of an unchanged script again).
* A figure that cannot be pickled is not given a dark copy (the light image is used) and the
  build warns. Python 3.14 cannot pickle a figure with a categorical axis, such as
  ``ax.barh(["a", "b"], ...)``; draw such a chart at numeric positions and name the ticks with
  ``set_yticks(positions, labels)``.
"""

from __future__ import annotations

import pickle
from pathlib import Path, PurePosixPath
from textwrap import indent
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
from sphinx.util import logging

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.config import Config
    from sphinx.util.typing import ExtensionMetadata

_THEMES = ("light", "dark")

logger = logging.getLogger(__name__)

# The examples of a family are shown in this order, which is the order of the API reference
# (docs/source/api/pyml). A script that is not in the list comes after them, in alphabetical
# order. Add a new example here when it is added.
EXAMPLE_ORDER = (
    "linear-regression",
    "ridge",
    "lasso",
    "elastic-net",
    "logistic-regression",
    "ridge-classifier",
    "lasso-classifier",
    "elastic-net-classifier",
    "gaussian-nb",
    "multinomial-nb",
    "knn-classifier",
    "knn-regressor",
    "radius-neighbors-classifier",
    "radius-neighbors-regressor",
    "decision-tree-classifier",
    "decision-tree-regressor",
    "kmeans",
    "dbscan",
    "min-max-scaler",
    "robust-scaler",
    "standard-scaler",
)


class ExampleOrder:
    """Sort key of sphinx-gallery that puts the examples of a family in ``EXAMPLE_ORDER``.

    It is given to sphinx-gallery by its dotted name (``within_subsection_order`` in
    ``conf.py``), because a class object in the configuration cannot be cached by Sphinx.

    Parameters
    ----------
    src_dir : str
        The source directory of the family, not used.
    """

    def __init__(self, src_dir: str) -> None:
        self.src_dir = src_dir

    def __call__(self, filename: str) -> tuple[int, str]:
        """Return the position of an example in the list and its name."""
        stem = Path(filename).stem
        rank = EXAMPLE_ORDER.index(stem) if stem in EXAMPLE_ORDER else len(EXAMPLE_ORDER)
        return rank, stem


class _State:
    """Settings read from ``conf.py`` when the build starts."""

    rcparams: dict[str, Any] = {}  # noqa: RUF012
    styles: dict[str, dict[str, Any]] = {}  # noqa: RUF012


def _load_style(app: Sphinx, theme: str) -> dict[str, Any]:
    """Read a ``.mplstyle`` file and return only the entries it sets."""
    path = Path(app.confdir, str(app.config.pyml_plot_styles[theme]))
    rc = mpl.rc_params_from_file(path, fail_on_error=True, use_default_template=False)
    return dict(rc.items())


def _on_config_inited(app: Sphinx, config: Config) -> None:
    """Remember the plot settings, the scraper functions cannot reach ``app``."""
    _State.rcparams = dict(config.plot_rcparams)
    _State.styles = {theme: _load_style(app, theme) for theme in _THEMES}


def reset_matplotlib(gallery_conf: dict[str, Any], fname: str) -> None:
    """Apply ``plot_rcparams`` and the light style before an example runs.

    Parameters
    ----------
    gallery_conf : dict
        Configuration of sphinx-gallery, not used.
    fname : str
        Name of the example, not used.
    """
    mpl.rcParams.update({**_State.rcparams, **_State.styles["light"]})


def _recolor(fig: Any, rc: dict[str, Any]) -> None:
    """Give the text, the axes, the grid and the legend frame the colours of ``rc``."""
    text, edge = rc.get("text.color"), rc.get("axes.edgecolor")
    tick, grid = rc.get("xtick.color", edge), rc.get("grid.color")
    legend_edge = rc.get("legend.edgecolor")
    if text is not None:
        for artist in fig.findobj(mpl.text.Text):
            artist.set_color(text)
    for ax in fig.axes:
        if edge is not None:
            for spine in ax.spines.values():
                spine.set_edgecolor(edge)
        ax.tick_params(
            axis="both",
            which="both",
            **{
                key: value
                for key, value in (("color", tick), ("labelcolor", text))
                if value is not None
            },
        )
        if grid is not None:
            for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
                line.set_color(grid)
        legend = ax.get_legend()
        if legend is not None and legend_edge is not None:
            legend.get_frame().set_edgecolor(legend_edge)


def _wrap(rst: str, css_class: str) -> str:
    """Put the images of ``rst`` in a container with a CSS class.

    The ``image-sg`` directive of sphinx-gallery keeps only the first of its classes, so the
    class cannot be added to the image itself.
    """
    return f".. container:: {css_class}\n\n{indent(rst, '   ')}"


def themed_scraper(
    block: Any, block_vars: dict[str, Any], gallery_conf: dict[str, Any], **kwargs: Any
) -> str:
    """Save the figures of a code block in the light and in the dark colours.

    Parameters
    ----------
    block : sphinx_gallery.py_source_parser.Block
        The code block that was executed.
    block_vars : dict
        Variables of the block, it holds the iterator of the image paths.
    gallery_conf : dict
        Configuration of sphinx-gallery.
    **kwargs : dict
        Keyword arguments for ``Figure.savefig``.

    Returns
    -------
    str
        The reStructuredText with the light images and, for HTML, the dark images.
    """
    import matplotlib.pyplot as plt
    from sphinx_gallery.scrapers import figure_rst, matplotlib_scraper

    dark_rc = _State.styles.get("dark", {})
    wants_dark = gallery_conf.get("builder_name", "html") == "html" and bool(dark_rc)

    copies = []
    if wants_dark:
        for number in plt.get_fignums():
            try:
                copy = pickle.loads(pickle.dumps(plt.figure(number)))  # noqa: S301
            except Exception as error:  # the figure is shown in light colours in both modes
                # Python 3.14 cannot pickle a figure with a categorical axis (a bar chart with
                # text labels), so say it instead of leaving the dark mode with a light image.
                logger.warning(
                    "figure %d of %s has no dark copy, it cannot be pickled: %s. Use numbers "
                    "for the positions and set the labels with set_xticks/set_yticks.",
                    number,
                    block_vars.get("src_file", "an example"),
                    error,
                    type="pyml_gallery",
                    subtype="pickle",
                )
                continue
            # Unpickling registers the copy in pyplot, where sphinx-gallery would save it as
            # one more figure of the example. It can still be saved without pyplot.
            plt.close(copy)
            copies.append(copy)

    # sphinx-gallery's scraper saves the figures and closes them.
    light_rst = matplotlib_scraper(block, block_vars, gallery_conf, **kwargs)
    if not copies:
        return light_rst

    paths = []
    for fig in copies:
        path = PurePosixPath(next(block_vars["image_path_iterator"]))
        _recolor(fig, dark_rc)
        fig.savefig(path, **kwargs)
        paths.append(path)
    dark_rst = figure_rst(paths, gallery_conf["src_dir"])
    # The page is written once, but another builder may read it later (Read the Docs builds
    # HTML first and then the PDF from the same pages), so the dark images are marked HTML-only
    # in the page itself instead of relying on the builder that ran the example.
    dark = f".. only:: html\n\n{indent(_wrap(dark_rst, 'only-dark'), '   ')}"
    return f"{_wrap(light_rst, 'only-light')}\n\n{dark}\n"


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the handler that reads the plot settings."""
    # It reads the style files that this extension defines.
    app.setup_extension("pyml_themed_plot")
    app.connect("config-inited", _on_config_inited)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
