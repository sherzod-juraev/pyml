"""Sphinx extension: render every ``.. plot::`` block once per colour theme.

``matplotlib.sphinxext.plot_directive`` draws each figure once, with one set
of ``rcParams``. With a light/dark theme (``pydata_sphinx_theme``) one image
cannot look right in both modes. This extension replaces the ``plot``
directive with a thin wrapper that runs matplotlib's own ``run`` function
once per theme:

* the **light** pass uses ``pyml_plot_styles["light"]`` and its image gets
  the ``only-light`` CSS class,
* the **dark** pass uses ``pyml_plot_styles["dark"]`` and gets ``only-dark``,
  and shows neither source code nor source link (they are already shown by
  the light pass).

The theme's own CSS hides the image that does not match the visitor's mode.
Non-HTML builders (LaTeX, text, ...) get the light pass only, because
``only-light``/``only-dark`` are plain CSS and would otherwise print both
images.

Everything else (``:caption:``, ``:alt:``, ``plot_formats``, download links)
is matplotlib's own behaviour, unchanged. ``.. plot::`` blocks in the ``.rst``
files need no edits.

Configuration (``conf.py``), all optional::

    pyml_plot_styles = {
        "light": "_plot_styles/light.mplstyle",  # relative to conf.py
        "dark": "_plot_styles/dark.mplstyle",
    }

Colours live in the two ``.mplstyle`` files; sizes and fonts stay in the
regular ``plot_rcparams`` and are applied to both themes (a style file wins
on conflicts).

Limitations:

* The ``:context:`` option is not supported (each theme needs a fresh figure
  state); using it raises a clear error.
* Relies on matplotlib's ``plot_directive.run`` and on ``render_figures``
  reading ``config.plot_rcparams`` at call time. Verified on matplotlib 3.11.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
from matplotlib.sphinxext import plot_directive

if TYPE_CHECKING:
    from collections.abc import Iterator

    from docutils import nodes
    from sphinx.application import Sphinx
    from sphinx.config import Config
    from sphinx.util.typing import ExtensionMetadata

_THEMES = ("light", "dark")
_DEFAULT_STYLES = {
    "light": "_plot_styles/light.mplstyle",
    "dark": "_plot_styles/dark.mplstyle",
}
_DIGEST_FILE = ".pyml-plot-style-digest"


def _plot_cache_dir(app: Sphinx) -> Path:
    """Return the directory where matplotlib's directive caches its images.

    Mirrors ``plot_directive.run`` (``<doctreedir>/../plot_directive``).
    """
    return Path(app.doctreedir).parent / "plot_directive"


def _style_path(app: Sphinx, theme: str) -> Path:
    return Path(app.confdir, str(app.config.pyml_plot_styles[theme]))


def _load_style(app: Sphinx, theme: str) -> dict[str, Any]:
    """Read a ``.mplstyle`` file and return only the entries it sets."""
    rc = mpl.rc_params_from_file(
        _style_path(app, theme), fail_on_error=True, use_default_template=False
    )
    return dict(rc.items())


@contextmanager
def _theme_config(
    config: Config, theme_rc: dict[str, Any], *, show_formats: bool
) -> Iterator[None]:
    """Temporarily adapt the plot config for one theme pass.

    ``render_figures`` resets matplotlib to its defaults and then applies
    ``config.plot_rcparams`` before running each code block, so merging
    ``theme_rc`` into it for the duration of one directive call selects the
    theme. ``plot_html_show_formats`` has no per-directive option, so the
    dark pass switches it off here to avoid a second "(png, pdf)" link row.
    """
    original_rc = config.plot_rcparams
    original_show_formats = config.plot_html_show_formats
    config.plot_rcparams = {**original_rc, **theme_rc}
    config.plot_html_show_formats = original_show_formats and show_formats
    try:
        yield
    finally:
        config.plot_rcparams = original_rc
        config.plot_html_show_formats = original_show_formats


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


class ThemedPlotDirective(plot_directive.PlotDirective):
    """``.. plot::`` that renders the block once per colour theme."""

    def run(self) -> list[nodes.Node]:
        """Render the block once per theme and return the combined nodes."""
        env = self.state.document.settings.env
        app: Sphinx = env.app

        if "context" in self.options:
            raise self.error(
                "pyml_themed_plot: the ':context:' option of 'plot' is not "
                "supported (each theme needs its own figure state)."
            )

        is_html = app.builder.format == "html"
        themes = _THEMES if is_html else _THEMES[:1]

        # matplotlib's ``run`` does not return nodes: it pushes generated reST
        # into the parser with ``insert_input``, and each call lands in front
        # of the previous one. Running the dark pass first therefore leaves
        # the final order as: source, light image, dark image.
        result: list[nodes.Node] = []
        for theme in reversed(themes):
            options = copy.deepcopy(self.options)
            if is_html:
                options["class"] = [*options.get("class", []), f"only-{theme}"]
            if theme != "light":
                # Source and download links are shown once, by the light pass.
                options["include-source"] = False
                options["show-source-link"] = False
                # The dark pass must not overwrite the light pass's image files.
                prefix = options.get("filename-prefix")
                if prefix:
                    options["filename-prefix"] = f"{prefix}-{theme}"
                elif self.arguments:
                    script = Path(self.arguments[0]).stem
                    options["filename-prefix"] = f"{_slug(script)}-{_slug(env.docname)}-{theme}"

            with _theme_config(env.config, _load_style(app, theme), show_formats=theme == "light"):
                result += plot_directive.run(  # type: ignore[no-untyped-call]
                    self.arguments,
                    self.content,
                    options,
                    self.state_machine,
                    self.state,
                    self.lineno,
                )
        return result


def _on_config_inited(app: Sphinx, config: Config) -> None:
    """Make style edits take effect on the next build.

    Two separate caches would otherwise keep showing old images after a
    ``.mplstyle`` or ``plot_rcparams`` edit:

    * Sphinx re-reads only changed ``.rst`` files. A config value that holds
      a digest of the styles (``rebuild="env"``) makes it re-read all of them.
    * matplotlib reuses an image when it is newer than its ``.rst`` source.
      Deleting its cache directory forces redrawing.
    """
    payload = b"".join(_style_path(app, theme).read_bytes() for theme in _THEMES)
    payload += repr(sorted(config.plot_rcparams.items())).encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    config.pyml_plot_style_digest = digest

    cache = _plot_cache_dir(app)
    stamp = cache / _DIGEST_FILE
    if cache.is_dir() and (not stamp.is_file() or stamp.read_text() != digest):
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register config values, the digest handler and the wrapper directive."""
    # Must load first: we override its directive and read its config values.
    app.setup_extension("matplotlib.sphinxext.plot_directive")
    app.add_config_value("pyml_plot_styles", _DEFAULT_STYLES, "env", types=frozenset({dict}))
    app.add_config_value("pyml_plot_style_digest", "", "env")
    app.connect("config-inited", _on_config_inited)
    app.add_directive("plot", ThemedPlotDirective, override=True)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
