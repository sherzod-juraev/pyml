"""Documentation commands for pyml: check, build, and live-reload."""

import sys

import click
from click_help_colors import HelpColorsGroup

from pyml._cli._runner import require_source_checkout, run_checks, run_live
from pyml._cli.paths import DOCS_BUILD, DOCS_ROOT, DOCS_SOURCE

_SOURCE = str(DOCS_SOURCE)
_LINKCHECK_BUILD = str(DOCS_ROOT / "build" / "linkcheck")

CHECK_STEPS = [
    ("sphinx-lint style check", ["sphinx-lint", _SOURCE]),
    ("doc8 format check", ["doc8", _SOURCE]),
    ("rstcheck syntax check", ["rstcheck", "--recursive", _SOURCE]),
    (
        "sphinx linkcheck",
        ["sphinx-build", "-b", "linkcheck", _SOURCE, _LINKCHECK_BUILD],
    ),
    (
        "sphinx-build",
        ["sphinx-build", "-b", "html", _SOURCE, str(DOCS_BUILD), "-W", "--keep-going"],
    ),
]


@click.group(
    name="docs",
    cls=HelpColorsGroup,
    help_headers_color="green",
    help_options_color="cyan",
)
def docs_group() -> None:
    """Documentation commands (check, build, live-reload)."""


@docs_group.command(name="check")
def check() -> None:
    """Check documentation quality."""
    require_source_checkout(DOCS_ROOT, "docs/")
    sys.exit(run_checks(CHECK_STEPS))


@docs_group.command(name="build")
@click.option(
    "--fresh",
    is_flag=True,
    help="Discard the cached environment and rewrite every output file (-E -a).",
)
def build(fresh: bool) -> None:
    """Build the HTML documentation."""
    require_source_checkout(DOCS_ROOT, "docs/")
    command = ["sphinx-build", "-b", "html", _SOURCE, str(DOCS_BUILD)]
    if fresh:
        command += ["-E", "-a"]
    sys.exit(run_live(command))


@docs_group.command(name="live")
def live() -> None:
    """Serve the docs with live-reload (sphinx-autobuild)."""
    require_source_checkout(DOCS_ROOT, "docs/")
    command = ["sphinx-autobuild", _SOURCE, str(DOCS_BUILD)]
    sys.exit(run_live(command))
