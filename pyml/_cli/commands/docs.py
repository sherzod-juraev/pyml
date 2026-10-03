"""Documentation commands for pyml: check, build, and live-reload."""

import webbrowser

import click
from click_help_colors import HelpColorsGroup

from pyml._cli._runner import remove_tree, require_source_checkout, run_checks, run_live
from pyml._cli.paths import DOCS_BUILD, DOCS_BUILD_ROOT, DOCS_ROOT, DOCS_SOURCE

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
@click.pass_context
def check(ctx: click.Context) -> None:
    """Check documentation quality."""
    require_source_checkout(DOCS_ROOT, "docs/")
    ctx.exit(run_checks(CHECK_STEPS))


@docs_group.command(name="build")
@click.option(
    "--fresh",
    is_flag=True,
    help="Discard the cached environment and rewrite every output file (-E -a).",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the built HTML documentation in your default browser after success.",
)
@click.pass_context
def build(ctx: click.Context, fresh: bool, open_browser: bool) -> None:
    """Build the HTML documentation."""
    require_source_checkout(DOCS_ROOT, "docs/")
    command = ["sphinx-build", "-b", "html", _SOURCE, str(DOCS_BUILD)]
    if fresh:
        command += ["-E", "-a"]
    exit_code = run_live(command)
    if exit_code == 0 and open_browser:
        index_file = DOCS_BUILD / "index.html"
        if index_file.exists():
            opened = webbrowser.open(index_file.resolve().as_uri())
            if not opened:
                click.secho(
                    f"Warning: could not open a browser automatically. "
                    f"View the docs at: {index_file.resolve().as_uri()}",
                    fg="yellow",
                    err=True,
                )
        else:
            click.secho(
                f"Warning: index.html not found at {index_file}",
                fg="yellow",
                err=True,
            )
    ctx.exit(exit_code)


@docs_group.command(name="live")
@click.pass_context
def live(ctx: click.Context) -> None:
    """Serve the docs with live-reload (sphinx-autobuild)."""
    require_source_checkout(DOCS_ROOT, "docs/")
    command = ["sphinx-autobuild", _SOURCE, str(DOCS_BUILD)]
    ctx.exit(run_live(command))


@docs_group.command(name="clean")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without actually deleting anything.",
)
def clean(dry_run: bool) -> None:
    """Remove all generated documentation build artifacts.

    Deletes docs/build/ entirely — including HTML output, the
    linkcheck report, and Sphinx's doctree cache.
    """
    require_source_checkout(DOCS_ROOT, "docs/")
    if not remove_tree(DOCS_BUILD_ROOT, allowed_root=DOCS_ROOT, dry_run=dry_run):
        click.echo(f"Nothing to clean: {DOCS_BUILD_ROOT} does not exist.")
