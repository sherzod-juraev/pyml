"""Code quality commands for the pyml package source."""

import sys

import click
from click_help_colors import HelpColorsGroup

from pyml._cli._runner import run_checks
from pyml._cli.paths import CODE_ROOT

_TARGET = str(CODE_ROOT)

CHECK_STEPS = [
    ("ruff check --fix", ["ruff", "check", _TARGET, "--fix"]),
    ("ruff format", ["ruff", "format", _TARGET]),
    ("ruff check", ["ruff", "check", _TARGET]),
    ("mypy", ["mypy", _TARGET]),
    ("interrogate", ["interrogate", _TARGET]),
]


@click.group(
    name="code",
    cls=HelpColorsGroup,
    help_headers_color="green",
    help_options_color="cyan",
)
def code_group() -> None:
    """Code quality commands (ruff, mypy, interrogate)."""


@code_group.command(name="check")
def check() -> None:
    """Check source code quality."""
    sys.exit(run_checks(CHECK_STEPS))
