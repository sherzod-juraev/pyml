"""Test suite commands for pyml."""

import sys

import click
from click_help_colors import HelpColorsGroup

from pyml._cli._runner import require_source_checkout, run_checks
from pyml._cli.paths import TESTS_ROOT

_TARGET = str(TESTS_ROOT)

CHECK_STEPS = [
    ("ruff check --fix", ["ruff", "check", _TARGET, "--fix"]),
    ("ruff format", ["ruff", "format", _TARGET]),
    ("ruff check", ["ruff", "check", _TARGET]),
    ("mypy", ["mypy", _TARGET]),
    ("pytest", ["pytest", _TARGET]),
]


@click.group(
    name="tests",
    cls=HelpColorsGroup,
    help_headers_color="green",
    help_options_color="cyan",
)
def tests_group() -> None:
    """Test suite commands (ruff, mypy, pytest)."""


@tests_group.command(name="check")
def check() -> None:
    """Check the test suite."""
    require_source_checkout(TESTS_ROOT, "tests/")
    sys.exit(run_checks(CHECK_STEPS))
