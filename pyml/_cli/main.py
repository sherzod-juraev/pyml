"""Command-line interface for the pyml development workflow."""

import sys

import click
from click_help_colors import HelpColorsGroup

from pyml import __version__
from pyml._cli._runner import require_source_checkout, run_checks
from pyml._cli.commands.code import CHECK_STEPS as CODE_CHECK_STEPS
from pyml._cli.commands.code import code_group
from pyml._cli.commands.docs import CHECK_STEPS as DOCS_CHECK_STEPS
from pyml._cli.commands.docs import docs_group
from pyml._cli.commands.tests import CHECK_STEPS as TEST_CHECK_STEPS
from pyml._cli.commands.tests import tests_group
from pyml._cli.paths import DOCS_ROOT, TESTS_ROOT

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(
    context_settings=CONTEXT_SETTINGS,
    cls=HelpColorsGroup,
    help_headers_color="green",
    help_options_color="cyan",
)
@click.version_option(
    __version__, "-V", "--version", prog_name="pyml", message="%(prog)s %(version)s"
)
def cli() -> None:
    """Development CLI for the pyml machine learning library."""


cli.add_command(code_group)
cli.add_command(docs_group)
cli.add_command(tests_group)


@cli.command(name="check")
def check_all() -> None:
    """Run all quality checks (code, docs, tests) in sequence."""
    require_source_checkout(DOCS_ROOT, "docs/")
    require_source_checkout(TESTS_ROOT, "tests/")
    click.echo(click.style("\n>> Checking code\n", fg="cyan", bold=True))
    code_result = run_checks(CODE_CHECK_STEPS)
    click.echo(click.style("\n>> Checking docs\n", fg="cyan", bold=True))
    docs_result = run_checks(DOCS_CHECK_STEPS)
    click.echo(click.style("\n>> Checking tests\n", fg="cyan", bold=True))
    tests_result = run_checks(TEST_CHECK_STEPS)
    sys.exit(0 if code_result == 0 and docs_result == 0 and tests_result == 0 else 1)
