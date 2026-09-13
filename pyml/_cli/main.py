"""Command-line interface for the pyml development workflow.

Exposes the ``pyml`` command, which groups project quality checks
(code, documentation, and tests) under a single entry point. Each
check runs a sequence of external tools (ruff, mypy, interrogate,
sphinx-lint, doc8, rstcheck, pytest) and exits with a non-zero status
code if any step fails, so the command integrates cleanly with CI.
"""

import sys

import click

from pyml import __version__
from pyml._cli._runner import require_source_checkout, run_steps
from pyml._cli.checks import DOCS_ROOT, TESTS_ROOT
from pyml._cli.checks.code import CODE_STEPS
from pyml._cli.checks.docs import DOCS_STEPS
from pyml._cli.checks.tests import TEST_STEPS

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__, prog_name="pyml")
def cli() -> None:
    """Development CLI for the pyml machine learning library."""


@cli.group(invoke_without_command=True)
@click.pass_context
def check(ctx: click.Context) -> None:
    """Run project quality checks (code, docs, tests).

    Running this command without a subcommand executes all checks.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(check_all)


@check.command(name="code")
def check_code() -> None:
    """Check source code quality (ruff, mypy, interrogate)."""
    click.echo(click.style("\n>> Checking code\n", fg="cyan", bold=True))
    sys.exit(run_steps(CODE_STEPS))


@check.command(name="docs")
def check_docs() -> None:
    """Check documentation quality (sphinx-lint, doc8, rstcheck)."""
    require_source_checkout(DOCS_ROOT, "docs/")
    click.echo(click.style("\n>> Checking docs\n", fg="cyan", bold=True))
    sys.exit(run_steps(DOCS_STEPS))


@check.command(name="tests")
def check_tests() -> None:
    """Check the test suite (ruff, mypy, pytest)."""
    require_source_checkout(TESTS_ROOT, "tests/")
    click.echo(click.style("\n>> Checking tests\n", fg="cyan", bold=True))
    sys.exit(run_steps(TEST_STEPS))


@check.command(name="all")
def check_all() -> None:
    """Run all quality checks in sequence."""
    require_source_checkout(DOCS_ROOT, "docs/")
    require_source_checkout(TESTS_ROOT, "tests/")
    click.echo(click.style("\n>> Checking code\n", fg="cyan", bold=True))
    code_result = run_steps(CODE_STEPS)
    click.echo(click.style("\n>> Checking docs\n", fg="cyan", bold=True))
    docs_result = run_steps(DOCS_STEPS)
    click.echo(click.style("\n>> Checking tests\n", fg="cyan", bold=True))
    tests_result = run_steps(TEST_STEPS)
    sys.exit(0 if code_result == 0 and docs_result == 0 and tests_result == 0 else 1)
