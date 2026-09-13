"""Shared step-runner used by all check_*.py scripts."""

import subprocess
import sys
from pathlib import Path

import click


def require_source_checkout(path: Path, label: str) -> None:
    """Exit with a clear error if a required source-only path is missing.

    Some checks (docs, tests) only make sense when run from a git
    checkout of the project, since ``tests/`` and ``docs/`` are
    excluded from the installed package. This gives a clear error
    instead of a confusing failure when run after ``pip install``.

    Parameters
    ----------
    path : Path
        The directory that must exist for the check to run.
    label : str
        Human-readable name of the missing directory, used in the
        error message.
    """
    if not path.is_dir():
        click.echo(
            click.style("error: ", fg="red", bold=True)
            + f"'{label}' not found at {path}.\n"
            + "This command only works from a source checkout of the "
            + "repository, not an installed package.",
            err=True,
        )
        sys.exit(1)


def run_steps(steps: list[tuple[str, list[str]]]) -> int:
    """Execute a sequence of check commands and print a summary.

    Each step's own stdout/stderr is suppressed — only the PASS/FAIL
    summary is shown. Re-run the specific underlying command directly
    (e.g. ``ruff check pyml``) to see a failing step's full output.

    Parameters
    ----------
    steps : list of (str, list of str)
        Pairs of a human-readable step name and the command to run.

    Returns
    -------
    int
        0 if all steps passed, 1 if any failed.
    """
    results: list[tuple[str, bool]] = []
    for name, command in steps:
        click.echo(f"Running: {name}...")
        result = subprocess.run(command, capture_output=True)
        results.append((name, result.returncode == 0))

    summary = click.style("Summary", fg="blue", bold=True)
    click.echo(f"\n=== {summary} ===")
    all_passed = True
    for name, passed in results:
        if passed:
            status = click.style("PASS", fg="green", bold=True)
        else:
            status = click.style("FAIL", fg="red", bold=True)
            all_passed = False
        click.echo(f"{status}: {name}")

    return 0 if all_passed else 1
