"""Shared execution helpers for pyml's CLI commands."""

import subprocess
import sys
from pathlib import Path

import click


def require_source_checkout(path: Path, label: str) -> None:
    """Exit with a clear error if a required source-only path is missing.

    Some commands (docs, tests) only make sense when run from a git
    checkout of the project, since ``tests/`` and ``docs/`` are
    excluded from the installed package. This gives a clear error
    instead of a confusing failure when run after ``pip install``.

    Parameters
    ----------
    path : Path
        The directory that must exist for the command to run.
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


def run_checks(steps: list[tuple[str, list[str]]]) -> int:
    """Execute a sequence of check commands and print a PASS/FAIL summary.

    Each step's own stdout/stderr is suppressed — only the summary is
    shown. Re-run the specific underlying command directly (e.g.
    ``ruff check pyml``) to see a failing step's full output. A step
    whose command is not found on PATH is reported as FAIL with a
    clear reason, rather than raising an unhandled exception.

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
    missing: dict[str, str] = {}
    for name, command in steps:
        click.echo(f"Running: {name}...")
        try:
            result = subprocess.run(command, capture_output=True)
            results.append((name, result.returncode == 0))
        except FileNotFoundError:
            results.append((name, False))
            missing[name] = command[0]

    summary = click.style("Summary", fg="blue", bold=True)
    click.echo(f"\n=== {summary} ===")
    all_passed = True
    for name, passed in results:
        if passed:
            status = click.style("PASS", fg="green", bold=True)
            click.echo(f"{status}: {name}")
        else:
            status = click.style("FAIL", fg="red", bold=True)
            all_passed = False
            if name in missing:
                click.echo(
                    f"{status}: {name} "
                    f"(command '{missing[name]}' not found — "
                    f"install the dev/docs extras: pip install -e '.[dev,docs]')"
                )
            else:
                click.echo(f"{status}: {name}")

    return 0 if all_passed else 1


def run_live(command: list[str]) -> int:
    """Run a long-lived command with its output streamed live.

    Used for commands like sphinx-autobuild whose own progress output
    the user needs to see in real time. Unlike `run_checks`, nothing
    is captured or summarized. Polls the process with a short timeout
    (rather than blocking indefinitely) so Ctrl+C is caught promptly
    even on Windows, where an unbounded wait() call can otherwise
    delay KeyboardInterrupt until the child exits on its own.

    Parameters
    ----------
    command : list of str
        The command and its arguments to run.

    Returns
    -------
    int
        The subprocess's exit code. 0 if interrupted with Ctrl+C.
    """
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

    try:
        process = subprocess.Popen(command, creationflags=creationflags)
    except FileNotFoundError:
        click.echo(
            click.style("error: ", fg="red", bold=True)
            + f"command '{command[0]}' not found — "
            + "install the docs extras: pip install -e '.[docs]'",
            err=True,
        )
        return 1

    try:
        while True:
            try:
                return process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        _stop_process_tree(process)
        return 0


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Force-stop a subprocess and all of its descendants.

    Parameters
    ----------
    process : subprocess.Popen
        The process to stop.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
        )
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()
