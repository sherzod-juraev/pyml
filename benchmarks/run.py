"""Command line entry point of the profiling tools.

Run it by the path of this file, from any folder::

    python benchmarks/run.py profile examples/tree/decision-tree-classifier.py

Do not run it with ``-m`` or ``-c``: those put the current folder first on ``sys.path``, so
from the repository root they could import the source tree instead of the installed ``pyml``.
A script path puts this folder first instead, which is why the files here must not reuse the
name of a standard library module (``profile.py`` would hide the module ``cProfile`` needs).
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from _profile import profile_script


def _positive_int(text: str) -> int:
    """Parse a positive integer for argparse."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    """Parse a positive number for argparse."""
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the parser of the command line.

    Returns
    -------
    argparse.ArgumentParser
        Parser with one sub-command, ``profile``.
    """
    parser = argparse.ArgumentParser(
        prog="python benchmarks/run.py", description="Profiling tools for pyml."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser(
        "profile",
        help="profile a script with pyinstrument and cProfile",
        description="Profile a script with pyinstrument, then with cProfile.",
    )
    profile.add_argument("path", type=Path, help="Python file to profile, such as an example")
    profile.add_argument(
        "--html",
        action="store_true",
        help="also write the pyinstrument report as HTML into benchmarks/results/",
    )
    profile.add_argument(
        "--top",
        type=_positive_int,
        default=20,
        help="rows in each cProfile table (default: 20)",
    )
    profile.add_argument(
        "--interval",
        type=_positive_float,
        default=0.001,
        help="seconds between two pyinstrument samples (default: 0.001)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command given on the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name. By default ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Exit status: 0 on success.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    path: Path = args.path.resolve()
    if not path.is_file():
        parser.error(f"not a file: {args.path}")
    profile_script(path, top=args.top, interval=args.interval, html=args.html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
