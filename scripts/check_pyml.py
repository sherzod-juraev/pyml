"""Run all quality checks against the pyml package in sequence.

Order: ruff --fix (auto-fixable lint issues) -> ruff format (code
style) -> ruff check (remaining lint issues) -> mypy (type checking)
-> interrogate (docstring coverage). Runs everything to completion
and reports a summary at the end, rather than stopping at the first
failure.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._runner import run_steps  # noqa: E402

TARGET = str(PROJECT_ROOT / "pyml")


STEPS = [
    ("ruff check --fix", ["ruff", "check", TARGET, "--fix"]),
    ("ruff format", ["ruff", "format", TARGET]),
    ("ruff check", ["ruff", "check", TARGET]),
    ("mypy", ["mypy", TARGET]),
    ("interrogate", ["interrogate", TARGET]),
]


def main() -> int:
    """Execute quality check steps and summarize results.

    This function sequentially executes code linters, formatters, type
    checkers, and docstring coverage tools against the pyml package.
    It captures exit codes to build a final compliance summary.

    Returns
    -------
    int
        0 if all steps passed successfully, 1 if any validation failed.
    """
    return run_steps(STEPS)


if __name__ == "__main__":
    sys.exit(main())
