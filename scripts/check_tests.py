"""Run all quality checks against the tests suite in sequence.

Order: ruff --fix -> ruff format -> ruff check -> mypy -> pytest.
Runs everything to completion and reports a summary at the end,
rather than stopping at the first failure.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._runner import run_steps  # noqa: E402

TARGET = str(PROJECT_ROOT / "tests")

STEPS = [
    ("ruff check --fix", ["ruff", "check", TARGET, "--fix"]),
    ("ruff format", ["ruff", "format", TARGET]),
    ("ruff check", ["ruff", "check", TARGET]),
    ("mypy", ["mypy", TARGET]),
    ("pytest", ["pytest", TARGET]),
]


def main() -> int:
    """Execute quality check steps and summarize results.

    This function iterates through all predefined development tools,
    runs them against the target tests directory, collects their exit codes,
    and prints a final pass/fail summary.

    Returns
    -------
    int
        0 if all steps passed, 1 if any step failed.
    """
    return run_steps(STEPS)


if __name__ == "__main__":
    sys.exit(main())
