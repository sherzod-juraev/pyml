"""Run all quality checks against the documentation source files in sequence.

Order: sphinx-lint (style check) -> doc8 (format check) -> rstcheck (syntax check).
Runs all validation tools to completion and reports a unified summary at the
end, rather than stopping at the first failure.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._runner import run_steps  # noqa: E402

TARGET = str(PROJECT_ROOT / "docs")

STEPS = [
    (
        "sphinx-lint style check",
        [
            "sphinx-lint",
            f"{TARGET}/source",
        ],
    ),
    (
        "doc8 format check ",
        [
            "doc8",
            f"{TARGET}/source",
        ],
    ),
    ("rstcheck syntax check", ["rstcheck", "--recursive", f"{TARGET}/source"]),
    (
        "sphinx linkcheck",
        [
            "sphinx-build",
            "-b",
            "linkcheck",
            f"{TARGET}/source",
            f"{TARGET}/build/linkcheck",
        ],
    ),
    (
        "sphinx-build",
        [
            "sphinx-build",
            "-b",
            "html",
            f"{TARGET}/source",
            f"{TARGET}/build/html",
            "-W",
            "--keep-going",
        ],
    ),
]


def main() -> int:
    """Execute documentation quality checks and summarize results.

    This function sequentially runs reStructuredText linters and syntax
    checkers (sphinx-lint, doc8, rstcheck) against the documentation
    source directory, collects their exit codes, and prints a final report.

    Returns
    -------
    int
        0 if all documentation checks passed, 1 if any check failed.
    """
    return run_steps(STEPS)


if __name__ == "__main__":
    sys.exit(main())
