"""Test suite quality check steps.

Order: ruff check --fix (auto-fixable lint issues) -> ruff format (code
formatting) -> ruff check (remaining lint issues) -> mypy (static type
checking) -> pytest (test execution).
"""

from pyml._cli.checks import TESTS_ROOT

_TARGET = str(TESTS_ROOT)

TEST_STEPS = [
    ("ruff check --fix", ["ruff", "check", _TARGET, "--fix"]),
    ("ruff format", ["ruff", "format", _TARGET]),
    ("ruff check", ["ruff", "check", _TARGET]),
    ("mypy", ["mypy", _TARGET]),
    ("pytest", ["pytest", _TARGET]),
]
