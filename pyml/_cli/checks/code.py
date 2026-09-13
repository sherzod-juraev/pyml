"""Code quality check steps for the pyml package source.

Order: ruff check --fix (auto-fixable lint issues) -> ruff format (code
formatting) -> ruff check (remaining lint issues) -> mypy (static type
checking) -> interrogate (docstring coverage).
"""

from pyml._cli.checks import CODE_ROOT

_TARGET = str(CODE_ROOT)

CODE_STEPS = [
    ("ruff check --fix", ["ruff", "check", _TARGET, "--fix"]),
    ("ruff format", ["ruff", "format", _TARGET]),
    ("ruff check", ["ruff", "check", _TARGET]),
    ("mypy", ["mypy", _TARGET]),
    ("interrogate", ["interrogate", _TARGET]),
]
