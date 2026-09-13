"""Documentation quality check steps.

Order: sphinx-lint (style check) -> doc8 (format check) -> rstcheck
(syntax check) -> sphinx linkcheck -> sphinx-build (HTML build).
"""

from pyml._cli.checks import DOCS_ROOT

_TARGET = str(DOCS_ROOT)

DOCS_STEPS = [
    (
        "sphinx-lint style check",
        [
            "sphinx-lint",
            f"{_TARGET}/source",
        ],
    ),
    (
        "doc8 format check",
        [
            "doc8",
            f"{_TARGET}/source",
        ],
    ),
    ("rstcheck syntax check", ["rstcheck", "--recursive", f"{_TARGET}/source"]),
    (
        "sphinx linkcheck",
        [
            "sphinx-build",
            "-b",
            "linkcheck",
            f"{_TARGET}/source",
            f"{_TARGET}/build/linkcheck",
        ],
    ),
    (
        "sphinx-build",
        [
            "sphinx-build",
            "-b",
            "html",
            f"{_TARGET}/source",
            f"{_TARGET}/build/html",
            "-W",
            "--keep-going",
        ],
    ),
]
