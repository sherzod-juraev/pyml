"""Filesystem paths shared across pyml's CLI commands."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "pyml"
DOCS_ROOT = PROJECT_ROOT / "docs"
DOCS_SOURCE = DOCS_ROOT / "source"
DOCS_BUILD = DOCS_ROOT / "build" / "html"
TESTS_ROOT = PROJECT_ROOT / "tests"
