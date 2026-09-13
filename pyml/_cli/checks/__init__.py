"""Shared filesystem paths used across pyml's check step modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOCS_ROOT = PROJECT_ROOT / "docs"
TESTS_ROOT = PROJECT_ROOT / "tests"
CODE_ROOT = PROJECT_ROOT / "pyml"
