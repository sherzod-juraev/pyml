"""Internal CLI package for pyml's own development workflow.

Not part of pyml's public API — never imported as `pyml._cli` by
library users. Exposed only via the `pyml` console script
(see `[project.scripts]` in pyproject.toml), which runs project
quality checks (code, docs, tests) during development.
"""
