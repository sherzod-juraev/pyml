from pathlib import Path

import nox

_DOCS_DIR = Path("docs")
_DOCS_SOURCE = _DOCS_DIR / "source"
_DOCS_BUILD = _DOCS_DIR / "build"
_DOCS_HTML = _DOCS_BUILD / "html"
_DOCS_LINKCHECK = _DOCS_BUILD / "linkcheck"

nox.options.sessions = ["check"]


@nox.session(python=["3.12", "3.13", "3.14"])
@nox.parametrize("path", ["pyml", "tests"])
def check(session: nox.Session, path: str) -> None:
    session.install(".[dev]")

    commands: list[list[str]] = [
        ["ruff", "check", path],
        ["ruff", "format", "--check", path],
        ["mypy", path],
    ]

    if path == "pyml":
        commands.append(["interrogate", path])
    elif path == "tests":
        commands.append(["pytest", path])

    for command in commands:
        session.run(*command)


@nox.session(python=["3.13"])
def docs(session: nox.Session) -> None:
    session.install(".[docs]")

    _DOCS_HTML.mkdir(parents=True, exist_ok=True)
    source_dir, html_dir = str(_DOCS_SOURCE), str(_DOCS_HTML)

    commands: list[list[str]] = [
        ["sphinx-lint", source_dir],
        ["doc8", source_dir],
        ["rstcheck", "--recursive", source_dir],
        ["sphinx-build", "-b", "html", source_dir, html_dir, "-W", "--keep-going"],
    ]

    for command in commands:
        session.run(*command)


@nox.session(python=["3.13"])
def linkcheck(session: nox.Session) -> None:
    session.install(".[docs]")

    _DOCS_LINKCHECK.mkdir(parents=True, exist_ok=True)
    source_dir, linkcheck_dir = str(_DOCS_SOURCE), str(_DOCS_LINKCHECK)

    session.run("sphinx-build", "-b", "linkcheck", source_dir, linkcheck_dir)
