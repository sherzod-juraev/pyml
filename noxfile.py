"""Nox sessions: the quality checks that run locally and in the CI.

Run ``nox -l`` to list the sessions and ``nox`` to run the default ones.
"""

import shutil
import tomllib
import zipfile
from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.12", "3.13", "3.14"]
DEFAULT_PYTHON = "3.13"

_SOURCE_PATHS = ["pyml", "tests", "noxfile.py"]
_DOCS_SOURCE = "docs/source"
_DOCS_HTML = "docs/build/html"
_DOCS_LINKCHECK = "docs/build/linkcheck"

_WHEEL_REQUIRED = ["pyml/py.typed", "pyml/__init__.py", "pyml/__init__.pyi"]
_WHEEL_FORBIDDEN_PREFIXES = ("tests/", "docs/")

# Runs inside the session, from a folder outside the repository, against the installed wheel.
_SMOKE_TEST = """
from pathlib import Path

import pyml
from pyml.linear_model import Ridge

location = Path(pyml.__file__).resolve()
assert "site-packages" in location.parts, f"pyml was not imported from site-packages: {location}"
assert isinstance(pyml.__version__, str)
assert Ridge(alpha=1.0).get_params()["alpha"] == 1.0
print("pyml", pyml.__version__, "imported from", location.parent)
"""

nox.options.sessions = ["lint", "types", "docstrings", "tests", "tests-min", "docs", "package"]
nox.options.error_on_external_run = True


def _minimum_requirements() -> list[str]:
    """Return the runtime dependencies of pyml with every lower bound pinned exactly.

    The dependencies are read from ``pyproject.toml``, so the lowest supported versions are
    written in one place only. Only simple ``name>=version`` requirements are supported.
    """
    with Path("pyproject.toml").open("rb") as file:
        dependencies: list[str] = tomllib.load(file)["project"]["dependencies"]
    return [requirement.replace(">=", "==") for requirement in dependencies]


def _check_wheel_contents(session: nox.Session, wheel: Path) -> None:
    """Fail the session if the wheel lacks files users need or ships files they do not.

    Parameters
    ----------
    session : nox.Session
        Session that reports the error.
    wheel : Path
        Wheel file to inspect.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    problems = [f"missing {name}" for name in _WHEEL_REQUIRED if name not in names]
    if sum(name.endswith(".pyi") for name in names) < 2:
        problems.append("the stub files of the subpackages are missing")
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        problems.append("missing the LICENSE file in the metadata")
    problems += [
        f"unexpected {name}" for name in names if name.startswith(_WHEEL_FORBIDDEN_PREFIXES)
    ]

    if problems:
        session.error(f"{wheel.name}: " + "; ".join(problems))


@nox.session(python=DEFAULT_PYTHON)
def lint(session: nox.Session) -> None:
    """Check the code style and the formatting with ruff."""
    session.install(".[dev]")
    session.run("ruff", "check", *_SOURCE_PATHS)
    session.run("ruff", "format", "--check", *_SOURCE_PATHS)


@nox.session(python=DEFAULT_PYTHON)
def types(session: nox.Session) -> None:
    """Check the types with mypy in strict mode."""
    session.install(".[dev]")
    session.run("mypy", "pyml")
    session.run("mypy", "tests")


@nox.session(python=DEFAULT_PYTHON)
def docstrings(session: nox.Session) -> None:
    """Check that the docstring coverage of the package is 100%."""
    session.install(".[dev]")
    session.run("interrogate", "pyml")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite on every supported Python version.

    Arguments after ``--`` are passed to pytest, for example ``nox -s tests -- -k tree``.
    """
    session.install(".[dev]")
    session.run("pytest", *session.posargs)


@nox.session(name="tests-min", python=PYTHON_VERSIONS[0])
def tests_min(session: nox.Session) -> None:
    """Run the test suite with the oldest NumPy and SciPy that pyml claims to support."""
    session.install(*_minimum_requirements())
    session.install(".[dev]")
    session.run(
        "python",
        "-c",
        "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)",
    )
    session.run("pytest", *session.posargs)


@nox.session(python=DEFAULT_PYTHON)
def docs(session: nox.Session) -> None:
    """Check the style of the documentation and build it with warnings as errors."""
    session.install(".[docs]")
    session.run("sphinx-lint", _DOCS_SOURCE)
    session.run("doc8", _DOCS_SOURCE)
    session.run("rstcheck", "--recursive", _DOCS_SOURCE)
    session.run("sphinx-build", "-b", "html", "-W", "--keep-going", "-E", _DOCS_SOURCE, _DOCS_HTML)


@nox.session(python=DEFAULT_PYTHON)
def linkcheck(session: nox.Session) -> None:
    """Check that every link in the documentation is valid (needs the network, so it is slow)."""
    session.install(".[docs]")
    session.run("sphinx-build", "-b", "linkcheck", "-E", _DOCS_SOURCE, _DOCS_LINKCHECK)


@nox.session(python=DEFAULT_PYTHON)
def package(session: nox.Session) -> None:
    """Build the wheel, check its contents and import an installed copy of it."""
    session.install("build")
    dist = Path(session.create_tmp()) / "dist"
    # Leftovers of an earlier build would be packed into the new wheel as they are.
    for leftover in (dist, Path("build"), *Path().glob("*.egg-info")):
        shutil.rmtree(leftover, ignore_errors=True)
    session.run("python", "-m", "build", "--wheel", "--outdir", str(dist))

    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        session.error(f"expected exactly one wheel, found {len(wheels)}.")
    _check_wheel_contents(session, wheels[0])

    session.install(str(wheels[0]))
    session.chdir(session.create_tmp())
    session.run("python", "-c", _SMOKE_TEST)
