# Contributing

> [!NOTE]
> This is a personal, educational project. I am not
> actively seeking contributors at this time.

However, this project is open-source under the
[MIT License](LICENSE), and you are **welcome to**:

- **Fork** this repository.
- **Copy** the code.
- **Modify** it for your own needs.
- **Develop** your own version independently.

If you build something interesting, I would love to hear about it.

The rest of this file is also a map of the project: where things are, how to check them, and
how a new model or a release is done.

## Quick start

```bash
git clone https://github.com/sherzod-juraev/pyml.git
cd pyml
pip install -e ".[nox]"
nox
```

`nox` runs every default session (see [Nox sessions](#nox-sessions)) in isolated virtual
environments, the same way as the CI.

## Project structure

```text
pyml/
├── .github/workflows/          # CI workflows
├── assets/                     # Branding
├── docs/                       # Sphinx documentation
├── pyml/                       # Source code
│   ├── cluster/                # Clustering
│   ├── core/                   # Base classes, validation, exceptions, type aliases
│   ├── linear_model/           # Linear models
│   ├── metrics/                # Metrics
│   ├── model_selection/        # Train/test split
│   ├── naive_bayes/            # Naive Bayes
│   ├── neighbors/              # KNN, Radius
│   ├── preprocessing/          # Scalers
│   └── tree/                   # Decision trees
├── tests/                      # Test suite, mirrors pyml/
├── .gitignore                  # Git ignore rules
├── .readthedocs.yaml           # Read the Docs config
├── CONTRIBUTING.md             # This file
├── LICENSE                     # MIT License
├── noxfile.py                  # Nox sessions
├── pyproject.toml              # Project metadata and tool config
└── README.md                   # Project overview
```

## Development setup

### Requirements

- Python 3.12, 3.13 and 3.14. Install all three to run the whole Nox matrix, or run a single
  version, for example `nox -s tests-3.13`.

### Optional extras

The project defines the following extras in [pyproject.toml](pyproject.toml).

| Extra    |                        Purpose                        |
|----------|:-----------------------------------------------------:|
| `nox`    |        Nox, which runs all the quality checks         |
| `dev`    |       mypy, ruff, pytest, interrogate (no Nox)        |
| `docs`   |            Sphinx and documentation tools             |
| `live`   |     Live documentation preview (Sphinx autobuild)     |
| `all`    |                   Everything above                    |

### Setup

```bash
git clone https://github.com/sherzod-juraev/pyml.git
cd pyml
```

Install the extras you need:

- Nox (recommended, runs everything in isolated environments):

```bash
pip install -e ".[nox]"
```

- Development dependencies, to run a tool directly in your own environment:

```bash
pip install -e ".[dev]"
```

- Documentation tools:

```bash
pip install -e ".[docs]"
```

- Live documentation preview:

```bash
pip install -e ".[live]"
```

- Everything at once:

```bash
pip install -e ".[all]"
```

## Nox sessions

This project uses [Nox](https://nox.thea.codes/) to run all quality checks in isolated
environments. Nox creates a separate virtual environment for each session, installs the
required dependencies, and runs the configured commands. This guarantees reproducibility
across machines and Python versions.

| Session      |      Python       |                             What it runs                              |
|:-------------|:-----------------:|:---------------------------------------------------------------------:|
| `lint`       |       3.13        |                  ruff check and ruff format --check                   |
| `types`      |       3.13        |                  mypy (strict) on `pyml` and `tests`                  |
| `docstrings` |       3.13        |             interrogate, docstring coverage must be 100%              |
| `tests`      | 3.12, 3.13, 3.14  |                                pytest                                 |
| `docs`       |       3.13        |            sphinx-lint, doc8, rstcheck, `sphinx-build -W`             |
| `linkcheck`  |       3.13        |         Sphinx linkcheck, validity of every link in the docs          |
| `tests-min`  |       3.12        |       pytest with the oldest NumPy and SciPy that pyml supports       |
| `package`    |       3.13        |   builds the wheel, checks its contents, imports an installed copy    |

`nox` without arguments runs every session except `linkcheck`, which needs the network and is
slow.

```bash
nox -l                          # list all sessions
nox -s lint                     # run one session
nox -s tests-3.13               # run one session on one Python version
nox -s tests -- -k tree         # arguments after -- are passed to pytest
nox -R                          # reuse the existing virtual environments, much faster
```

## Live documentation

Needs the `live` extra. To preview the documentation locally with auto-reload:

```bash
sphinx-autobuild docs/source docs/build/html/ # Linux/macOS
# or
sphinx-autobuild docs\source docs\build\html\ # Windows
```

Then open http://127.0.0.1:8000 in your browser and stop the server with `Ctrl+C`.

## Conventions

### Code

- Public imports have two levels only: `from pyml.linear_model import Ridge`, never
  `from pyml import Ridge`. Every package uses a lazy-loading `__init__.py` together with an
  `__init__.pyi` stub (`as` re-export style).
- `Regressor`, `Classifier`, `Transformer` and `Clusterer` each inherit `BaseEstimator`
  independently and declare their own abstract methods. Code shared by several models goes
  into a private base or a mixin (`_LinearModelBase`, `_DecisionTreeBase`), and a base is
  shared only when the algorithm really is shared.
- Errors come from the hierarchy in `pyml/core/exceptions/`, starting at `PymlError`.
- Everything is checked by mypy in strict mode, ruff (line length 100) and interrogate
  (100% docstring coverage, private members included).

### Docstrings

- NumPy style. A docstring that contains `.. math::` is a raw string (`r"""`).
- The module docstring is short prose, without math.
- The class docstring is the single source of truth for the parameters, so `__init__` is not
  documented separately. It holds the math derivation, `Parameters`, `Attributes` and a
  `.. note::`.
- Concrete public models also get a `.. plot::` demo. Abstract bases do not.

### Tests

- `tests/` mirrors `pyml/`, and `conftest.py` provides the `rng` fixture (seed 42).
- Tests are grouped in classes by behavior (`TestFit`, `TestPredict`, `TestScore`). There is
  only a module docstring, no docstrings on the tests.
- A test file covers the behavior of its own layer and does not repeat what lower layers
  already guarantee. Abstract bases have no test file of their own.

### Commits and versions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `feat`, `fix`,
  `docs`, `test`, `build`, `refactor` and `chore`, with a body that explains what changed and
  why.
- Work happens on `dev`. `main` receives a merge only when all checks pass.
- `__version__` in `pyml/__init__.py` is the single source of truth, `pyproject.toml` and the
  docs read it. It changes only when the public API changes (a new model family or a breaking
  change is a MINOR bump while the version is 0.x). Housekeeping commits do not change it.

## Adding a model

1. Decide the math and the architecture first: which base class, and whether a shared
   private base is justified.
2. Implement it in `pyml/<family>/` with a lazy-loading `__init__.py` and `__init__.pyi`.
3. Write the docstrings following the conventions above.
4. Write the tests in `tests/<family>/`, including the lazy-loading `__init__` test.
5. Add the documentation: an API page in `docs/source/api/pyml/`, an example page in
   `docs/source/quick_start/examples/<family>/`, and a row in the README table.
6. Run `nox` until every session passes.

## Releasing

1. Run `nox` on `dev`, and make sure the CI is green.
2. Bump `__version__` in `pyml/__init__.py` if the public API changed.
3. Merge `dev` into `main` and push.
4. Create the tag `vX.Y.Z` on `main` and a GitHub release titled `vX.Y.Z — <Model Family Name>`.
5. Check that the Read the Docs build is green.
