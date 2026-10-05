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

## Project structure

```text
pyml/
├── .github/workflows/          # CI workflows
├── assets/                     # Branding
├── docs/                       # Sphinx documentation
├── pyml/                       # Source code
│   ├── cluster/                # Clustering
│   ├── core/                   # Base classes
│   ├── linear_model/           # Linear models
│   ├── metrics/                # Metrics
│   ├── model_selection/        # Train/test split
│   ├── naive_bayes/            # Naive Bayes
│   ├── neighbors/              # KNN, Radius
│   ├── preprocessing/          # Scalers
│   └── tree/                   # Decision trees
├── tests/                      # Test suite
├── .gitignore                  # Git ignore rules
├── .readthedocs.yaml           # Read the Docs config
├── CONTRIBUTING.md             # Contributing guide
├── README.md                   # Project overview
├── LICENSE                     # MIT License
├── noxfile.py                  # Nox sessions
└── pyproject.toml              # Project metadata and tool config
```

## Development setup

### Requirements

- Python 3.12 / 3.13 / 3.14

### Optional extras

The project defines the following extras in [pyproject.toml](pyproject.toml).

| Extra  |                        Purpose                        |
|:-------|:-----------------------------------------------------:|
| `nox`  |           Nox (for running quality checks)            |
| `dev`  |            mypy, ruff, pytest, interrogate            |
| `docs` |            Sphinx and documentation tools             |
| `live` |     Live documentation preview (Sphinx autobuild)     |
| `all`  |                   Everything above                    |

### Setup

```bash
git clone https://github.com/sherzod-juraev/pyml.git
cd pyml
```

Install the extras you need:

- Development dependencies:

```bash
pip install -e ".[dev]"
```

- Nox (for running quality checks):

```bash
pip install -e ".[nox]"
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

## Live documentation

To preview the documentation locally with auto-reload:

```bash
sphinx-autobuild docs/source docs/build/html/ # Linux/macOS
# or
sphinx-autobuild docs\source docs\build\html\ # Windows
```

Then open http://127.0.0.1:8000 in your browser.

## Nox sessions

This project uses [Nox](https://nox.thea.codes/) to run all quality checks in isolated
environments. Nox creates a separate virtual environment for each
session, installs the required dependencies, and runs the configured
commands. This guarantees reproducibility across machines and Python
versions.

Run all default sessions:

```bash
nox
```

List all available sessions:

```bash
nox -l
```

Run a specific session:

```bash
nox -s <session-name>
```

Replace `<session-name>` with one of the sessions listed by `nox -l`.

## Quality tooling

| Tool                        |              Purpose               |
|:----------------------------|:----------------------------------:|
| mypy (strict)               |        Static type checking        |
| interrogate                 |     Docstring coverage (100%)      |
| ruff                        |       Linting and formatting       |
| pytest                      |             Test suite             |
| sphinx-lint, doc8, rstcheck |   Documentation style and syntax   |
| sphinx linkcheck            | Validity of every link in the docs |
