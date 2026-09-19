<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/branding/logo.svg">
    <img alt="pyml" src="assets/branding/logo.svg" width="300">
  </picture>
</p>

[![Pyml](https://github.com/sherzod-juraev/pyml/actions/workflows/pyml.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/pyml.yml)
[![Tests](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml)
[![Docs](https://github.com/sherzod-juraev/pyml/actions/workflows/docs.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/docs.yml)
[![Documentation Status](https://readthedocs.org/projects/pyml-edu/badge/?version=latest)](https://pyml-edu.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/Ruff-enabled-brightgreen)](https://docs.astral.sh/ruff/)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)
[![Interrogate](https://img.shields.io/badge/Interrogate-100%25-brightgreen)](https://interrogate.readthedocs.io/)

A from-scratch machine learning library built on NumPy and SciPy, with
a shared estimator architecture, manual gradient derivations, and full
mathematical documentation. No TensorFlow, no PyTorch — just the math.

📖 [pyml-edu.readthedocs.io](https://pyml-edu.readthedocs.io)

Curious why this project exists, not just what it does? See
[Philosophy](https://pyml-edu.readthedocs.io/en/latest/philosophy.html).

## Quick example

```python
from pyml.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Installation

```bash
pip install git+https://github.com/sherzod-juraev/pyml.git
```

## What's inside

| Category              |                                     Models                                      |
|:----------------------|:-------------------------------------------------------------------------------:|
| Linear regression     |                  Linear Regression, Ridge, Lasso, Elastic Net                   |
| Linear classification | Logistic Regression, Ridge Classifier, Lasso Classifier, Elastic Net Classifier |
| Naive Bayes           |                            GaussianNB, MultinomialNB                            |
| Nearest neighbors     |       KNN Classifier, KNN Regressor, Radius Classifier, Radius Regressor        |
| Clustering            |                                 KMeans, DBSCAN                                  |
| Preprocessing         |                  Standard Scaler, MinMax Scaler, Robust Scaler                  |
| Model selection       |                                Train/test split                                 |

## Project structure

```text
pyml/
├── .github/
│   └── workflows/
│       ├── docs.yml
│       ├── docs-linkcheck.yml
│       ├── pyml.yml
│       └── tests.yml
│
├── assets/
│
├── docs/
│
├── pyml/
│   ├── _cli/
│   ├── cluster/
│   ├── core/
│   ├── linear_model/
│   ├── metrics/
│   ├── model_selection/
│   ├── naive_bayes/
│   ├── neighbors/
│   └── preprocessing/
│
├── tests/
│   ├── cluster/
│   ├── core/
│   ├── linear_model/
│   ├── metrics/
│   ├── model_selection/
│   ├── naive_bayes/
│   ├── neighbors/
│   ├── preprocessing/
│   └── conftest.py
│
├── .gitignore
├── .readthedocs.yaml
├── pyproject.toml
├── README.md
└── LICENSE
```

## Quality tooling

| Tool                        |               Checks                |
|:----------------------------|:-----------------------------------:|
| mypy (strict)               |        Static type checking         |
| interrogate                 |      Docstring coverage (100%)      |
| ruff                        |       Linting and formatting        |
| pytest                      |             Test suite              |
| sphinx-lint, doc8, rstcheck |   Documentation style and syntax    |
| sphinx linkcheck            | Validity of every link in the docs  |

Each row runs via the `pyml` CLI. See [Development](https://pyml-edu.readthedocs.io/en/latest/quick_start/cli.html)
for setup and usage.

## License

MIT License. See [LICENSE](LICENSE) for details.