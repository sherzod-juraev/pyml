<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/branding/logo.svg">
    <img alt="pyml" src="assets/branding/logo.svg" width="300">
  </picture>
</p>

[![Release](https://img.shields.io/github/v/release/sherzod-juraev/pyml)](https://github.com/sherzod-juraev/pyml/releases)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Pyml](https://github.com/sherzod-juraev/pyml/actions/workflows/pyml.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/pyml.yml)
[![Tests](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml)
[![Docs](https://github.com/sherzod-juraev/pyml/actions/workflows/docs.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/docs.yml)
[![Ruff](https://img.shields.io/badge/Ruff-enabled-brightgreen)](https://docs.astral.sh/ruff/)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)
[![Interrogate](https://img.shields.io/badge/Interrogate-100%25-brightgreen)](https://interrogate.readthedocs.io/)
[![Nox](https://img.shields.io/badge/nox-sessions-blue)](https://nox.thea.codes/)

A from-scratch machine learning library built on NumPy and SciPy, with
a shared estimator architecture, manual gradient derivations, and full
mathematical documentation. No TensorFlow, no PyTorch — just the math.

- **Documentation**: https://pyml-edu.readthedocs.io
- **Source code**: https://github.com/sherzod-juraev/pyml
- **Datasets**: [pyml-datasets](https://pyml-datasets.readthedocs.io/en/latest/), a companion
  package with the classic datasets used in the examples

Curious why this project exists, not just what it does? See
[Philosophy](https://pyml-edu.readthedocs.io/en/latest/philosophy.html).

## Quick example

Every model follows the same `fit` / `predict` / `score` convention. This example uses
[pyml-datasets](https://github.com/sherzod-juraev/pyml-datasets) for the data.

```python
from pyml.model_selection import train_test_split
from pyml.neighbors import KNNClassifier
from pyml_datasets import load_iris

X, y = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNNClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # accuracy on the test set
```

## Installation

pyml is a personal project and is not published on PyPI. It requires Python 3.12 or newer
and is installed directly from GitHub:

```bash
pip install git+https://github.com/sherzod-juraev/pyml.git
```

To install a fixed release, add its tag after `@`, replacing `vX.Y.Z` with a tag from the
[Releases](https://github.com/sherzod-juraev/pyml/releases) page:

```bash
pip install git+https://github.com/sherzod-juraev/pyml.git@vX.Y.Z
```

The quick example above also needs the datasets package:

```bash
pip install git+https://github.com/sherzod-juraev/pyml-datasets.git
```

## What's inside

| Category              |                                     Models                                      |
|:----------------------|:-------------------------------------------------------------------------------:|
| Linear regression     |                  Linear Regression, Ridge, Lasso, Elastic Net                   |
| Linear classification | Logistic Regression, Ridge Classifier, Lasso Classifier, Elastic Net Classifier |
| Naive Bayes           |                            GaussianNB, MultinomialNB                            |
| Nearest neighbors     |       KNN Classifier, KNN Regressor, Radius Classifier, Radius Regressor        |
| Decision tree         |                Decision Tree Classifier, Decision Tree Regressor                |
| Clustering            |                                 KMeans, DBSCAN                                  |
| Preprocessing         |                  Standard Scaler, MinMax Scaler, Robust Scaler                  |
| Model selection       |                                Train/test split                                 |

## Documentation

- [Quick Start](https://pyml-edu.readthedocs.io/en/latest/quick_start/index.html)
- [API Reference](https://pyml-edu.readthedocs.io/en/latest/api/pyml/index.html)
- [Examples](https://pyml-edu.readthedocs.io/en/latest/quick_start/examples/index.html)
- [Philosophy](https://pyml-edu.readthedocs.io/en/latest/philosophy.html)

## Development

The development setup, the Nox sessions and the project conventions are described in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE) for details.
