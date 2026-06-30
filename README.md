# pyml — Pure Python Machine Learning

**Classical ML algorithms built from scratch with NumPy.**

[![Tests](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml) 
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/) 
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](https://github.com/sherzod-juraev/pyml/releases)

## About

`pyml` is an educational machine learning library that implements classical
algorithms entirely from scratch using only NumPy and SciPy. No external ML
frameworks are used in any model implementation — every algorithm is built
by hand from mathematical foundations.

The library is designed for students and practitioners who want to understand
*how* machine learning algorithms work internally, not just *how to use* them.
Each model is documented with full mathematical derivations, algorithmic
descriptions, and numerical stability considerations.

**scikit-learn is used exclusively for synthetic dataset generation in tests**
(e.g., `make_classification`, `make_regression`). No scikit-learn estimators
are used for comparison or validation — the library stands on its own.

## Installation

```bash
git clone https://github.com/sherzod-juraev/pyml.git
cd pyml
pip install -e .
```
For development tools
```bash
pip install -e ".[dev]"
```
For documentation tools
```bash
pip install -e ".[docs]"
```

## Quick start
```python
import numpy as np
from pyml.linear_model import LinearRegression
from pyml.preprocess import StandardScaler

# Create data
X = np.random.randn(100, 3)
true_w = np.array([1.5, -2.0, 0.5])
y = X @ true_w + 4.0 + 0.1 * np.random.randn(100)

# Scale and fit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression(learning_rate=0.01, max_iter=1000)
model.fit(X_scaled, y)

# Predict
X_new = np.random.randn(10, 3)
predictions = model.predict(scaler.transform(X_new))
```

## Implemented Models
| Category       |                         Models                         |
|:---------------|:------------------------------------------------------:|
| Linear Models  |  Linear Regression, Ridge, Lasso, Logistic Regression  |
| Neighbors      |             KNN Classifier, KNN Regressor              |
| Tree           |                Decision Tree Classifier                |
| Cluster        |                    K-Means, DBSCAN                     |
| Preprocessing  |       MinMaxScaler, StandardScaler, RobustScaler       |

13 models & scalers — all implemented from scratch with NumPy/SciPy.

## Project structure
```text
pyml/
├── linear_model   # Linear & Logistic Regression with L1/L2
├── neighbors      # KNN Classifier & Regressor
├── tree           # Decision Tree Classifier
├── cluster        # K-Means, DBSCAN
├── preprocess     # Scalers
└── exc            # Custom exceptions
tests/             # Test suite  
```

## Testing
```bash
pytest tests/ -v
```

### All models are tested for:
- Shape consistency
- Finite values (no NaN/inf)
- Basic accuracy on simple problems
- Edge cases (single sample, constant target, zero variance)

Test datasets are generated using numpy.random and, where convenient,
sklearn.datasets (synthetic data only — no sklearn models are invoked).

## Quality

| Tool               |   Status    |                         Configuration                          |
|:-------------------|:-----------:|:--------------------------------------------------------------:|
| pytest             | 204 passing |                                                                |
| ruff (linting)     |  0 errors   | `E, F, I, N, W, UP, B, C4, SIM, D, ARG, RUF, T20, FLY, Q, RSE` |
| mypy (type check)  |  0 errors   |                  `strict = true` Python 3.12                   |
| CI                 |   Passing   |                         Github Actions                         |

## Documentation
API documentation with full mathematical derivations is generated using
Sphinx and can be built locally. The documentation has not yet been
published to a public website.

### Install documentation dependencies
```bash
pip install -e ".[docs]"
```
### Navigate to the docs directory
```bash
cd docs
```
### Build HTML documentation
```bash
make html        # on Linux/macOS
.\make.bat html  # on Windows
```
### Open in browser
```bash
start build\html\index.html     # Windows
open build/html/index.html      # macOS
xdg-open build/html/index.html  # Linux
```
The documentation is written in NumPy docstring format with LaTeX math
rendered via MathJax.

## Academic context
This project is developed as part of an undergraduate Artificial Intelligence
curriculum at the [National University of Uzbekistan](https://nuu.uz).

The philosophy behind this project is the same: understand by building.
No black boxes — only transparent mathematical implementations in code.

## Limitations
This is **educational software**, not a production library. Known limitations
include:

- No GPU acceleration or distributed computing support.
- Limited to dense NumPy arrays (no sparse matrix support).

These design choices prioritize clarity of implementation over computational
efficiency.

## Author
**Sherzod Juraev** — a 3rd-year Artificial Intelligence student
at the [National University of Uzbekistan](https://nuu.uz).

## License
MIT — see [LICENSE](LICENSE) for details

Copyright © 2026 Sherzod Juraev.
