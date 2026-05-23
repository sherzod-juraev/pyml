# pyml — Pure Python Machine Learning

**Classical ML algorithms built from scratch with NumPy.**

[![Tests](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml/badge.svg)](https://github.com/sherzod-juraev/pyml/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](https://github.com/sherzod-juraev/pyml/releases)

---

## About

`pyml` is a self-contained machine learning library implementing classic
algorithms entirely from scratch. Every model is built with NumPy and SciPy —
no TensorFlow, no PyTorch, just the math.

This is an educational project, a companion to [`pydsa`](https://github.com/sherzod-juraev/pydsa) (Data Structures &
Algorithms from scratch). The goal is understanding ML internals through
implementation, testing, and comparison with scikit-learn.

---

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

12 machine learning models + 3 preprocessing scalers | 111 tests passing

## Project structure
```text
pyml/
├── linear_model   # Linear & Logistic Regression with L1/L2
├── neighbors      # KNN Classifier & Regressor
├── tree           # Decision Tree Classifier
├── cluster        # K-Means, DBSCAN
├── preprocess     # Scalers
└── exc            # Custom exceptions
tests/             # 111 tests passing   
```

## Testing
```bash
pytest tests/ -v
```
All models tested for
- Shape consistency
- Finite values (no NaN/inf)
- Basic accuracy
- Edge cases (single sample, constant target)
- Scikit-learn parity (ballpark accuracy)

## Quality

| Tool               | Status      |
|--------------------|-------------|
| pytest (111 tests) | ✅           |
| ruff (linting)     | 0 errors    |
| mypy (strict mode) | 0 errors    |
| GitHub Actions CI  | ✅           |

## Disclaimer
This is **alpha educational software** — not intended for production use.
Updates are frequent and APIs may change between releases.

## Author
**Sherzod Juraev** — a 3rd-year Artificial Intelligence student at the [National University of Uzbekistan](https://nuu.uz).

This project is part of a larger journey: understanding the foundations
of computer science and machine learning by building them from scratch.
See also: [`pydsa`](https://github.com/sherzod-juraev/pydsa) — Data Structures & Algorithms in pure Python.

## License
MIT — see [LICENSE](LICENSE) for details

Copyright © 2026 Sherzod Juraev.
