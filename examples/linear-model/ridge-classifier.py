"""
Ridge Classifier
================

:class:`~pyml.linear_model.RidgeClassifier` is logistic regression with a penalty on the size
of the weights. It decides between two classes in the same way, a weighted sum of the features
squeezed through the sigmoid, but the loss it minimizes also grows with the sum of the squared
weights. The parameter ``alpha`` says how much that penalty counts. The weights are pulled
toward zero, and none of them is set exactly to zero.

Use it when the plain :class:`~pyml.linear_model.LogisticRegression` gives weights that are
very large or that change a lot from one sample of the data to another, which happens when many
features carry the same information. The price is a bias: too strong a penalty makes the model
unsure about everything, and the last sections show where that point is.
"""

# %%
# Parameters
# ----------
#
# * ``alpha`` is the strength of the penalty. With ``alpha=0`` the model is logistic regression,
#   and the larger it is, the smaller the weights.
# * ``learning_rate``, ``max_iter``, ``tol`` and ``fit_intercept`` are the same as in
#   :class:`~pyml.linear_model.LogisticRegression`. A larger ``alpha`` needs a smaller
#   ``learning_rate``, the examples below use ``0.1 / (1 + alpha)``. The last section shows what
#   happens otherwise.
#
# The labels must be 0 and 1. The full list of parameters, with the default values, is in the
# :class:`API reference <pyml.linear_model.RidgeClassifier>`.

# %%
# Dataset
# -------
#
# The task is to tell a malignant tumor from a benign one using measurements of cell nuclei
# from a breast biopsy. The label ``0`` is malignant and ``1`` is benign.
#
# .. dataset-info:: breast_cancer

# %%
# Split and standardize
# ---------------------
#
# The penalty treats all weights alike, so the features must be on the same scale, and they
# are standardized. The scaler learns its mean and spread from the training part only, so no
# information from the test part leaks into the model. ``split`` repeats these steps for any
# random seed, the later sections need it.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_breast_cancer

from pyml.linear_model import LogisticRegression, RidgeClassifier
from pyml.model_selection import train_test_split
from pyml.preprocessing import StandardScaler

data = load_breast_cancer()
names = data.feature_names


def split(seed: int) -> tuple[Any, Any, NDArray[np.int64], NDArray[np.int64]]:
    """Return the standardized train and test features and the labels of one split."""
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        data.X, data.y, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a class label.
    y_train = np.asarray(y_train_raw, dtype=np.int64)
    y_test = np.asarray(y_test_raw, dtype=np.int64)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def fit_ridge(alpha: float, features: Any, labels: NDArray[np.int64]) -> RidgeClassifier:
    """Fit a ridge classifier with a step size that is safe for this ``alpha``."""
    model = RidgeClassifier(alpha=alpha, learning_rate=0.1 / (1 + alpha), max_iter=20000, tol=1e-6)
    return model.fit(features, labels)


A_train, A_test, y_train, y_test = split(42)
print(f"{len(y_train)} tumors to train on, {len(y_test)} to test on")

# %%
# Why a penalty
# -------------
#
# Many of the 30 features are near copies of each other: radius, perimeter and area all measure
# the size of a nucleus. The classes can also be separated almost perfectly. For plain logistic
# regression that is a bad mix: there is always a slightly better fit with slightly larger
# weights, so the loop keeps pushing them up. With the default stopping rule it stops after a
# couple of hundred steps, but only because the loss has become flat. Given 5000 steps and a
# much tighter ``tol``, the weights are more than twice as large and still growing. The ridge
# classifier settles at a finite answer on its own.
short = LogisticRegression(learning_rate=0.1, max_iter=1000).fit(A_train, y_train)
long = LogisticRegression(learning_rate=0.1, max_iter=5000, tol=1e-6).fit(A_train, y_train)
ridge = fit_ridge(0.01, A_train, y_train)


def report(label: str, fitted: Any) -> None:
    """Print how many steps a fit took and how large its weights are."""
    size = np.linalg.norm(fitted.coef_)
    print(f"{label:<22} {fitted.n_iter_:>5} steps, size of the weights {size:.1f}")


report("logistic, default tol:", short)
report("logistic, tol=1e-6:", long)
report("ridge, alpha=0.01:", ridge)
print(f"test accuracy: logistic {short.score(A_test, y_test):.3f}", end=", ")
print(f"ridge {ridge.score(A_test, y_test):.3f}")

# %%
# Choose ``alpha``
# ----------------
#
# The table fits the model for a range of ``alpha`` on the split above. ``weights`` is the
# length of the weight vector, and ``confidence`` is the typical probability that the model
# gives to the class it picks. A larger ``alpha`` shrinks the weights, and with them the
# confidence: the probabilities move toward 0.5. The accuracy stays the same up to a point and
# then falls.
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
print("alpha     steps  train  test  weights  confidence")
for alpha in alphas:
    fitted = fit_ridge(alpha, A_train, y_train)
    proba = fitted.predict_proba(A_test)
    confidence = np.median(np.maximum(proba, 1 - proba))
    print(
        f"{alpha:<8} {fitted.n_iter_:>6}  {fitted.score(A_train, y_train):.3f}  "
        f"{fitted.score(A_test, y_test):.3f}  {np.linalg.norm(fitted.coef_):>7.2f}  "
        f"{confidence:>10.2f}"
    )

# %%
# One split of 113 tumors is a small test, and one tumor is almost one percent. To see the
# trend, the figure repeats the experiment on ten different splits and averages the accuracy.
# Here the penalty does not make the model more accurate. The training and the test accuracy are
# close from the start, so there is no overfitting for the penalty to cure, and a stronger
# penalty only takes accuracy away from both.
splits = [split(seed) for seed in range(10)]
train_accuracy = np.zeros(len(alphas))
test_accuracy = np.zeros(len(alphas))
for a_train, a_test, labels_train, labels_test in splits:
    for position, alpha in enumerate(alphas):
        fitted = fit_ridge(alpha, a_train, labels_train)
        train_accuracy[position] += fitted.score(a_train, labels_train) / len(splits)
        test_accuracy[position] += fitted.score(a_test, labels_test) / len(splits)

fig, ax = plt.subplots()
ax.plot(alphas, train_accuracy, marker="o", label="Train")
ax.plot(alphas, test_accuracy, marker="o", label="Test")
ax.set_xscale("log")
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy (mean of 10 splits)")
ax.set_title("A stronger penalty lowers both accuracies")
ax.legend(loc="lower left")
plt.show()

# %%
# The weights
# -----------
#
# For each ``alpha`` the eight features with the largest weights in the weakest-penalty model
# are followed. They all shrink together toward zero. The three size features, ``worst radius``,
# ``worst perimeter`` and ``worst area``, stay close to each other the whole way, because the
# ridge penalty prefers to share a weight between features that carry the same information.
# Plain logistic regression has no such preference and can give the whole weight to one of them.
path = np.array([fit_ridge(alpha, A_train, y_train).coef_ for alpha in alphas])
top = np.argsort(np.abs(path[0]))[-8:]

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.6)
for column in top:
    ax.plot(alphas, path[:, column], marker="o", label=names[column])
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xscale("log")
ax.set_xlabel("alpha")
ax.set_ylabel("weight")
ax.set_title("The weights shrink together")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.show()

# %%
# A pitfall: ``alpha`` and the step size
# --------------------------------------
#
# The penalty adds to the curvature of the loss, so a larger ``alpha`` allows only a smaller
# ``learning_rate``. A step that was fine for ``alpha=0.01`` can be too large for ``alpha=10``,
# and then the weights grow with every step instead of settling. No error is raised. The fit
# ends with weights that are thousands of times too large, and the accuracy is worse than
# guessing. (The accuracy of the stable fit is low as well, only because ``alpha=10`` is a
# very strong penalty, as the table showed. The point here is the size of the weights.)
with np.errstate(all="ignore"):
    unsafe = RidgeClassifier(alpha=10, learning_rate=0.1, max_iter=1000).fit(A_train, y_train)
safe = fit_ridge(10, A_train, y_train)
for label, fitted in (("learning_rate=0.1", unsafe), ("0.1 / (1 + alpha)", safe)):
    size = np.linalg.norm(fitted.coef_)
    accuracy = fitted.score(A_test, y_test)
    print(f"{label:<18} size of the weights {size:>8.2f}, test accuracy {accuracy:.3f}")
