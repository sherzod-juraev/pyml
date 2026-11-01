"""
K Nearest Neighbors Regressor
=============================

:class:`~pyml.neighbors.KNNRegressor` predicts a number by looking at the training samples that
are most like the new one. It finds the ``n_neighbors`` training samples at the smallest
distance and returns the average of their targets. As in
:class:`~pyml.neighbors.KNNClassifier`, ``fit`` only stores the data, and the work happens when
a prediction is asked for.

Use it when the target changes with the features in a way that a straight line cannot follow,
and you have enough data to cover the cases. It makes no assumption about the shape of the
relation. It cannot say anything about a sample that is outside the range of the training data,
and it gets slower as the data grows. The last section shows the first of these.
"""

# %%
# Parameters
# ----------
#
# * ``n_neighbors`` is how many neighbors are averaged. One neighbor follows the data very
#   closely, many neighbors give a smooth answer.
# * ``metric`` is how the distance is measured: ``"euclidean"`` (the straight line),
#   ``"cityblock"`` (the sum of the differences of each feature) or ``"chebyshev"`` (the largest
#   difference of one feature).
# * ``weights`` says whether every neighbor counts the same (``"uniform"``) or a closer
#   neighbor counts more (``"distance"``).
#
# The :class:`API reference <pyml.neighbors.KNNRegressor>` has the default values.

# %%
# Dataset
# -------
#
# The task is the one of the :doc:`Linear Regression </examples/linear-model/linear-regression>`
# example: predict the compressive strength of a concrete mixture from its ingredients and its
# age. That makes the two models easy to compare.
#
# .. dataset-info:: concrete_strength

# %%
# Split and standardize
# ---------------------
#
# The model is judged on mixtures it has not seen, so 20 percent of the data is kept for the
# test. Distances add up the differences of all features, so a feature with large numbers counts
# more than one with small numbers. Each feature is standardized, with the mean and the spread
# of the training part only.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_concrete_strength

from pyml.linear_model import LinearRegression
from pyml.metrics import mean_absolute_error, root_mean_squared_error
from pyml.model_selection import train_test_split
from pyml.neighbors import KNNRegressor
from pyml.preprocessing import StandardScaler

Floats = NDArray[np.floating[Any]]

data = load_concrete_strength()
names = data.feature_names


def split(seed: int) -> tuple[Floats, Floats, Floats, Floats]:
    """Return the standardized train and test features and the targets of one split."""
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        data.X, data.y, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a float target.
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_test),
        np.asarray(y_train_raw, dtype=np.float64),
        np.asarray(y_test_raw, dtype=np.float64),
    )


A_train, A_test, y_train, y_test = split(42)
print(f"{len(y_train)} mixtures to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``fit`` keeps the training data in ``X_`` and ``y_`` and does nothing else. ``score`` is the R
# squared: 1 is perfect, and 0 is no better than always predicting the mean. The straight-line
# model of the other example is fit on the same split for comparison, and the nearest neighbors
# follow the strength clearly better.
model = KNNRegressor(n_neighbors=5).fit(A_train, y_train)
line = LinearRegression(learning_rate=0.1, max_iter=1000).fit(A_train, y_train)
y_pred = model.predict(A_test)
print(f"R squared, nearest neighbors: {model.score(A_test, y_test):.2f}")
print(f"R squared, straight line:     {line.score(A_test, y_test):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.1f} MPa")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.1f} MPa")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, s=20, alpha=0.7)
limits = [float(y_test.min()), float(y_test.max())]
ax.plot(limits, limits, color="C3", linestyle="--", label="Perfect prediction")
ax.set_xlabel("actual strength (MPa)")
ax.set_ylabel("predicted strength (MPa)")
ax.set_title("Predicted against actual")
ax.legend(loc="upper left")
plt.show()

# %%
# What a prediction does
# ----------------------
#
# The whole model fits in a few lines of NumPy: the distance from each test mixture to every
# training mixture, the five smallest, and the average of their strengths. It matches the model
# on every test mixture.
distance = np.linalg.norm(A_test[:, None, :] - A_train[None, :, :], axis=2)
nearest = np.argsort(distance, axis=1)[:, :5]
by_hand = y_train[nearest].mean(axis=1)
print("same predictions as the model:", bool(np.allclose(by_hand, y_pred)))

# %%
# Choose ``n_neighbors`` and ``weights``
# --------------------------------------
#
# One neighbor copies the strength of the closest mixture, so it matches the training data
# exactly and follows its noise. More neighbors average the noise away, up to a point, and then
# they average over mixtures that are no longer alike, and both scores fall. With
# ``weights="distance"`` the closer neighbors count more, which keeps the answer sharp for longer,
# and it scores higher than the plain average at every ``n_neighbors`` above one. The scores are
# the mean of ten splits.
ks = [1, 2, 3, 5, 8, 12, 20, 40, 80]
splits = [split(seed) for seed in range(10)]
scores = {
    ("uniform", "train"): np.zeros(len(ks)),
    ("uniform", "test"): np.zeros(len(ks)),
    ("distance", "test"): np.zeros(len(ks)),
}
for a_train, a_test, targets_train, targets_test in splits:
    for position, k in enumerate(ks):
        plain = KNNRegressor(n_neighbors=k).fit(a_train, targets_train)
        weighted = KNNRegressor(n_neighbors=k, weights="distance").fit(a_train, targets_train)
        scores["uniform", "train"][position] += plain.score(a_train, targets_train) / len(splits)
        scores["uniform", "test"][position] += plain.score(a_test, targets_test) / len(splits)
        scores["distance", "test"][position] += weighted.score(a_test, targets_test) / len(splits)

fig, ax = plt.subplots()
ax.plot(ks, scores["uniform", "train"], marker="o", linestyle="--", label="Train, uniform")
ax.plot(ks, scores["uniform", "test"], marker="o", label="Test, uniform")
ax.plot(ks, scores["distance", "test"], marker="o", label="Test, distance")
ax.set_xscale("log")
ax.set_xlabel("n_neighbors")
ax.set_ylabel("R squared (mean of 10 splits)")
ax.set_title("A few neighbors, weighted by distance")
ax.legend(loc="lower left")
plt.show()

# %%
# A pitfall: outside the training data
# ------------------------------------
#
# An average of training targets can never be more extreme than the training targets, and it
# can only describe what the training data has seen. To see the effect, the model is trained on
# the mixtures with less than 350 kg of cement and tested on the ones with more, which are
# stronger on average. The nearest neighbors are the strongest of the low-cement mixtures,
# and their average is too low for the test mixtures: on average 39 MPa when the truth is
# 48 MPa. The straight line carries the trend on, and with it the R squared is much better, even
# though it was the worse model on ordinary data.
cement = names.index("cement")
low_cement = data.X[:, cement] < 350
scaler = StandardScaler().fit(data.X[low_cement])
E_train = scaler.transform(data.X[low_cement])
E_test = scaler.transform(data.X[~low_cement])
t_train = np.asarray(data.y, dtype=np.float64)[low_cement]
t_test = np.asarray(data.y, dtype=np.float64)[~low_cement]

far_knn = KNNRegressor(n_neighbors=5, weights="distance").fit(E_train, t_train)
far_line = LinearRegression(learning_rate=0.1, max_iter=1000).fit(E_train, t_train)
print(f"{len(t_train)} mixtures to train on, {len(t_test)} to test on, with more cement")
print(f"mean strength: actual {t_test.mean():.0f}, neighbors {far_knn.predict(E_test).mean():.0f}")
print(f"R squared: neighbors {far_knn.score(E_test, t_test):.2f}", end=", ")
print(f"line {far_line.score(E_test, t_test):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained", sharey=True)
for ax, (label, fitted) in zip(
    axes, (("Neighbors", far_knn), ("Straight line", far_line)), strict=True
):
    ax.scatter(t_test, fitted.predict(E_test), s=14, alpha=0.7)
    ax.plot([0, 90], [0, 90], color="C3", linestyle="--")
    ax.set_xlabel("actual strength (MPa)")
    ax.set_title(f"{label}, R squared {fitted.score(E_test, t_test):.2f}")
axes[0].set_ylabel("predicted strength (MPa)")
plt.show()
