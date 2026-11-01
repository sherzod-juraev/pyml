"""
Radius Neighbors Regressor
==========================

:class:`~pyml.neighbors.RadiusNeighborsRegressor` predicts a number as the average of the
targets of all the training samples that lie within a given distance. It is the sibling of
:class:`~pyml.neighbors.KNNRegressor`. The K nearest neighbors average the same number of
samples everywhere. The radius neighbors average whatever lies within the distance: hundreds of
samples in a crowded region, a handful in a sparse one, and perhaps none at all.

Use it when "close" has a meaning of its own, a distance that you can state, and when the data
is spread evenly enough that a single radius suits all of it. When it is not, as in the example
below, the number of neighbors changes a lot from one sample to the next and the K nearest
neighbors are the safer choice. The last sections show what goes wrong.
"""

# %%
# Parameters
# ----------
#
# * ``radius`` is the distance within which a training sample counts as a neighbor. It is the
#   main decision, and it only makes sense for features that have been scaled.
# * ``metric`` is how the distance is measured: ``"euclidean"`` (the straight line),
#   ``"cityblock"`` (the sum of the differences of each feature) or ``"chebyshev"`` (the largest
#   difference of one feature).
# * ``weights`` says whether every neighbor counts the same (``"uniform"``) or a closer
#   neighbor counts more (``"distance"``).
# * ``outlier_label`` is the answer for a sample that has no neighbor within the radius. With
#   the default ``None`` the model raises an error instead.
#
# The :class:`API reference <pyml.neighbors.RadiusNeighborsRegressor>` has the default values.

# %%
# Dataset
# -------
#
# The task is to predict the age of an abalone, a sea snail, from physical measurements. The
# age is counted as the number of rings in the shell, which is slow work with a microscope, so
# a prediction from lengths and weights is useful. The ``sex`` column is a category stored as a
# code, and a distance would treat the code as a number, so it is left out.
#
# .. dataset-info:: abalone

# %%
# Split and standardize
# ---------------------
#
# The model is judged on abalone it has not seen, so 20 percent of the data is kept for the
# test. A radius is a distance, so it depends on the scale of the features. Each feature is
# standardized, with the mean and the spread of the training part only, and then a distance of 1
# means one standard deviation.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_abalone

from pyml.core.exceptions import NoNeighborsError
from pyml.model_selection import train_test_split
from pyml.neighbors import KNNRegressor, RadiusNeighborsRegressor
from pyml.preprocessing import StandardScaler

Floats = NDArray[np.floating[Any]]

data = load_abalone()
names = list(data.feature_names)
keep = [index for index, name in enumerate(names) if name != "sex"]
features = np.asarray(data.X, dtype=np.float64)[:, keep]
rings = np.asarray(data.y, dtype=np.float64)


def split(seed: int) -> tuple[Floats, Floats, Floats, Floats, Floats]:
    """Return the raw and standardized train and test features, and the targets."""
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        features, rings, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a float target.
    scaler = StandardScaler().fit(X_train)
    return (
        np.asarray(X_test, dtype=np.float64),
        scaler.transform(X_train),
        scaler.transform(X_test),
        np.asarray(y_train_raw, dtype=np.float64),
        np.asarray(y_test_raw, dtype=np.float64),
    )


X_test, A_train, A_test, y_train, y_test = split(42)
print(f"{len(y_train)} abalone to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``fit`` keeps the training data and does nothing else. A radius of 0.8 standard deviations
# leaves a few test abalone without a neighbor, and ``outlier_label`` gives them an answer, the
# mean number of rings of the training data. ``score`` is the R squared: 1 is perfect, and 0 is
# no better than always predicting the mean. The K nearest neighbors, with 20 neighbors, are
# fit for comparison.
model = RadiusNeighborsRegressor(radius=0.8, outlier_label=float(y_train.mean()))
model.fit(A_train, y_train)
reference = KNNRegressor(n_neighbors=20).fit(A_train, y_train)
print(f"R squared, radius neighbors:    {model.score(A_test, y_test):.2f}")
print(f"R squared, K nearest neighbors: {reference.score(A_test, y_test):.2f}")

# %%
# How many neighbors
# ------------------
#
# The number of neighbors is not set, so it can be counted from the distances. Abalone are very
# uneven: most have a middling weight and lie close together, and the heavy ones are few and far
# apart. The figure shows the number of neighbors of every test abalone against its weight. An
# abalone of middling weight has hundreds of neighbors and the answer is an average over a big
# crowd. A heavy abalone has a few dozen, a handful, or none, and its answer rests on very few
# animals. A single radius cannot suit both.
distance = np.linalg.norm(A_test[:, None, :] - A_train[None, :, :], axis=2)
count = (distance <= 0.8).sum(axis=1)
print(f"neighbors per test abalone: fewest {count.min()}, median {np.median(count):.0f}", end=", ")
print(f"most {count.max()}")
print(f"without any neighbor: {np.sum(count == 0)} of {len(count)}")

whole = names.index("whole_weight") - 1  # the column of "sex" was dropped
fig, ax = plt.subplots()
ax.scatter(X_test[:, whole], count + 1, s=12, alpha=0.6)
ax.set_yscale("log")
ax.set_xlabel("whole_weight (grams)")
ax.set_ylabel("neighbors within the radius, plus 1")
ax.set_title("Hundreds of neighbors in the middle, few for the heaviest")
plt.show()

# %%
# Choose ``radius``
# -----------------
#
# The left figure shows the share of test abalone that have no neighbor, and the right figure
# the R squared, averaged over ten splits, with the mean of the training targets as the answer
# for a sample without neighbors. A small radius leaves many abalone to that fallback, which is
# no model at all, and the R squared is low. A large radius averages over too many animals and
# loses the detail. The best radius is 0.8 to 1, and even there it is behind the
# K nearest neighbors, which pick the same number of neighbors in the crowd and in the tail.
radii = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
splits = [split(seed) for seed in range(10)]
without = np.zeros(len(radii))
score = np.zeros(len(radii))
neighbors_score = 0.0
for _, a_train, a_test, targets_train, targets_test in splits:
    gaps = np.linalg.norm(a_test[:, None, :] - a_train[None, :, :], axis=2)
    for position, radius in enumerate(radii):
        without[position] += np.mean(np.all(gaps > radius, axis=1)) / len(splits)
        fitted = RadiusNeighborsRegressor(radius=radius, outlier_label=float(targets_train.mean()))
        score[position] += fitted.fit(a_train, targets_train).score(a_test, targets_test) / len(
            splits
        )
    nearest = KNNRegressor(n_neighbors=20).fit(a_train, targets_train)
    neighbors_score += nearest.score(a_test, targets_test) / len(splits)

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
left.plot(radii, without, marker="o", color="gray")
left.set_xscale("log")
left.set_xlabel("radius (standard deviations)")
left.set_ylabel("share of abalone without neighbors")
left.set_title("A small radius leaves gaps")
right.plot(radii, score, marker="o", label="Radius neighbors")
right.axhline(neighbors_score, color="C1", linestyle="--", label="K nearest neighbors, 20")
right.set_xscale("log")
right.set_xlabel("radius (standard deviations)")
right.set_ylabel("R squared (mean of 10 splits)")
right.set_title("Best at 0.8 to 1")
right.legend(loc="lower center")
plt.show()

# %%
# A pitfall: no neighbors
# -----------------------
#
# With ``outlier_label=None``, the default, a single abalone without a neighbor stops the whole
# prediction with an error, even if all the others have plenty. A radius of 0.3 is too small for
# a good share of the test abalone.
try:
    RadiusNeighborsRegressor(radius=0.3).fit(A_train, y_train).predict(A_test)
except NoNeighborsError as error:
    print("error:", error)

# %%
# The way out is to set ``outlier_label``, but the number that it returns is not a prediction.
# It is a constant, and it is mixed in with the real predictions and counted by ``score`` as if
# it came from the model. The two lines below separate them: the R squared on the abalone that
# have neighbors is higher than on all of them, because the easy cases are the crowded
# ones. Report how many samples got the fallback together with the score.
small = RadiusNeighborsRegressor(radius=0.3, outlier_label=float(y_train.mean())).fit(
    A_train, y_train
)
has_neighbors = np.any(distance <= 0.3, axis=1)
print(f"without neighbors: {np.sum(~has_neighbors)} of {len(has_neighbors)} test abalone")
print(f"R squared on all of them:         {small.score(A_test, y_test):.2f}")
kept = small.score(A_test[has_neighbors], y_test[has_neighbors])
print(f"R squared on those with neighbors: {kept:.2f}")
