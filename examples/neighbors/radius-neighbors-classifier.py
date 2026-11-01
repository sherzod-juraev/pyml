"""
Radius Neighbors Classifier
===========================

:class:`~pyml.neighbors.RadiusNeighborsClassifier` classifies a sample by the vote of all the
training samples that lie within a given distance of it. It is the sibling of
:class:`~pyml.neighbors.KNNClassifier`. The K nearest neighbors always use the same number of
samples, however far they are. The radius neighbors always use the same distance, however many
samples fall inside: many in a crowded region, few in a sparse one, and perhaps none at all.

Use it when "close" has a meaning of its own, a distance that you can state, and when you want
the model to say that it has no answer for a sample that is far from everything it has seen.
The price is a distance to choose, and a number that depends on the units of the features. The
last sections show both.
"""

# %%
# Parameters
# ----------
#
# * ``radius`` is the distance within which a training sample counts as a neighbor. Choosing it
#   is the main decision, and it only makes sense for features that have been scaled.
# * ``metric`` is how the distance is measured: ``"euclidean"`` (the straight line),
#   ``"cityblock"`` (the sum of the differences of each feature) or ``"chebyshev"`` (the largest
#   difference of one feature).
# * ``weights`` says whether every neighbor has one vote (``"uniform"``) or a closer neighbor
#   has a larger vote (``"distance"``).
# * ``outlier_label`` is the answer for a sample that has no neighbor within the radius. With
#   the default ``None`` the model raises an error instead.
#
# The :class:`API reference <pyml.neighbors.RadiusNeighborsClassifier>` has the default values.

# %%
# Dataset
# -------
#
# The task is to tell which of three grape varieties a wine was made from, using 13 chemical
# measurements, as in the :doc:`K Nearest Neighbors Classifier <knn-classifier>` example.
#
# .. dataset-info:: wine

# %%
# Split and standardize
# ---------------------
#
# The model is judged on wines it has not seen, so 20 percent of the data is kept for the test.
# A radius is a distance, so it depends on the scale of the features. Each feature is
# standardized, with the mean and the spread of the training part only, and then a distance of 1
# means one standard deviation. The last section shows what happens without it.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_wine

from pyml.core.exceptions import NoNeighborsError
from pyml.model_selection import train_test_split
from pyml.neighbors import RadiusNeighborsClassifier
from pyml.preprocessing import StandardScaler

Floats = NDArray[np.floating[Any]]
Labels = NDArray[np.int64]

data = load_wine()
names = data.feature_names


def split(seed: int) -> tuple[Floats, Floats, Floats, Floats, Labels, Labels]:
    """Return the raw and the standardized train and test features, and the labels."""
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        data.X, data.y, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a class label.
    y_train = np.asarray(y_train_raw, dtype=np.int64)
    y_test = np.asarray(y_test_raw, dtype=np.int64)
    scaler = StandardScaler().fit(X_train)
    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(X_test, dtype=np.float64),
        scaler.transform(X_train),
        scaler.transform(X_test),
        y_train,
        y_test,
    )


X_train, X_test, A_train, A_test, y_train, y_test = split(42)
print(f"{len(y_train)} wines to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``fit`` keeps the training data and does nothing else. With a radius of 3.5 standard
# deviations the test wines have very different numbers of neighbors, which the code counts with
# NumPy. Some wines sit in a crowded region and have dozens of neighbors, and the loneliest has
# two. ``score`` is the accuracy, and the 35 test wines are too few to rely on it, the later
# sections average ten splits.
model = RadiusNeighborsClassifier(radius=3.5).fit(A_train, y_train)
distance = np.linalg.norm(A_test[:, None, :] - A_train[None, :, :], axis=2)
count = (distance <= 3.5).sum(axis=1)
print(f"neighbors per test wine: fewest {count.min()}, median {np.median(count):.0f}", end=", ")
print(f"most {count.max()}")
print(f"test accuracy: {model.score(A_test, y_test):.3f}")

# %%
# The region the model knows
# --------------------------
#
# With two features the answer can be drawn. These are new models trained on only
# ``flavanoids`` and ``color_intensity``, with a radius of 0.5, and ``outlier_label=-1`` as the
# answer for "no neighbor". The grey area is where no training wine lies within the radius. A
# wine that falls there gets no class, where the K nearest neighbors would still give one, based
# on neighbors that are far away.
pair = [names.index("flavanoids"), names.index("color_intensity")]
low = A_train[:, pair].min(axis=0) - 1
high = A_train[:, pair].max(axis=0) + 1
xx, yy = np.meshgrid(np.linspace(low[0], high[0], 150), np.linspace(low[1], high[1], 150))
grid = np.column_stack([xx.ravel(), yy.ravel()])

pair_model = RadiusNeighborsClassifier(radius=0.5, outlier_label=-1).fit(A_train[:, pair], y_train)
regions = pair_model.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(
    xx,
    yy,
    regions,
    levels=[-1.5, -0.5, 0.5, 1.5, 2.5],
    colors=["gray", "C0", "C1", "C2"],
    alpha=0.25,
)
for label in range(3):
    points = A_train[y_train == label][:, pair]
    ax.scatter(points[:, 0], points[:, 1], s=14, color=f"C{label}", label=f"Class {label}")
ax.set_xlabel("flavanoids (standardized)")
ax.set_ylabel("color_intensity (standardized)")
ax.set_title("Grey: no training wine within the radius")
ax.legend(loc="upper left")
plt.show()

# %%
# Choose ``radius``
# -----------------
#
# The left figure shows the share of test wines that have no neighbor, and the right figure the
# accuracy, averaged over ten splits. A small radius leaves many wines without an answer. The
# wines that do get one are right almost every time, because they are very close to some training
# wine, and a wine without an answer counts as a mistake in the lower curve. A large radius gives
# everybody an answer and mixes the classes, and the accuracy falls to the share of the most
# common class. In between, around 3.5 to 4 standard deviations, nearly all wines have neighbors
# and the accuracy is about that of the K nearest neighbors.
radii = [1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8]
splits = [split(seed) for seed in range(10)]
without = np.zeros(len(radii))
overall = np.zeros(len(radii))
answered = np.zeros(len(radii))
for _, _, a_train, a_test, labels_train, labels_test in splits:
    for position, radius in enumerate(radii):
        fitted = RadiusNeighborsClassifier(radius=radius, outlier_label=-1)
        predicted = fitted.fit(a_train, labels_train).predict(a_test)
        has_answer = predicted != -1
        without[position] += np.mean(~has_answer) / len(splits)
        overall[position] += np.mean(predicted == labels_test) / len(splits)
        if has_answer.any():
            right = predicted[has_answer] == labels_test[has_answer]
            answered[position] += np.mean(right) / len(splits)

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
left.plot(radii, without, marker="o", color="gray")
left.set_xlabel("radius (standard deviations)")
left.set_ylabel("share of wines without neighbors")
left.set_title("A small radius leaves gaps")
right.plot(radii, answered, marker="o", linestyle="--", label="Wines that got an answer")
right.plot(radii, overall, marker="o", label="All wines")
right.set_xlabel("radius (standard deviations)")
right.set_ylabel("accuracy (mean of 10 splits)")
right.set_title("A large radius mixes the classes")
right.legend(loc="lower center")
plt.show()

# %%
# A pitfall: no neighbors
# -----------------------
#
# With ``outlier_label=None``, the default, a single wine without a neighbor stops the whole
# prediction with an error, even if all the others have plenty. A radius of 2 is too small for
# some of the test wines.
try:
    RadiusNeighborsClassifier(radius=2).fit(A_train, y_train).predict(A_test)
except NoNeighborsError as error:
    print("error:", error)

# %%
# The way out is to set ``outlier_label``. The label has to be one that no real class uses,
# such as ``-1``, and the code that reads the predictions has to expect it. A ``-1`` that is
# counted as a class by mistake, in the accuracy for example, is the same as a wrong answer.
outlier_model = RadiusNeighborsClassifier(radius=2, outlier_label=-1).fit(A_train, y_train)
predicted = outlier_model.predict(A_test)
print(f"no answer for {np.sum(predicted == -1)} of {len(predicted)} test wines")
print(
    f"accuracy of the others: {np.mean(predicted[predicted != -1] == y_test[predicted != -1]):.3f}"
)

# %%
# A pitfall: the units of the features
# ------------------------------------
#
# The radius is measured in the units of the features, so the same number means something else
# for each dataset and each scaling. In the raw data ``proline`` is in the hundreds, so the
# distance between two wines is mostly the difference of their proline, and a radius of 3.5,
# which suited the standardized features, finds no neighbor for almost every test wine.
raw_model = RadiusNeighborsClassifier(radius=3.5, outlier_label=-1).fit(X_train, y_train)
raw_predicted = raw_model.predict(X_test)
print(
    f"raw features, radius 3.5: no answer for {np.sum(raw_predicted == -1)} of {len(y_test)} wines"
)
