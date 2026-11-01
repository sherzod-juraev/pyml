"""
K Nearest Neighbors Classifier
==============================

:class:`~pyml.neighbors.KNNClassifier` classifies a sample by looking at the training samples
that are most like it. To classify a new sample it finds the ``n_neighbors`` training samples at
the smallest distance, and the class that most of them have is the answer. There is no
training in the usual sense: ``fit`` only stores the data, and all the work happens when a
prediction is asked for.

Use it when the classes form groups in the feature space but the boundary between them has no
simple formula, and as a baseline that makes no assumption about the shape of the boundary. It
needs the training data at prediction time, it gets slower as the data grows, and it depends on
distances, so the features must be on the same scale. The last section shows how much that
matters.
"""

# %%
# Parameters
# ----------
#
# * ``n_neighbors`` is how many neighbors vote. One neighbor follows the data very closely,
#   many neighbors smooth the answer. Choosing it is the main decision, and one of the sections
#   below does it.
# * ``metric`` is how the distance is measured: ``"euclidean"`` (the straight line),
#   ``"cityblock"`` (the sum of the differences of each feature) or ``"chebyshev"`` (the largest
#   difference of one feature).
# * ``weights`` says whether every neighbor has one vote (``"uniform"``) or a closer neighbor
#   has a larger vote (``"distance"``).
#
# The :class:`API reference <pyml.neighbors.KNNClassifier>` has the default values.

# %%
# Dataset
# -------
#
# The task is to tell which of three grape varieties a wine was made from, using 13 chemical
# measurements.
#
# .. dataset-info:: wine

# %%
# Split and standardize
# ---------------------
#
# The model is judged on wines it has not seen, so 20 percent of the data is kept for the test.
# Distances add up the differences of all features, so a feature with large numbers would decide
# everything. Each feature is standardized, with the mean and the spread of the training part
# only. The last section shows what happens without it.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_wine

from pyml.model_selection import train_test_split
from pyml.neighbors import KNNClassifier
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
# ``fit`` keeps the training data in ``X_`` and ``y_`` and does nothing else. ``score`` is the
# accuracy. Almost all the test wines get the right variety.
model = KNNClassifier(n_neighbors=5).fit(A_train, y_train)
print("stored training data:", model.X_.shape)
print(f"train accuracy: {model.score(A_train, y_train):.3f}")
print(f"test accuracy:  {model.score(A_test, y_test):.3f}")

# %%
# What a prediction does
# ----------------------
#
# The whole model fits in a few lines of NumPy: the distance from each test wine to every
# training wine, the five smallest, and the class that most of them have. The first test wine
# below has its five neighbors listed, and the by-hand result matches the model on every test
# wine.
distance = np.linalg.norm(A_test[:, None, :] - A_train[None, :, :], axis=2)
nearest = np.argsort(distance, axis=1)[:, :5]
votes = np.array([np.bincount(y_train[row], minlength=3) for row in nearest])
print("same predictions as the model:", bool(np.all(votes.argmax(axis=1) == model.predict(A_test))))

print("first test wine, true class", y_test[0])
for rank, index in enumerate(nearest[0], start=1):
    print(f"  neighbor {rank}: class {y_train[index]}, distance {distance[0, index]:.2f}")

# %%
# Choose ``n_neighbors``
# ----------------------
#
# One neighbor follows the training data exactly: every training wine is its own nearest
# neighbor, so the training accuracy is 1 and the answer changes with every noisy sample. Many
# neighbors smooth it, and when ``n_neighbors`` gets close to the size of the training data, the
# vote is nearly the same everywhere, the most common class, and both accuracies fall. Between
# the two, from 3 to about 50, this dataset is easy and the test accuracy is flat, so the choice
# hardly matters here. On noisier data the middle is narrower. The accuracy is the mean of ten
# splits.
ks = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]
splits = [split(seed) for seed in range(10)]
train_accuracy = np.zeros(len(ks))
test_accuracy = np.zeros(len(ks))
for _, _, a_train, a_test, labels_train, labels_test in splits:
    for position, k in enumerate(ks):
        fitted = KNNClassifier(n_neighbors=k).fit(a_train, labels_train)
        train_accuracy[position] += fitted.score(a_train, labels_train) / len(splits)
        test_accuracy[position] += fitted.score(a_test, labels_test) / len(splits)

fig, ax = plt.subplots()
ax.plot(ks, train_accuracy, marker="o", label="Train")
ax.plot(ks, test_accuracy, marker="o", label="Test")
ax.set_xscale("log")
ax.set_xlabel("n_neighbors")
ax.set_ylabel("accuracy (mean of 10 splits)")
ax.set_title("Too few and too many neighbors both cost")
ax.legend(loc="lower left")
plt.show()

# %%
# The boundary
# ------------
#
# With two features the boundary can be drawn. These are new models trained on only
# ``flavanoids`` and ``color_intensity``. With one neighbor the boundary follows single wines,
# including the ones that sit among another group, and with 25 neighbors it is a smoother border
# that ignores them. Smoother is not the same as more accurate: on this test of 35 wines the
# difference between the two is three wines, which is too few to call one of them better, and
# the ten splits above are the better guide.
pair = [names.index("flavanoids"), names.index("color_intensity")]
low = A_train[:, pair].min(axis=0) - 0.5
high = A_train[:, pair].max(axis=0) + 0.5
xx, yy = np.meshgrid(np.linspace(low[0], high[0], 150), np.linspace(low[1], high[1], 150))
grid = np.column_stack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained", sharey=True)
for ax, k in zip(axes, (1, 25), strict=True):
    pair_model = KNNClassifier(n_neighbors=k).fit(A_train[:, pair], y_train)
    regions = pair_model.predict(grid).reshape(xx.shape)
    accuracy = pair_model.score(A_test[:, pair], y_test)
    ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5, 2.5], colors=["C0", "C1", "C2"], alpha=0.2)
    for label in range(3):
        points = A_train[y_train == label][:, pair]
        ax.scatter(points[:, 0], points[:, 1], s=14, color=f"C{label}")
    ax.set_xlabel("flavanoids (standardized)")
    ax.set_title(f"n_neighbors={k}, test accuracy {accuracy:.2f}")
axes[0].set_ylabel("color_intensity (standardized)")
plt.show()

# %%
# A pitfall: skipping the scaling
# -------------------------------
#
# In the raw data ``proline`` is in the hundreds and ``hue`` is below 2, so the distance between
# two wines is nearly the difference of their proline and the other 12 features hardly count.
# The model still runs and gives an answer, a much worse one. The table gives the mean accuracy
# of ten splits with and without the scaling.
for k in (1, 5, 25):
    raw_scores, scaled_scores = [], []
    for r_train, r_test, a_train, a_test, labels_train, labels_test in splits:
        raw_fit = KNNClassifier(n_neighbors=k).fit(r_train, labels_train)
        scaled_fit = KNNClassifier(n_neighbors=k).fit(a_train, labels_train)
        raw_scores.append(raw_fit.score(r_test, labels_test))
        scaled_scores.append(scaled_fit.score(a_test, labels_test))
    print(
        f"n_neighbors={k:<3} raw features {np.mean(raw_scores):.3f}, "
        f"standardized {np.mean(scaled_scores):.3f}"
    )
