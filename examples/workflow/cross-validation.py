"""
Cross-Validation
================

A model is judged on wines it has not seen, but one set of test wines is one opinion. Another
split would give another score, and on a small dataset the difference can be as large as the
difference between two models. Cross-validation measures the model several times, on several
different parts of the data, and reports all the scores.

This page teaches no model. It shows how to measure one, with
:func:`~pyml.model_selection.train_test_split`, :class:`~pyml.model_selection.KFold`,
:class:`~pyml.pipeline.Pipeline` and :func:`~pyml.model_selection.cross_val_score`. The models
are :class:`~pyml.tree.DecisionTreeClassifier` and :class:`~pyml.neighbors.KNNClassifier`, and
they are only something to measure.

Use cross-validation to compare models or settings, and whenever the data is so small that a
single split would decide the answer.
"""

# %%
# Dataset
# -------
#
# The task is to tell which of three grape varieties a wine was made from, using 13 chemical
# measurements. There are only 178 wines, which is what makes one test set unreliable. The
# page needs no scaling to start with: the first model, a tree, does not care about the scale of
# the features.
#
# .. dataset-info:: wine

# %%
# One split is one opinion
# ------------------------
#
# The same model is trained and tested 30 times. Only the seed of the split changes, and each
# time 20 percent of the wines, 35 of them, are the test set. The scores spread over a wide
# range, so a report of one score depends on which wines happened to be in the test set.
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_wine

from pyml.core.protocols import SupervisedLike
from pyml.model_selection import KFold, cross_val_score, train_test_split
from pyml.neighbors import KNNClassifier
from pyml.pipeline import Pipeline
from pyml.preprocessing import StandardScaler
from pyml.tree import DecisionTreeClassifier

data = load_wine()
X, y = data.X, data.y

split_scores = []
for seed in range(30):
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a class label.
    y_train = np.asarray(y_train_raw, dtype=np.int64)
    y_test = np.asarray(y_test_raw, dtype=np.int64)
    tree = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    split_scores.append(tree.score(X_test, y_test))

print(f"lowest score:  {min(split_scores):.3f}")
print(f"highest score: {max(split_scores):.3f}")

fig, ax = plt.subplots()
ax.hist(split_scores, bins=8)
ax.set_xlabel("test accuracy")
ax.set_ylabel("number of splits")
ax.set_title("The same model on 30 different splits")
plt.show()

# %%
# Folds
# -----
#
# :class:`~pyml.model_selection.KFold` divides the wines into ``n_splits`` groups, the folds.
# Each fold is the validation set once, and the other folds together are the training set, so
# every wine is used for validation exactly once. ``split`` only returns indices, and the table
# counts the wines of each variety in every validation fold.
#
# The wines in this file are sorted by variety: the first 59 are of one variety, the next 71 of
# the second and the last 48 of the third. Without shuffling, every fold is a block of
# consecutive wines, and the first validation fold has only the first variety. The model that is
# scored on it was trained on just 23 of the 59 wines of that variety.
Counts = NDArray[np.int64]


def validation_counts(folds: KFold) -> Counts:
    """Return the number of wines of each variety in each validation fold."""
    return np.array(
        [np.bincount(y[val_idx], minlength=3) for _, val_idx in folds.split(X, y)],
        dtype=np.int64,
    )


plain = KFold(n_splits=5)
shuffled = KFold(n_splits=5, shuffle=True, random_state=0)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharey=True)
positions = np.arange(5)
for ax, folds, title in zip(axes, (plain, shuffled), ("No shuffling", "Shuffled"), strict=True):
    counts = validation_counts(folds)
    left = np.zeros(5, dtype=np.int64)
    for variety in range(3):
        ax.barh(
            positions,
            counts[:, variety],
            left=left,
            color=f"C{variety}",
            label=f"variety {variety}",
        )
        left += counts[:, variety]
    ax.set_title(title)
    ax.set_xlabel("wines in the validation fold")
axes[0].set_yticks(positions, [f"fold {i + 1}" for i in positions])
axes[0].invert_yaxis()
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=3)
plt.show()

# %%
# A pitfall: sorted data and ``cv=5``
# -----------------------------------
#
# :func:`~pyml.model_selection.cross_val_score` fits a fresh copy of the model on the training
# folds of every fold and scores it on the validation fold. Given an integer, ``cv=5`` means
# ``KFold(5)``, which does not shuffle. On sorted data the scores are lower and they vary from
# one fold to the next. A ``KFold`` with ``shuffle=True`` and a ``random_state`` gives folds
# that mix the varieties and are the same on every run.
model = DecisionTreeClassifier(max_depth=3)

plain_scores = cross_val_score(model, X, y, cv=5)
shuffled_scores = cross_val_score(model, X, y, cv=shuffled)
print("cv=5, no shuffling:", plain_scores.round(3), f"mean {plain_scores.mean():.3f}")
print("shuffled KFold:    ", shuffled_scores.round(3), f"mean {shuffled_scores.mean():.3f}")

# %%
# Preprocessing belongs inside the folds
# --------------------------------------
#
# :class:`~pyml.neighbors.KNNClassifier` measures distances, so the features must be on the same
# scale. Here ``proline`` is in the hundreds and ``hue`` is below 2, so without scaling the
# distance is nearly the difference of the proline alone. The scaler is a step of a
# :class:`~pyml.pipeline.Pipeline`, and the pipeline is what ``cross_val_score`` measures: in
# every fold the scaler learns the mean and the spread from the training wines only, and then
# it is applied to the validation wines, as it will be applied to new wines later. The first
# score below is without scaling and the second is with it.
knn = KNNClassifier(n_neighbors=5)
scaled_knn = Pipeline([("scale", StandardScaler()), ("knn", KNNClassifier(n_neighbors=5))])

raw_scores = cross_val_score(knn, X, y, cv=shuffled)
pipeline_scores = cross_val_score(scaled_knn, X, y, cv=shuffled)
print("without scaling:", raw_scores.round(3), f"mean {raw_scores.mean():.3f}")
print("with the pipeline:", pipeline_scores.round(3), f"mean {pipeline_scores.mean():.3f}")
print("the pipeline itself was fitted:", getattr(scaled_knn, "is_fitted_", False))

# %%
# Why not scale everything first
# ------------------------------
#
# The shorter way is to standardize all the wines once and cross-validate the model on the
# result. Then the mean and the spread of every validation fold were used to scale the training
# folds, which is information the model could not have in real use. This is a leak. For a
# scaler on this dataset the leak is tiny: it passes only the 13 means and 13 spreads, and the
# table gives the mean difference of 20 different shuffles. The habit is still worth keeping,
# because a step that learns more from the data (a selection of features, a model) leaks more,
# and the pipeline is correct for every step without any extra thought.
gaps = []
for seed in range(20):
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    inside = cross_val_score(scaled_knn, X, y, cv=folds).mean()
    standardized = StandardScaler().fit_transform(X)
    outside = cross_val_score(KNNClassifier(n_neighbors=5), standardized, y, cv=folds).mean()
    gaps.append(outside - inside)
print(f"scaling first minus pipeline, mean of 20 shuffles: {np.mean(gaps):+.4f}")

# %%
# Compare models
# --------------
#
# To compare models, give them the same ``KFold`` object, so that they are scored on the same
# folds. The line chart shows the score of every fold. The printed lines repeat the measurement
# for 10 different shuffles and give the mean of the means, because one run of five folds on 178
# wines is still a small sample: when the lines cross, as they do here, or when the difference
# is smaller than the spread between the folds, one run cannot say which model is better.
models: dict[str, SupervisedLike] = {
    "Tree, max_depth=3": DecisionTreeClassifier(max_depth=3),
    "Scaled KNN, n_neighbors=5": scaled_knn,
}

fig, ax = plt.subplots()
fold_numbers = np.arange(1, 6)
for name, candidate in models.items():
    ax.plot(
        fold_numbers,
        cross_val_score(candidate, X, y, cv=shuffled),
        marker="o",
        label=name,
    )
ax.set_xticks(fold_numbers, [f"fold {i}" for i in fold_numbers])
ax.set_ylabel("validation accuracy")
ax.set_ylim(0.78, 1.02)
ax.set_title("The same five folds for both models")
ax.legend(loc="lower left")
plt.show()

for name, candidate in models.items():
    means = [
        cross_val_score(candidate, X, y, cv=KFold(5, shuffle=True, random_state=seed)).mean()
        for seed in range(10)
    ]
    print(f"{name:<28} mean of 10 shuffles {np.mean(means):.3f}, lowest {min(means):.3f}")
