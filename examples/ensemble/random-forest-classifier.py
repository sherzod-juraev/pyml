"""
Random Forest Classifier
========================

:class:`~pyml.ensemble.RandomForestClassifier` predicts a class by asking many decision trees and
taking a vote. Each tree is grown on a different random sample of the training data, and at every
question it looks at only some of the features. The trees therefore make different mistakes, and
the average of their answers is steadier than the answer of any one of them.

Use it when a single :class:`~pyml.tree.DecisionTreeClassifier` is the right kind of model for the
task, but its answer changes too much from one training set to another. It keeps the good sides of
a tree (no scaling of the features, a boundary of any shape) and it gives up the one that made the
tree easy to read: a hundred trees cannot be followed question by question. The last sections show
what the extra trees buy and where they stop helping.
"""

# %%
# Parameters
# ----------
#
# * ``max_depth`` is the largest number of questions in a row in every tree. It is required, as in
#   the :class:`~pyml.tree.DecisionTreeClassifier`. The trees of a forest can be deep, because the
#   vote evens out the noise that a single deep tree fits.
# * ``n_estimators`` is the number of trees.
# * ``max_features`` is how many features a tree looks at for each question. They are drawn at
#   random for every question, so that the trees do not all ask the same first question.
# * ``bootstrap`` decides whether each tree is grown on a sample of the rows, drawn with
#   replacement, or on all of them.
# * ``random_state`` fixes the draws, so that the same forest is grown every time.
# * ``min_samples`` and ``min_impurity_decrease`` are passed to every tree, and mean there what
#   they mean in the :class:`~pyml.tree.DecisionTreeClassifier`.
#
# The :class:`API reference <pyml.ensemble.RandomForestClassifier>` has the default values.

# %%
# Dataset
# -------
#
# The task is to tell three species of penguins apart from four measurements of the bill, the
# flippers and the body. A few penguins have missing measurements, and a tree cannot use a missing
# number, so those rows are left out. It is the same task as in the
# :class:`~pyml.tree.DecisionTreeClassifier` example, so that a forest and a single tree can be
# compared on the same penguins.
#
# .. dataset-info:: penguins

# %%
# Split the data
# --------------
#
# The model is judged on penguins it has not seen, so 20 percent of the data is kept for the test.
# The trees ask about one feature at a time, so the features are used as they are, without scaling.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_penguins

from pyml.ensemble import RandomForestClassifier
from pyml.model_selection import train_test_split
from pyml.tree import DecisionTreeClassifier

Floats = NDArray[np.floating[Any]]
Labels = NDArray[np.int64]

data = load_penguins()
all_names = list(data.feature_names)
measured = [all_names.index(name) for name in all_names[1:5]]  # the four measurements
names = [all_names[index] for index in measured]
species = tuple(data.target_names or ())
features = np.asarray(data.X, dtype=np.float64)[:, measured]
labels = np.asarray(data.y, dtype=np.int64)
complete = ~np.isnan(features).any(axis=1)
features, labels = features[complete], labels[complete]


def split(seed: int, columns: list[int] | None = None) -> tuple[Floats, Floats, Labels, Labels]:
    """Return the train and test features and labels, optionally with fewer columns."""
    X_train, X_test, y_train, y_test = train_test_split(
        features if columns is None else features[:, columns],
        labels,
        test_size=0.2,
        random_state=seed,
    )
    # train_test_split is typed for both kinds of target; say that this one is a class.
    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(y_test, dtype=np.int64),
    )


X_train, X_test, y_train, y_test = split(42)
print(f"{len(y_train)} penguins to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``fit`` grows the trees, and ``score`` is the share of test penguins that get the right species.
# The forest has 100 trees of depth 8, and the single tree next to it has the same depth. Both are
# deep enough to learn the training penguins by heart.
tree = DecisionTreeClassifier(max_depth=8)
tree.fit(X_train, y_train)
forest = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
print(f"accuracy on the training penguins, one tree: {tree.score(X_train, y_train):.3f}")
print(f"accuracy on the training penguins, forest:   {forest.score(X_train, y_train):.3f}")
print(f"accuracy on the test penguins, one tree:     {tree.score(X_test, y_test):.3f}")
print(f"accuracy on the test penguins, forest:       {forest.score(X_test, y_test):.3f}")

# %%
# The forest is right for 66 of the 68 test penguins, and the single tree for 63. Both fit the
# training penguins without a single mistake, so what separates them is how they do on penguins
# they have not seen. One split is a small test, though: the difference is three penguins. The
# next sections repeat the comparison on several splits.

# %%
# How the forest votes
# --------------------
#
# ``predict_proba`` gives one row per penguin and one column per species. Each tree answers with
# the species of the leaf that the penguin ends in, and the forest averages the answers of its
# trees. For some penguins every tree gives the same species, and the row is one 1 and two 0. The
# printout counts them, and then shows the penguin on which the trees agree least, with the
# number of trees that vote for each species.
proba = forest.predict_proba(X_test)
doubtful = int(np.argmin(proba.max(axis=1)))
votes = np.array([member.predict(X_test[[doubtful]])[0] for member in forest.estimators_])
counts = np.bincount(votes, minlength=len(species))
unanimous = int((proba.max(axis=1) == 1.0).sum())
print(f"all the trees agree on {unanimous} of the {len(y_test)} test penguins")
truth = species[y_test[doubtful]]
answer = species[forest.predict(X_test)[doubtful]]
print(f"the penguin is a {truth}, the forest says {answer}")
for number, name in enumerate(species):
    share = f"{counts[number]:>3} of {len(forest.estimators_)} trees"
    print(f"{name:<10} {share}   probability {proba[doubtful, number]:.2f}")

# %%
# All the trees agree on 36 of the 68 test penguins. On the penguin in the printout they are
# split: 49 of the 100 trees say Gentoo, 29 say Chinstrap and 22 say Adelie, and the probabilities
# are these counts divided by 100. The forest answers with the species of the largest probability,
# Gentoo, which is right, and the 0.49 is a sign that the penguin lies between species. A single
# tree would give one species and no sign of how close the choice was. The numbers equal the vote
# counts here because the leaves of deep trees hold one species; in general ``predict_proba``
# averages the probabilities of the leaves.

# %%
# The shape of the boundary
# -------------------------
#
# To draw the boundary, the models are fit on two features only: the length of the bill and the
# length of the flipper. A single tree cuts the plane into rectangles. The forest also draws
# rectangles, but its cuts are the vote of a hundred trees, each grown on a different sample of
# the penguins, so the edge between two species is placed where most of the trees put it and not
# where one tree did.
columns = [0, 2]
B_train, B_test, c_train, c_test = split(42, columns)
grid_x, grid_y = np.meshgrid(
    np.linspace(B_train[:, 0].min() - 2, B_train[:, 0].max() + 2, 150),
    np.linspace(B_train[:, 1].min() - 4, B_train[:, 1].max() + 4, 150),
)
grid = np.c_[grid_x.ravel(), grid_y.ravel()]

models: list[tuple[str, DecisionTreeClassifier | RandomForestClassifier]] = [
    ("One tree", DecisionTreeClassifier(max_depth=8)),
    ("Forest of 100 trees", RandomForestClassifier(max_depth=8, n_estimators=100, random_state=42)),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
for ax, (title, model) in zip(axes, models, strict=True):
    model.fit(B_train, c_train)
    ax.contourf(
        grid_x, grid_y, model.predict(grid).reshape(grid_x.shape), alpha=0.25, cmap="viridis"
    )
    ax.scatter(B_train[:, 0], B_train[:, 1], c=c_train, cmap="viridis", edgecolor="k", s=25)
    ax.set_xlabel(names[columns[0]])
    ax.set_title(f"{title}, test accuracy {model.score(B_test, c_test):.2f}")
axes[0].set_ylabel(names[columns[1]])
plt.show()

# %%
# On this split the two get the same 63 of 68 penguins right, and their boundaries are about as
# long, so the figure shows where the models differ and not a gap in the score. They cut the plane
# in somewhat different places at the edges between the species, where the species overlap. One
# split is too small to tell which of them is better there, so the models are fit again on five
# splits.
tree_mean = forest_mean = 0.0
for seed in range(5):
    P_train, P_test, q_train, q_test = split(seed, columns)
    best = DecisionTreeClassifier(max_depth=8).fit(P_train, q_train)
    voted = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=0)
    voted.fit(P_train, q_train)
    tree_mean += best.score(P_test, q_test) / 5
    forest_mean += voted.score(P_test, q_test) / 5
print(f"mean test accuracy over 5 splits, one tree: {tree_mean:.3f}")
print(f"mean test accuracy over 5 splits, forest:   {forest_mean:.3f}")

# %%
# How many trees
# --------------
#
# The forest is judged here on the four measurements, with trees of depth 8, averaged over five
# splits. The dashed line is a single tree of the same depth. A forest of one tree is only a tree
# that is grown on a sample of the rows and looks at two features per question, so it scores
# about as a single tree does. The test accuracy rises with the number of trees and then flattens,
# and the spread between the splits (the bars) gets smaller.
tree_counts = [1, 3, 10, 30, 100]
mean_scores = np.zeros(len(tree_counts))
spread = np.zeros(len(tree_counts))
single = 0.0
for position, n_trees in enumerate(tree_counts):
    scores = []
    for seed in range(5):
        A_train, A_test, b_train, b_test = split(seed)
        grown = RandomForestClassifier(max_depth=8, n_estimators=n_trees, random_state=0)
        scores.append(grown.fit(A_train, b_train).score(A_test, b_test))
    mean_scores[position] = np.mean(scores)
    spread[position] = np.std(scores)
for seed in range(5):
    A_train, A_test, b_train, b_test = split(seed)
    single += DecisionTreeClassifier(max_depth=8).fit(A_train, b_train).score(A_test, b_test) / 5

fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
ax.errorbar(tree_counts, mean_scores, yerr=spread, marker="o", capsize=3, label="Forest")
ax.axhline(single, color="gray", linestyle="--", label="One tree")
ax.set_xscale("log")
ax.set_xlabel("n_estimators")
ax.set_ylabel("test accuracy (mean and spread of 5 splits)")
ax.set_title("More trees help, and then they stop helping")
ax.legend(loc="lower right")
plt.show()

# %%
# A forest of one tree scores 0.95, and the single tree 0.96. With 3 trees the forest is at 0.97,
# with 10 trees at 0.98, and the last 70 trees add less than 0.01. The spread falls from
# about 0.02 for one or three trees to under 0.01 from ten trees on. Most of the gain comes from
# the first ten or so trees, so a few dozen are enough for a task of this size. Adding trees does
# not make a forest overfit; it only costs time, since every tree has to be grown and asked.

# %%
# A pitfall: weak features
# ------------------------
#
# The vote can only average what the trees know. Here the features are the depth of the bill and
# the body mass, which do not separate the species well. A tree and a forest of depth 20 both fit
# the training penguins almost without a mistake (0.99 and 0.98), and both get about 0.73 of the
# test penguins right, against 0.95 or more with the four measurements. Many trees do not make up
# for features that carry too little, and the training accuracy cannot tell you this. Only the
# test accuracy can.
columns = [1, 3]
train_accuracy = {"One tree": 0.0, "Forest": 0.0}
test_accuracy = {"One tree": 0.0, "Forest": 0.0}
for seed in range(5):
    D_train, D_test, e_train, e_test = split(seed, columns)
    weak: dict[str, DecisionTreeClassifier | RandomForestClassifier] = {
        "One tree": DecisionTreeClassifier(max_depth=20),
        "Forest": RandomForestClassifier(max_depth=20, n_estimators=50, random_state=0),
    }
    for label, model in weak.items():
        model.fit(D_train, e_train)
        train_accuracy[label] += model.score(D_train, e_train) / 5
        test_accuracy[label] += model.score(D_test, e_test) / 5
for label in train_accuracy:
    print(
        f"{label:<9} training accuracy {train_accuracy[label]:.2f}, "
        f"test accuracy {test_accuracy[label]:.2f}"
    )
