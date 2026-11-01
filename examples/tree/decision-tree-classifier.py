"""
Decision Tree Classifier
========================

:class:`~pyml.tree.DecisionTreeClassifier` predicts a class by asking a series of yes/no
questions about the features, such as "is the flipper longer than 206 mm?". Each answer sends
the sample down one of two branches, and the sample ends in a leaf that holds the class. Training
chooses, at every step, the question that separates the classes best.

Use it when you want a model that can be read and explained, that needs no scaling of the
features, and that can follow a boundary of any shape. It has one weakness that you must watch: a
deep tree can learn the training data by heart. The last sections show how to see that and what to
do about it.
"""

# %%
# Parameters
# ----------
#
# * ``max_depth`` is the largest number of questions in a row. It is required, and it is the
#   main decision of this model: a shallow tree is simple and may miss the pattern, a deep tree
#   can fit the noise.
# * ``min_samples`` is the smallest number of samples that a branch must hold to be split further.
# * ``min_impurity_decrease`` is the smallest gain that makes a question worth asking.
#
# The :class:`API reference <pyml.tree.DecisionTreeClassifier>` has the default values.

# %%
# Dataset
# -------
#
# The task is to tell three species of penguins apart from four measurements of the bill, the
# flippers and the body. A few penguins have missing measurements, and a tree cannot use a missing
# number, so those rows are left out.
#
# .. dataset-info:: penguins

# %%
# Split the data
# --------------
#
# The model is judged on penguins it has not seen, so 20 percent of the data is kept for the test.
# A tree does not compare sizes between features, so the features are used as they are, without
# scaling.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_penguins

from pyml.model_selection import train_test_split
from pyml.preprocessing import StandardScaler
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


def count_leaves(node: Any) -> int:
    """Count the leaves of a fitted tree."""
    return 1 if node.is_leaf else count_leaves(node.left) + count_leaves(node.right)


X_train, X_test, y_train, y_test = split(42)
print(f"{len(y_train)} penguins to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``fit`` builds the tree. ``score`` is the share of test penguins that get the right species.
# A tree of depth 3 is enough to start with.
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
print(f"accuracy on the training penguins: {model.score(X_train, y_train):.3f}")
print(f"accuracy on the test penguins:     {model.score(X_test, y_test):.3f}")

# %%
# Read the tree
# -------------
#
# The fitted tree is kept in ``tree_``. Each node holds a question ("is this feature at or below
# this value?") or, for a leaf, a class. Printing it as indented text gives the whole model in a
# few lines, which is what makes a tree easy to explain to someone else. Samples that answer yes
# go to the first branch under a question and the others to the second.


def describe(node: Any, depth: int = 0) -> None:
    """Print a node and the nodes below it as indented text."""
    indent = "    " * depth
    if node.is_leaf:
        print(f"{indent}-> {species[int(node.value)]}")
        return
    print(f"{indent}{names[node.feature_idx]} <= {node.threshold:.1f}?")
    describe(node.left, depth + 1)
    describe(node.right, depth + 1)


describe(model.tree_)

# %%
# The first question is about the flipper. Penguins with longer flippers are mostly Gentoo, and the
# depth of the bill confirms it. The others are told apart by the length of the bill and by the
# weight. Two of the questions (bill length at 42.3 and body mass at 4125) end in two leaves
# with the same species. Such a question makes the groups purer, but it never changes a
# prediction, so it can be skipped when you read the tree.

# %%
# Choose ``max_depth``
# --------------------
#
# The left figure shows the accuracy on the training penguins and on the test penguins, averaged
# over ten splits, for trees of growing depth. The right figure shows how many leaves the tree
# has. A tree of depth 0 is a single leaf that always gives the most common species, and it is
# right only about four times in ten. Depth 1 and 2 add the questions that matter, and at depth 2
# the test accuracy is already close to its best. After that the tree keeps fitting the training
# penguins better, up to 1.0, while the test accuracy does not follow, and the new leaves only
# describe single penguins.
depths = [0, 1, 2, 3, 4, 5, 6, 8]
train_scores = np.zeros(len(depths))
test_scores = np.zeros(len(depths))
leaf_counts = np.zeros(len(depths))
for seed in range(10):
    A_train, A_test, b_train, b_test = split(seed)
    for position, depth in enumerate(depths):
        fitted = DecisionTreeClassifier(max_depth=depth).fit(A_train, b_train)
        train_scores[position] += fitted.score(A_train, b_train) / 10
        test_scores[position] += fitted.score(A_test, b_test) / 10
        leaf_counts[position] += count_leaves(fitted.tree_) / 10

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
left.plot(depths, train_scores, marker="o", label="Training penguins")
left.plot(depths, test_scores, marker="o", label="Test penguins")
left.set_xlabel("max_depth")
left.set_ylabel("accuracy (mean of 10 splits)")
left.set_title("The test accuracy is flat from depth 2")
left.legend(loc="lower right")
right.plot(depths, leaf_counts, marker="o", color="gray")
right.set_xlabel("max_depth")
right.set_ylabel("leaves in the tree")
right.set_title("The tree stops growing when its leaves are pure")
plt.show()

# %%
# The shape of the boundary
# -------------------------
#
# To draw the boundary, the model is fit on two features only: the length of the bill and the
# length of the flipper. Every question is a cut along one axis, so the boundary is made of
# rectangles. A tree of depth 2 cuts the plane in four, and a tree of depth 8 cuts it into many
# more small rectangles.
columns = [0, 2]
B_train, B_test, c_train, c_test = split(42, columns)
grid_x, grid_y = np.meshgrid(
    np.linspace(B_train[:, 0].min() - 2, B_train[:, 0].max() + 2, 300),
    np.linspace(B_train[:, 1].min() - 4, B_train[:, 1].max() + 4, 300),
)
grid = np.c_[grid_x.ravel(), grid_y.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
for ax, depth in zip(axes, (2, 8), strict=True):
    shallow = DecisionTreeClassifier(max_depth=depth).fit(B_train, c_train)
    ax.contourf(
        grid_x, grid_y, shallow.predict(grid).reshape(grid_x.shape), alpha=0.25, cmap="viridis"
    )
    ax.scatter(B_train[:, 0], B_train[:, 1], c=c_train, cmap="viridis", edgecolor="k", s=25)
    ax.set_xlabel(names[columns[0]])
    ax.set_title(f"max_depth={depth}, test accuracy {shallow.score(B_test, c_test):.2f}")
axes[0].set_ylabel(names[columns[1]])
plt.show()

# %%
# The two trees differ by a few cuts at the edge between two species, where single penguins
# fall on the other side. The test set is too small to say that one of them is better, and the ten
# splits give about the same accuracy for the two, 0.95 for depth 2 and 0.93 for depth 8.

# %%
# No scaling needed
# -----------------
#
# A tree asks about one feature at a time, so the size of the numbers in one feature does not
# matter to another. If the features are standardized, the cuts move to the matching values and
# the predictions stay the same. This is the difference from the distance based models of the
# :doc:`Neighbors </examples/neighbors/index>` family.
scaler = StandardScaler().fit(X_train)
scaled = DecisionTreeClassifier(max_depth=4).fit(scaler.transform(X_train), y_train)
raw = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
same = np.mean(raw.predict(X_test) == scaled.predict(scaler.transform(X_test)))
print(f"test penguins with the same prediction, raw and standardized: {same:.0%}")

# %%
# A pitfall: a deep tree on weak features
# ---------------------------------------
#
# Now the features are the depth of the bill and the body mass, which do not separate the species
# well. A tree of depth 2 gets about 0.8 of the test penguins right, and it is stable. A tree
# of depth 20 has no limit, so it keeps asking until every training penguin is alone in its leaf,
# and it scores 0.99 on the training penguins. On the test penguins it does worse than the small
# tree, because the many small rectangles follow the noise of the training set. The training
# accuracy cannot tell you this. Only the test accuracy, or a separate part of the data, can.
columns = [1, 3]
D_train, D_test, e_train, e_test = split(42, columns)
for depth in (2, 20):
    weak = DecisionTreeClassifier(max_depth=depth).fit(D_train, e_train)
    print(
        f"max_depth={depth:>2}: {count_leaves(weak.tree_):>2} leaves, "
        f"training accuracy {weak.score(D_train, e_train):.2f}, "
        f"test accuracy {weak.score(D_test, e_test):.2f}"
    )
