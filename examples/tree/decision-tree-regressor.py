"""
Decision Tree Regressor
=======================

:class:`~pyml.tree.DecisionTreeRegressor` predicts a number by asking a series of yes/no questions
about the features, in the same way as :class:`~pyml.tree.DecisionTreeClassifier`. The sample ends
in a leaf, and the leaf holds the average target of the training samples that ended there. The
model is a staircase: the prediction is constant inside each group and jumps from one group to the
next.

Use it when the target depends on the features in a way that is not a straight line, for example
when it changes suddenly at some value, or when two features act together. It needs no scaling of
the features. Like the classifier, it can learn the training data by heart if it is allowed to grow
too deep, and it cannot predict a value outside the range of the targets that it has seen. The
last sections show both.
"""

# %%
# Parameters
# ----------
#
# * ``max_depth`` is the largest number of questions in a row. It is required. Each question
#   splits a group in two, so a tree of depth ``d`` has at most ``2**d`` groups.
# * ``min_samples`` is the smallest number of samples that a group must hold to be split
#   further. A larger value gives fewer, larger groups.
# * ``min_impurity_decrease`` is the smallest gain that makes a question worth asking.
#
# The :class:`API reference <pyml.tree.DecisionTreeRegressor>` has the default values.

# %%
# Dataset
# -------
#
# The task is to predict the fuel consumption of a car, in miles per gallon, from its engine, its
# weight and its year. A few cars have no value for the horsepower, and a tree cannot use a
# missing number, so those rows are left out. ``cylinders`` and ``origin`` are whole numbers that
# stand for categories, and a tree can use them, because it only asks whether a value is above or
# below a cut.
#
# .. dataset-info:: auto_mpg

# %%
# Split the data
# --------------
#
# The model is judged on cars it has not seen, so 20 percent of the data is kept for the test.
# A tree does not compare sizes between features, so the features are used as they are.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_auto_mpg

from pyml.linear_model import LinearRegression
from pyml.model_selection import train_test_split
from pyml.preprocessing import StandardScaler
from pyml.tree import DecisionTreeRegressor

Floats = NDArray[np.floating[Any]]

data = load_auto_mpg()
names = list(data.feature_names)
features = np.asarray(data.X, dtype=np.float64)
mpg = np.asarray(data.y, dtype=np.float64)
complete = ~np.isnan(features).any(axis=1)
features, mpg = features[complete], mpg[complete]


def split(seed: int) -> tuple[Floats, Floats, Floats, Floats]:
    """Return the train and test features and targets."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, mpg, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a float target.
    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_train, dtype=np.float64),
        np.asarray(y_test, dtype=np.float64),
    )


def count_leaves(node: Any) -> int:
    """Count the leaves of a fitted tree."""
    return 1 if node.is_leaf else count_leaves(node.left) + count_leaves(node.right)


X_train, X_test, y_train, y_test = split(42)
print(f"{len(y_train)} cars to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``fit`` builds the tree. ``score`` is the R squared: 1 is perfect, and 0 is no better than
# always predicting the mean. A line fit on standardized features is the usual first model for
# this task, so it is fit as well, for comparison.
model = DecisionTreeRegressor(max_depth=5, min_samples=20)
model.fit(X_train, y_train)
scaler = StandardScaler().fit(X_train)
line = LinearRegression().fit(scaler.transform(X_train), y_train)
print(f"R squared, tree: {model.score(X_test, y_test):.2f}")
print(f"R squared, line: {line.score(scaler.transform(X_test), y_test):.2f}")

# %%
# Read the tree
# -------------
#
# Each node holds a question or, for a leaf, the average mpg of the cars that end there. A tree of
# depth 3 is small enough to read.
small = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)


def describe(node: Any, depth: int = 0) -> None:
    """Print a node and the nodes below it as indented text."""
    indent = "    " * depth
    if node.is_leaf:
        print(f"{indent}-> {node.value:.1f} mpg")
        return
    print(f"{indent}{names[node.feature_idx]} <= {node.threshold:.1f}?")
    describe(node.left, depth + 1)
    describe(node.right, depth + 1)


describe(small.tree_)

# %%
# The first question is the number of cylinders: cars with fewer than six cylinders (the first
# branch) use much less fuel than the others. After that the year and the weight matter for those
# cars, and the power and the year for the others. Each leaf is an average, so this tree predicts
# one of eight values, and every car in a group gets the same number.

# %%
# A staircase
# -----------
#
# To see the staircase, the model is fit on one feature only, the weight of the car. A tree of
# depth 2 has four steps. A tree of depth 6 has up to 64, and it follows the points much more
# closely, including the ones that lie away from the others.
weight = names.index("weight")
heavy = np.linspace(features[:, weight].min(), features[:, weight].max(), 400)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
for ax, depth in zip(axes, (2, 6), strict=True):
    steps = DecisionTreeRegressor(max_depth=depth).fit(X_train[:, [weight]], y_train)
    test_r2 = steps.score(X_test[:, [weight]], y_test)
    ax.scatter(X_train[:, weight], y_train, s=12, alpha=0.5)
    ax.plot(heavy, steps.predict(heavy[:, None]), color="C1", linewidth=2)
    ax.set_xlabel("weight (pounds)")
    ax.set_title(f"max_depth={depth}, test R squared {test_r2:.2f}")
axes[0].set_ylabel("mpg")
plt.show()

# %%
# The deeper tree fits the training cars better, but its test R squared is a little lower than
# the one of the small tree. The next section looks at this on all the features.

# %%
# Choose ``max_depth`` and ``min_samples``
# ----------------------------------------
#
# The left figure shows the R squared on the training cars and on the test cars, averaged over ten
# splits, for trees of growing depth. The training R squared goes to 1, because a deep tree gives
# each car its own leaf. The test R squared rises until depth 4 or 5 and then falls. The right
# figure keeps the depth free (30) and limits the size of a group instead: ``min_samples`` stops
# the tree from splitting a group that is small. This works as well as limiting the depth, and
# here it is slightly better, with the best test R squared at 20 samples.
depths = [1, 2, 3, 4, 5, 6, 8, 12]
sizes = [2, 5, 10, 20, 40, 80]
train_depth = np.zeros(len(depths))
test_depth = np.zeros(len(depths))
test_size = np.zeros(len(sizes))
test_line = 0.0
for seed in range(10):
    A_train, A_test, b_train, b_test = split(seed)
    for position, depth in enumerate(depths):
        fitted = DecisionTreeRegressor(max_depth=depth).fit(A_train, b_train)
        train_depth[position] += fitted.score(A_train, b_train) / 10
        test_depth[position] += fitted.score(A_test, b_test) / 10
    for position, size in enumerate(sizes):
        fitted = DecisionTreeRegressor(max_depth=30, min_samples=size).fit(A_train, b_train)
        test_size[position] += fitted.score(A_test, b_test) / 10
    scaler = StandardScaler().fit(A_train)
    straight = LinearRegression().fit(scaler.transform(A_train), b_train)
    test_line += straight.score(scaler.transform(A_test), b_test) / 10

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
left.plot(depths, train_depth, marker="o", label="Training cars")
left.plot(depths, test_depth, marker="o", label="Test cars")
left.set_xlabel("max_depth")
left.set_ylabel("R squared (mean of 10 splits)")
left.set_title("Deeper trees fit the training cars only")
left.legend(loc="lower right")
right.plot(sizes, test_size, marker="o", label="Tree, test cars")
right.axhline(test_line, color="C2", linestyle="--", label="Line, test cars")
right.set_xscale("log")
right.set_xlabel("min_samples (max_depth=30)")
right.set_ylabel("R squared (mean of 10 splits)")
right.set_title("Larger groups, better test R squared")
right.legend(loc="lower center")
plt.show()

# %%
# The best tree and the line score about the same on the test cars. A tree is not a better model by
# default: here the relation of the mpg to the features is close to a line, and the tree has to
# approximate it with steps. Its advantages are elsewhere, in the questions that it can answer
# without any scaling or any choice of transformation.

# %%
# A pitfall: a new range
# ----------------------
#
# A tree can only predict averages of the targets it has seen, so it cannot continue a trend
# beyond the training data. Here the model is trained on the light cars (below 3000 pounds) and
# tested on the heavy ones. The lowest answer the tree can give is the lowest average of a group
# of light cars, 19.5 mpg, so its predictions for the heavy cars are too high. The line continues
# the trend, and it goes too far in the other direction. Neither is right, and the tree gives
# no sign that the cars are outside of what it has seen.
light = features[:, weight] < 3000
L_train, L_test = features[light], features[~light]
m_train, m_test = mpg[light], mpg[~light]
tree = DecisionTreeRegressor(max_depth=5, min_samples=20).fit(L_train, m_train)
scaler = StandardScaler().fit(L_train)
line = LinearRegression().fit(scaler.transform(L_train), m_train)
print(f"{len(m_train)} light cars to train on, {len(m_test)} heavy cars to test on")
print(f"average mpg of the heavy cars: {m_test.mean():.1f}")
print(f"average prediction of the tree: {tree.predict(L_test).mean():.1f}")
print(f"average prediction of the line: {line.predict(scaler.transform(L_test)).mean():.1f}")
print(f"lowest mpg the tree can give:   {tree.predict(L_train).min():.1f}")
