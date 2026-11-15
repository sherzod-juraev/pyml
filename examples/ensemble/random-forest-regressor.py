"""
Random Forest Regressor
=======================

:class:`~pyml.ensemble.RandomForestRegressor` predicts a number by asking many decision trees and
taking the average of their answers. Each tree is grown on a different random sample of the
training data, and at every question it looks at only some of the features, so the trees make
different mistakes. The average smooths them out: a single
:class:`~pyml.tree.DecisionTreeRegressor` is a staircase that follows the training data closely,
and the forest is a staircase with many more, smaller steps.

Use it when the target depends on the features in a way that is not a straight line, and a single
tree is too unsteady to trust. It needs no scaling of the features, and it does not have to be
tuned as carefully as a single tree. Like a tree, it can only answer with averages of the targets
that it has seen, so it cannot continue a trend. The last sections show both sides.
"""

# %%
# Parameters
# ----------
#
# * ``max_depth`` is the largest number of questions in a row in every tree. It is required, as in
#   the :class:`~pyml.tree.DecisionTreeRegressor`. A forest can use deep trees, because the average
#   evens out the noise that a single deep tree fits.
# * ``n_estimators`` is the number of trees.
# * ``max_features`` is how many features a tree looks at for each question. They are drawn at
#   random for every question, so that the trees do not all ask the same first question.
# * ``bootstrap`` decides whether each tree is grown on a sample of the rows, drawn with
#   replacement, or on all of them.
# * ``random_state`` fixes the draws, so that the same forest is grown every time.
# * ``min_samples`` and ``min_impurity_decrease`` are passed to every tree, and mean there what
#   they mean in the :class:`~pyml.tree.DecisionTreeRegressor`.
#
# The :class:`API reference <pyml.ensemble.RandomForestRegressor>` has the default values.

# %%
# Dataset
# -------
#
# The task is to predict the fuel consumption of a car, in miles per gallon, from its engine, its
# weight and its year. A few cars have no value for the horsepower, and a tree cannot use a
# missing number, so those rows are left out. ``cylinders`` and ``origin`` are whole numbers that
# stand for categories, and a tree can use them, because it only asks whether a value is above or
# below a cut. It is the same task as in the :class:`~pyml.tree.DecisionTreeRegressor` example, so
# that a forest and a single tree can be compared on the same cars.
#
# .. dataset-info:: auto_mpg

# %%
# Split the data
# --------------
#
# The model is judged on cars it has not seen, so 20 percent of the data is kept for the test.
# The trees ask about one feature at a time, so the features are used as they are, without scaling.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_auto_mpg

from pyml.ensemble import RandomForestRegressor
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


X_train, X_test, y_train, y_test = split(42)
print(f"{len(y_train)} cars to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``fit`` grows the trees, and ``score`` is the R squared: 1 is perfect, and 0 is no better than
# always predicting the mean. The forest has 100 trees of depth 8, and the single tree next to it
# has the same depth and the same ``min_samples``. A line fit on standardized features is the
# usual first model for this task, so it is fit as well, for comparison.
tree = DecisionTreeRegressor(max_depth=8, min_samples=5)
tree.fit(X_train, y_train)
forest = RandomForestRegressor(max_depth=8, min_samples=5, n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
scaler = StandardScaler().fit(X_train)
line = LinearRegression().fit(scaler.transform(X_train), y_train)
print(f"R squared on the training cars, tree:   {tree.score(X_train, y_train):.2f}")
print(f"R squared on the training cars, forest: {forest.score(X_train, y_train):.2f}")
print(f"R squared on the test cars, tree:   {tree.score(X_test, y_test):.2f}")
print(f"R squared on the test cars, forest: {forest.score(X_test, y_test):.2f}")
print(f"R squared on the test cars, line:   {line.score(scaler.transform(X_test), y_test):.2f}")

# %%
# On the training cars both models are almost perfect (0.98 and 0.97), but on the test cars the
# tree falls to 0.69, below the line at 0.77, and the forest gets 0.85. The tree is deep enough to
# follow the noise of the training cars, and the forest uses trees of the same depth. What it adds
# is the average. The 78 test cars are a small test, so the next sections repeat the comparison on
# five splits.

# %%
# The forest does not need a careful depth
# ----------------------------------------
#
# The figure shows the R squared on the training cars (dashed) and on the test cars (solid) for
# trees and forests of growing depth, averaged over five splits. The test R squared of a single
# tree rises until depth 4 and then falls, as the tree starts to follow single cars. The test
# R squared of the forest keeps rising with the depth, because every tree is deep and the average
# of many different deep trees is steadier than any one of them.
depths = [2, 4, 8, 30]
scores: dict[str, NDArray[np.float64]] = {
    name: np.zeros(len(depths))
    for name in ("tree train", "tree test", "forest train", "forest test")
}
for seed in range(5):
    A_train, A_test, b_train, b_test = split(seed)
    for position, depth in enumerate(depths):
        grown_tree = DecisionTreeRegressor(max_depth=depth).fit(A_train, b_train)
        grown_forest = RandomForestRegressor(max_depth=depth, n_estimators=30, random_state=0)
        grown_forest.fit(A_train, b_train)
        scores["tree train"][position] += grown_tree.score(A_train, b_train) / 5
        scores["tree test"][position] += grown_tree.score(A_test, b_test) / 5
        scores["forest train"][position] += grown_forest.score(A_train, b_train) / 5
        scores["forest test"][position] += grown_forest.score(A_test, b_test) / 5

fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
ax.plot(depths, scores["tree test"], marker="o", color="C0", label="Tree, test cars")
ax.plot(depths, scores["tree train"], marker="o", color="C0", linestyle="--", alpha=0.6)
ax.plot(depths, scores["forest test"], marker="o", color="C1", label="Forest, test cars")
ax.plot(depths, scores["forest train"], marker="o", color="C1", linestyle="--", alpha=0.6)
ax.set_xscale("log")
ax.set_xticks(depths, [str(depth) for depth in depths])
ax.set_xlabel("max_depth")
ax.set_ylabel("R squared (mean of 5 splits)")
ax.set_title("The forest keeps improving with depth")
ax.legend(loc="center right")
plt.show()

# %%
# The R squared of the tree on the test cars is 0.64, 0.79, 0.75 and 0.73 for the depths 2, 4, 8
# and 30, while its R squared on the training cars goes to 1.00. The forest scores 0.78, 0.83, 0.86
# and 0.87, and it is better than the tree at every depth in the figure, even better than the best
# tree (0.79 at depth 4). Its R squared on the training cars is 0.98 at depth 30, so the deep
# forest also fits the training cars closely. It does not pay for it on the test cars. This makes
# ``max_depth`` a much safer choice than in a single tree: it has to be deep enough, and there is
# no narrow best value to look for.

# %%
# How many features for each question
# -----------------------------------
#
# ``max_features`` is the other choice that shapes a forest. With 1 feature the question of every
# node is drawn blindly, and the trees differ a lot, but each one is weak. With all 7 the trees
# ask the best question that exists, and they become alike, so that averaging them gains less. The
# default, ``"sqrt"``, is the square root of the number of features rounded down, which is 2 here.
# The figure uses forests of 25 trees of depth 8, averaged over five splits.
feature_counts = [1, 2, 7]
by_features = np.zeros(len(feature_counts))
for seed in range(5):
    A_train, A_test, b_train, b_test = split(seed)
    for position, count in enumerate(feature_counts):
        few = RandomForestRegressor(
            max_depth=8, n_estimators=25, max_features=count, random_state=0
        )
        by_features[position] += few.fit(A_train, b_train).score(A_test, b_test) / 5

fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
ax.plot(feature_counts, by_features, marker="o")
ax.set_xticks(feature_counts, ["1", "2 (sqrt)", "7 (all)"])
ax.set_xlabel("max_features")
ax.set_ylabel("test R squared (mean of 5 splits)")
ax.set_title("A few features per question are enough")
plt.show()

# %%
# One feature per question is the worst of the three, with a test R squared of 0.84. Two features,
# the default, give 0.87, and all seven give 0.86, so the gap between the last two is small and
# ``max_features`` is not a choice that needs much care here. A smaller number also makes the
# trees faster to grow, since every question looks at fewer features. For regression a third of
# the features is a common choice as well, which would be 2 of 7 here, so both rules agree.

# %%
# A pitfall: a new range
# ----------------------
#
# A forest, like a tree, can only answer with averages of the targets it has seen, so it cannot
# continue a trend beyond the training data. Here the models are trained on the light cars (below
# 3000 pounds) and tested on the heavy ones. The lowest answer that a forest can give is an
# average of the mpg of light cars, 19.8 mpg, while the heavy cars average 16.8. The forest
# predicts 21.6 on average, so its predictions for the heavy cars are too high. The line continues
# the trend, and it goes too far in the other direction, to 11.1. Averaging many trees does not
# change this, and the forest gives no sign that the cars are outside of what it has seen.
weight = names.index("weight")
light = features[:, weight] < 3000
L_train, L_test = features[light], features[~light]
m_train, m_test = mpg[light], mpg[~light]
light_forest = RandomForestRegressor(
    max_depth=5, min_samples=20, n_estimators=100, random_state=0
).fit(L_train, m_train)
light_scaler = StandardScaler().fit(L_train)
light_line = LinearRegression().fit(light_scaler.transform(L_train), m_train)
print(f"{len(m_train)} light cars to train on, {len(m_test)} heavy cars to test on")
print(f"average mpg of the heavy cars: {m_test.mean():.1f}")
print(f"average prediction of the forest: {light_forest.predict(L_test).mean():.1f}")
line_mean = light_line.predict(light_scaler.transform(L_test)).mean()
print(f"average prediction of the line:   {line_mean:.1f}")
print(f"lowest mpg the forest can give:   {light_forest.predict(L_train).min():.1f}")
