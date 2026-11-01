"""
Standard Scaler
===============

:class:`~pyml.preprocessing.StandardScaler` puts every feature on the same scale. For each
feature it subtracts the mean and divides by the standard deviation, so that the feature ends up
with a mean of 0 and a standard deviation of 1. The numbers say how many standard deviations a
value is above or below the average.

Use it before any model that adds up differences between features or takes steps of one size in
all of them: the distance based models (:doc:`Neighbors </examples/neighbors/index>`) and the
models that are fit by gradient descent (:doc:`Linear Model </examples/linear-model/index>`).
Trees do not need it. The last section shows the mistake that is easiest to make with a scaler,
which is to fit it on the wrong data.
"""

# %%
# What it learns
# --------------
#
# The scaler has no parameters. ``fit`` computes the mean (``mean_``) and the standard deviation
# (``std_``) of each feature. ``transform`` applies them, to any data, and
# ``inverse_transform`` goes back to the original units. Unlike a model, it does not use a target.
#
# The :class:`API reference <pyml.preprocessing.StandardScaler>` has the details.

# %%
# Dataset
# -------
#
# Wines are described by 13 chemical measurements. They are on very different scales: the
# proline is measured in hundreds, and the hue in tenths.
#
# .. dataset-info:: wine

# %%
# Fit the scaler
# --------------
#
# The scaler learns from the training part of the data only. The test wines must be treated as
# new wines that arrive later, and a new wine is scaled with what was learned from the old ones.
import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_wine

from pyml.model_selection import train_test_split
from pyml.neighbors import KNNClassifier
from pyml.preprocessing import StandardScaler

data = load_wine()
names = list(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    np.asarray(data.X, dtype=np.float64),
    np.asarray(data.y, dtype=np.int64),
    test_size=0.2,
    random_state=42,
)
# train_test_split is typed for both kinds of target; say that this one is a class.
X_train, X_test = np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)
y_train, y_test = np.asarray(y_train, dtype=np.int64), np.asarray(y_test, dtype=np.int64)

scaler = StandardScaler()
scaler.fit(X_train)
scaled_train = scaler.transform(X_train)
scaled_test = scaler.transform(X_test)
print(f"proline: mean {scaler.mean_[-1]:.0f}, standard deviation {scaler.std_[-1]:.0f}")
print(f"hue:     mean {scaler.mean_[-3]:.2f}, standard deviation {scaler.std_[-3]:.2f}")

# %%
# The scale of each feature
# -------------------------
#
# The left figure shows the standard deviation of each feature in its own units, on a logarithmic
# axis. The proline varies about 2600 times more than the nonflavanoid phenols. The right figure
# shows the same features after the scaler: all of them are at 1.
positions = np.arange(len(names))
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout="constrained")
left.barh(positions, scaler.std_)
left.set_xscale("log")
left.set_yticks(positions, names)
left.invert_yaxis()
left.set_xlabel("standard deviation")
left.set_title("Before: from 0.1 to 300")
right.barh(positions, scaled_train.std(axis=0), color="C1")
right.set_xlabel("standard deviation")
right.set_title("After: all equal to 1")
plt.show()

# %%
# Why it matters for distances
# ----------------------------
#
# Take two wines from the training data and square the difference of each feature. A distance is
# the square root of the sum of these. On the raw features almost all of that sum comes from one
# feature, the proline, whatever the two wines are, and the other twelve are hardly counted.
# After the scaler each feature has a share of about one thirteenth. The two shares below are
# averages over 2000 pairs of wines.
rng = np.random.default_rng(0)
pairs = rng.integers(0, len(X_train), size=(2000, 2))
pairs = pairs[pairs[:, 0] != pairs[:, 1]]
raw_gap = (X_train[pairs[:, 0]] - X_train[pairs[:, 1]]) ** 2
scaled_gap = (scaled_train[pairs[:, 0]] - scaled_train[pairs[:, 1]]) ** 2
share_raw = np.mean(raw_gap[:, -1] / raw_gap.sum(axis=1))
share_scaled = np.mean(scaled_gap[:, -1] / scaled_gap.sum(axis=1))
print(f"share of the proline in the squared distance, raw:    {share_raw:.0%}")
print(f"share of the proline in the squared distance, scaled: {share_scaled:.0%}")

# %%
# The test wines and the way back
# -------------------------------
#
# The training data has a mean of exactly 0 and a standard deviation of exactly 1, because the
# scaler was fit on it. The test wines are scaled with the numbers of the training wines, so
# their means and standard deviations are close to 0 and 1, but not equal. This is correct, and
# it is how the model will see new wines. ``inverse_transform`` returns the original values, up
# to the rounding of the computer.
test_std = scaled_test.std(axis=0)
restored = scaler.inverse_transform(scaled_test)
print(f"largest mean of a training feature:  {np.abs(scaled_train.mean(axis=0)).max():.1e}")
print(f"largest mean of a test feature:      {np.abs(scaled_test.mean(axis=0)).max():.2f}")
print(f"standard deviation of a test feature: from {test_std.min():.2f} to {test_std.max():.2f}")
print(f"largest error after inverse_transform: {np.abs(restored - X_test).max():.1e}")

# %%
# A pitfall: fit on the data you predict
# --------------------------------------
#
# The scaler must be fit once, on the training data, and then only used to ``transform``. A common
# mistake is to fit it again on the wines that are being predicted, which happens naturally when
# wines arrive one at a time. The mean of one wine is the wine itself, and its standard deviation
# is 0, so every feature becomes 0 and all wines look the same. The model then gives the
# same answer for every one of them: it is right only for the share of the test wines that
# belong to that class.
model = KNNClassifier(n_neighbors=5).fit(scaled_train, y_train)
print(f"accuracy, scaler fit on the training wines: {model.score(scaled_test, y_test):.2f}")

alone = np.array(
    [
        model.predict(StandardScaler().fit(wine[None, :]).transform(wine[None, :]))[0]
        for wine in X_test
    ]
)
print(f"accuracy, scaler fit on each wine alone:    {np.mean(alone == y_test):.2f}")
print(f"classes that the model predicted: {sorted(set(alone.tolist()))}")

# %%
# Fitting the scaler on the training and the test wines together is a milder form of the same
# mistake. The test wines then take part in the choice of the numbers, and the evaluation is no
# longer about wines that the model has not seen. Here the means move by less than 0.04 standard
# deviations, so the accuracy does not change, but with a small data set or an outlier in the test
# part the difference can be large.
together = StandardScaler().fit(np.vstack([X_train, X_test]))
shift = np.abs((together.mean_ - scaler.mean_) / scaler.std_).max()
leaky = KNNClassifier(n_neighbors=5).fit(together.transform(X_train), y_train)
print(f"largest shift of a mean, in standard deviations: {shift:.3f}")
leaky_score = leaky.score(together.transform(X_test), y_test)
print(f"accuracy, scaler fit on training and test wines: {leaky_score:.2f}")
