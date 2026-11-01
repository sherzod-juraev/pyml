"""
Min-Max Scaler
==============

:class:`~pyml.preprocessing.MinMaxScaler` squeezes every feature into the range from 0 to 1. The
smallest value of a feature in the training data becomes 0, the largest becomes 1, and the values
in between keep their place on that line. The shape of the distribution does not change, only its
position and its width.

Use it when the features have natural limits and you want the result to be bounded, for example
for the pixels of an image, or for a model that expects inputs between 0 and 1. It has one
weakness: the range is set by the two most extreme values, so a single outlier can squeeze all
the other values into a small part of the interval. The last section shows this.
"""

# %%
# What it learns
# --------------
#
# The scaler has no parameters. ``fit`` finds the smallest value (``min_``) and the range
# (``range_``, the largest value minus the smallest) of each feature. ``transform`` subtracts
# the smallest value and divides by the range, and ``inverse_transform`` goes back to the
# original units.
#
# The :class:`API reference <pyml.preprocessing.MinMaxScaler>` has the details.

# %%
# Dataset
# -------
#
# Abalone are described by seven measurements of the size and the weight of the animal. The first
# column of the data, the sex, is a category stored as a code, and it is left out. The measurements
# are in different units, and their ranges differ: the length goes up to 0.8, the whole weight to
# 2.8.
#
# .. dataset-info:: abalone

# %%
# Fit the scaler
# --------------
#
# As with every scaler, ``fit`` sees the training part of the data only, and ``transform`` is
# then used on the training and the test part.
import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_abalone

from pyml.model_selection import train_test_split
from pyml.preprocessing import MinMaxScaler

data = load_abalone()
names = list(data.feature_names)[1:]
features = np.asarray(data.X, dtype=np.float64)[:, 1:]
rings = np.asarray(data.y, dtype=np.float64)
X_train, X_test, _, _ = train_test_split(features, rings, test_size=0.2, random_state=42)
X_train, X_test = np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)

scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_train = scaler.transform(X_train)
scaled_test = scaler.transform(X_test)
print(f"training data: from {scaled_train.min():.2f} to {scaled_train.max():.2f}")
print(f"test data:     from {scaled_test.min():.3f} to {scaled_test.max():.3f}")

# %%
# The training data fills the interval from 0 to 1, as it must. The test data comes close to it, but
# not exactly: a test abalone that is larger than any training abalone is scaled to more than 1,
# and a smaller one to less than 0. The scaler does not clip them. Here two values are slightly
# above 1. With another split the largest abalone can fall in the test part, and then its height
# is scaled to more than 2.
outside = np.sum((scaled_test < 0) | (scaled_test > 1))
print(f"values of the test data outside the interval: {outside} of {scaled_test.size}")

# %%
# Where the data lies in the interval
# -----------------------------------
#
# The figure shows, for each feature, the interval that holds the middle half of the training
# abalone after scaling. For six of the features it is about 0.2 wide. The height is different:
# the middle half is squeezed into a width of 0.04, close to the bottom of the interval.
positions = np.arange(len(names))
low, high = np.percentile(scaled_train, [25, 75], axis=0)
fig, ax = plt.subplots(layout="constrained")
ax.barh(positions, high - low, left=low)
ax.set_yticks(positions, names)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("scaled value, middle half of the training abalone")
ax.set_title("The height uses a small part of the interval")
plt.show()

# %%
# A pitfall: one outlier sets the range
# -------------------------------------
#
# The height of two abalone is far from all the others: 0.515 and 1.13, when all the other
# abalone are 0.25 or lower. These may well be mistakes in the data. As the largest of them sets
# the range, the other abalone take up only the first fifth of the interval, and the differences
# between them, which are the useful ones, become small numbers. The left figure shows the
# heights after scaling. The right figure shows them when the scaler is fit without the two
# abalone: the heights now use the whole interval. (Both figures use all the abalone, to
# show the effect of the outliers on its own.)
height = names.index("height")
typical = features[:, height] < 0.3
with_outliers = MinMaxScaler().fit(features)
without = MinMaxScaler().fit(features[typical])
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, layout="constrained")
left.hist(with_outliers.transform(features)[:, height], bins=40, range=(0, 1))
left.set_xlabel("scaled height")
left.set_ylabel("abalone")
left.set_title("With the outliers")
right.hist(without.transform(features[typical])[:, height], bins=40, range=(0, 1), color="C1")
right.set_xlabel("scaled height")
right.set_title("Without the two outliers")
plt.show()

# %%
# The middle half of the heights is 0.04 wide in the first case and 0.20 wide in the second. A
# model that compares features by distance sees almost no difference in height between a small
# and a large abalone in the first case. If your data has outliers, remove them first, or use
# :class:`~pyml.preprocessing.RobustScaler`, which does not depend on the extremes.
h_low, h_high = np.percentile(with_outliers.transform(features)[:, height], [25, 75])
w_low, w_high = np.percentile(without.transform(features[typical])[:, height], [25, 75])
print(f"width of the middle half of the heights, with the outliers:    {h_high - h_low:.2f}")
print(f"width of the middle half of the heights, without the outliers: {w_high - w_low:.2f}")
print(f"abalone taller than 0.3: {np.sum(~typical)}")
