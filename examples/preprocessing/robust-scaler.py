"""
Robust Scaler
=============

:class:`~pyml.preprocessing.RobustScaler` puts the features on a common scale in a way that
extreme values hardly affect. For each feature it subtracts the median and divides by the
interquartile range, the distance between the value that 25 percent of the samples are below and
the value that 75 percent are below. Both numbers depend only on the middle of the data, so a few
very large or very small values do not change them.

Use it when a feature has a long tail or some outliers that you want to keep. The other two
scalers take the extremes into account, :class:`~pyml.preprocessing.StandardScaler` through the
mean and the standard deviation, and :class:`~pyml.preprocessing.MinMaxScaler` through the
smallest and the largest value. The last section shows a case where this scaler fails: a feature
that is the same for more than half of the samples.
"""

# %%
# What it learns
# --------------
#
# The scaler has no parameters. ``fit`` computes the median (``q2_``) and the interquartile
# range (``iqr_``) of each feature. ``transform`` subtracts the median and divides by the range,
# and ``inverse_transform`` goes back to the original units.
#
# The :class:`API reference <pyml.preprocessing.RobustScaler>` has the details.

# %%
# Dataset
# -------
#
# The data describes the customers of a bank that were called to offer a term deposit. One of the
# features is the balance of the account of the customer. Most balances are a few hundred
# euros, and a few are very large, or negative.
#
# .. dataset-info:: bank_marketing

# %%
# Compare the three scalers
# -------------------------
#
# Each scaler is fit on the balance of the training customers. The figure shows the scaled
# balances of all three on the same axis, between -3 and 3. A balance of 445 euros is the median,
# and the middle half of the customers lies between about 70 and 1400 euros.
#
# * With the standard scaler the middle half takes only 0.44 of the axis, because the standard
#   deviation is made large by the few very large balances.
# * With the min-max scaler it takes 0.012: the largest balance, more than 100 000 euros, becomes
#   1, and almost every customer is squeezed to the bottom of the range.
# * With the robust scaler it takes exactly 1, whatever the extremes are.
import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_bank_marketing

from pyml.model_selection import train_test_split
from pyml.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

data = load_bank_marketing()
names = list(data.feature_names)
X_train, X_test, _, _ = train_test_split(
    np.asarray(data.X, dtype=np.float64), np.asarray(data.y), test_size=0.2, random_state=42
)
X_train, X_test = np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)
balance = names.index("balance")
balances = X_train[:, [balance]]

fig, ax = plt.subplots(layout="constrained")
for label, scaler_class in (
    ("Standard", StandardScaler),
    ("Min-max", MinMaxScaler),
    ("Robust", RobustScaler),
):
    scaled = scaler_class().fit(balances).transform(balances)[:, 0]
    low, high = np.percentile(scaled, [25, 75])
    ax.hist(scaled, bins=120, range=(-3, 3), histtype="step", linewidth=2, label=label)
    width = high - low
    print(
        f"{label:>8}: middle half {width:.3f} wide, from {scaled.min():.2f} to {scaled.max():.1f}"
    )
ax.set_yscale("log")
ax.set_xlabel("scaled balance")
ax.set_ylabel("customers (logarithmic)")
ax.set_title("The robust scaler gives the middle half a width of 1")
ax.legend()
plt.show()

# %%
# The shape of the distribution is the same in the three cases, because every scaler is a straight
# line. What differs is the number that the line is made of. The robust scaler is the only one
# whose result does not depend on how far the largest balance is.
#
# It does not remove the outliers. The largest balance is still 75 times the width of the middle
# half away from the median. The scaler only keeps the outliers from setting the scale for the
# customers that are not outliers.
scaler = RobustScaler().fit(X_train)
print(f"median balance: {scaler.q2_[balance]:.0f} euros")
print(f"interquartile range of the balance: {scaler.iqr_[balance]:.0f} euros")

# %%
# A pitfall: a feature that is the same for most customers
# ---------------------------------------------------------
#
# The interquartile range is 0 when more than half of the samples have the same value. In
# ``pdays``, the number of days since the customer was last contacted, 82 percent of the
# customers have the value -1 (never contacted before). The median is -1 and the range is 0. The
# scaler adds a very small number to the range to avoid dividing by zero, and the result is that
# the customers who were contacted get values in the trillions.
pdays = names.index("pdays")
scaled_train = scaler.transform(X_train)
print(f"customers with pdays = -1: {np.mean(X_train[:, pdays] == -1):.0%}")
print(f"largest scaled pdays: {scaled_train[:, pdays].max():.1e}")

# %%
# This is not only one feature. The lines below list all the features for which the range is 0,
# and the largest absolute value that the scaler gives to each. All five are huge, and a model
# that adds up differences would see only these features. For ``poutcome``, whose most common
# value 3 is also its largest, the huge values are negative.
flat = np.flatnonzero(scaler.iqr_ == 0)
for index in flat:
    largest_value = np.abs(scaled_train[:, index]).max()
    print(f"{names[index]:>9}: largest absolute scaled value {largest_value:.1e}")

# %%
# The fix is to leave such features out of the scaler. Features that have only a few values, or
# that are the same for most samples, do not have outliers to guard against, and they can be
# used as they are or scaled by another scaler. Here the scaler is fit on the other 11 features
# only, and the values of the test customers stay small. The largest one is a balance.
others = [index for index in range(len(names)) if index not in flat]
better = RobustScaler().fit(X_train[:, others])
largest = np.abs(better.transform(X_test[:, others])).max(axis=0)
print(f"largest absolute value of the {len(others)} other features: {largest.max():.0f}")
