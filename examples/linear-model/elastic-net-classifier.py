"""
Elastic Net Classifier
======================

:class:`~pyml.linear_model.ElasticNetClassifier` is logistic regression with both penalties of
:class:`~pyml.linear_model.RidgeClassifier` and :class:`~pyml.linear_model.LassoClassifier` at
once. The L1 part sets weights to zero and chooses features. The L2 part shrinks the weights
and, when several features carry the same information, makes them share the weight instead of
leaving it all to one of them. The parameter ``l1_ratio`` sets the mix: ``0`` is pure L2 and
``1`` is pure L1.

Use it when there are many correlated features and you still want a smaller model. The lasso
alone picks one of a group of near copies, and which one it picks can change from one sample to
another. The elastic net keeps the group.
"""

# %%
# Parameters
# ----------
#
# * ``alpha`` is the strength of the whole penalty.
# * ``l1_ratio`` is the share of the L1 penalty, between 0 and 1.
# * ``learning_rate``, ``max_iter``, ``tol`` and ``fit_intercept`` are the same as in
#   :class:`~pyml.linear_model.LogisticRegression`. As in the
#   :doc:`Lasso Classifier <lasso-classifier>` example, the weights keep moving toward zero
#   slowly, so the examples below fix the number of steps with ``tol=0``.
#
# The labels must be 0 and 1. The full list of parameters, with the default values, is in the
# :class:`API reference <pyml.linear_model.ElasticNetClassifier>`.

# %%
# Dataset
# -------
#
# The task is to tell a malignant tumor from a benign one using 30 measurements of cell nuclei
# from a breast biopsy. The label ``0`` is malignant and ``1`` is benign. Radius, perimeter and
# area appear three times each (the mean, the worst and the error), and inside each group they
# are close to the same number.
#
# .. dataset-info:: breast_cancer

# %%
# Split and standardize
# ---------------------
#
# The penalty treats all weights alike, so the features must be on the same scale, and they
# are standardized. The scaler learns its mean and spread from the training part only, so no
# information from the test part leaks into the model. ``split`` repeats these steps for any
# random seed, the later sections need it.
# The card of this example shows the second figure. Every figure is saved twice (light, dark), so
# the number of the light copy of figure 2 is 3.
# sphinx_gallery_thumbnail_number = 3
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_breast_cancer

from pyml.linear_model import ElasticNetClassifier, LassoClassifier, RidgeClassifier
from pyml.model_selection import train_test_split
from pyml.preprocessing import StandardScaler

data = load_breast_cancer()
names = data.feature_names
negligible = 0.1


def split(seed: int) -> tuple[Any, Any, NDArray[np.int64], NDArray[np.int64]]:
    """Return the standardized train and test features and the labels of one split."""
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        data.X, data.y, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a class label.
    y_train = np.asarray(y_train_raw, dtype=np.int64)
    y_test = np.asarray(y_test_raw, dtype=np.int64)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def fit_elastic(alpha: float, l1_ratio: float, features: Any, labels: Any) -> ElasticNetClassifier:
    """Fit an elastic net classifier, always with 5000 steps."""
    model = ElasticNetClassifier(
        alpha=alpha, l1_ratio=l1_ratio, learning_rate=0.1, max_iter=5000, tol=0
    )
    return model.fit(features, labels)


A_train, A_test, y_train, y_test = split(42)
print(f"{len(y_train)} tumors to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# With ``alpha=0.03`` and an even mix, ``l1_ratio=0.5``, the model keeps half of
# the features. Weights smaller than ``negligible = 0.1`` in size are counted as dropped, as
# in the lasso example, because gradient descent brings the weights close to zero and does not
# put them exactly there.
model = fit_elastic(0.03, 0.5, A_train, y_train)
kept = np.abs(model.coef_) >= negligible
print(f"features kept: {kept.sum()} of {len(kept)}")
print(f"test accuracy: {model.score(A_test, y_test):.3f}")

# %%
# From ridge to lasso
# -------------------
#
# Moving ``l1_ratio`` from 0 to 1 moves the model from the ridge classifier to the lasso
# classifier. The left figure counts the features that are kept and the right figure the
# accuracy, both averaged over ten splits because one split of 113 tumors is a small test.
# With more L1 the model keeps fewer features, and the accuracy gives up a little: the whole
# range is less than one percentage point. At ``l1_ratio=0.5`` it keeps about half of the
# features and is within a few tenths of a point of the best accuracy.
ratios = [0, 0.25, 0.5, 0.75, 1]
splits = [split(seed) for seed in range(10)]
kept_count = np.zeros(len(ratios))
test_accuracy = np.zeros(len(ratios))
for a_train, a_test, labels_train, labels_test in splits:
    for position, ratio in enumerate(ratios):
        fitted = fit_elastic(0.03, ratio, a_train, labels_train)
        kept_count[position] += np.sum(np.abs(fitted.coef_) >= negligible) / len(splits)
        test_accuracy[position] += fitted.score(a_test, labels_test) / len(splits)

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
left.plot(ratios, kept_count, marker="o", color="C0")
left.set_xlabel("l1_ratio")
left.set_ylabel("features kept (mean of 10 splits)")
left.set_title("More L1, fewer features")
right.plot(ratios, test_accuracy, marker="o", color="C1")
right.set_xlabel("l1_ratio")
right.set_ylabel("test accuracy (mean of 10 splits)")
right.set_title("The accuracy gives up a little")
fig.tight_layout()
plt.show()

# %%
# Correlated features
# -------------------
#
# The six features below all measure the size of a nucleus. The figure shows their weights for
# three values of ``l1_ratio``. The lasso (L1 only) puts the weight on two of them,
# ``worst radius`` and ``worst perimeter``, and drops the other four. With an even mix the weight
# is shared by the whole group. Which ones the lasso picks is a matter of small differences in
# the data, and the elastic net does not have to bet on them.
group = [
    names.index(name)
    for name in (
        "mean radius",
        "mean perimeter",
        "mean area",
        "worst radius",
        "worst perimeter",
        "worst area",
    )
]
positions = np.arange(len(group))
settings = [
    (0, "C0", "l1_ratio=0 (ridge)"),
    (0.5, "C1", "l1_ratio=0.5"),
    (1, "C2", "l1_ratio=1 (lasso)"),
]

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.3, bottom=0.3)
for shift, (ratio, color, label) in zip((0.27, 0, -0.27), settings, strict=True):
    weights = fit_elastic(0.03, ratio, A_train, y_train).coef_[group]
    ax.barh(positions + shift, weights, height=0.25, color=color, label=label)
ax.set_yticks(positions, [names[i] for i in group])
ax.invert_yaxis()
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("weight (log-odds per standard deviation)")
ax.set_title("The weight is shared by the group")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.show()

# %%
# The two ends
# ------------
#
# The ends of the scale are the other two classifiers. With ``l1_ratio=1`` the elastic net is the
# lasso classifier with the same ``alpha``. With ``l1_ratio=0`` it is the ridge classifier with
# half the ``alpha``, because the L2 part of the penalty is written with a factor of one half.
# Both are exact: the weights are the same.
ridge = RidgeClassifier(alpha=0.015, learning_rate=0.1, max_iter=5000, tol=0).fit(A_train, y_train)
lasso = LassoClassifier(alpha=0.03, learning_rate=0.1, max_iter=5000, tol=0).fit(A_train, y_train)
pure_l2 = fit_elastic(0.03, 0, A_train, y_train)
pure_l1 = fit_elastic(0.03, 1, A_train, y_train)
print("l1_ratio=0 against Ridge(alpha/2):", np.abs(pure_l2.coef_ - ridge.coef_).max())
print("l1_ratio=1 against Lasso:         ", np.abs(pure_l1.coef_ - lasso.coef_).max())

# %%
# A pitfall: two knobs
# --------------------
#
# ``alpha`` and ``l1_ratio`` act together, and the same ``alpha`` is a different strength at a
# different ``l1_ratio``. An ``alpha`` taken from the ridge example does not carry over to the
# elastic net: ``l1_ratio=0`` with ``alpha=0.03`` is a ridge of 0.015, as shown above, and the
# same ``alpha`` keeps 24 features at ``l1_ratio=0`` and only 7 at ``l1_ratio=1``. Choose one of
# the two first, here ``l1_ratio`` from how many features you want to keep, and then search
# ``alpha`` with it fixed, on several splits as in the earlier examples.
for ratio in (0, 0.5, 1):
    fitted = fit_elastic(0.03, ratio, A_train, y_train)
    count = np.sum(np.abs(fitted.coef_) >= negligible)
    print(f"alpha=0.03, l1_ratio={ratio}: {count} features kept")
