"""
Lasso Classifier
================

:class:`~pyml.linear_model.LassoClassifier` is logistic regression with an L1 penalty: the loss
grows with the sum of the absolute values of the weights, and the parameter ``alpha`` says how
much that counts. The L1 penalty has a useful habit. As ``alpha`` grows, the weights of the
features that matter little go to zero one after another, so the model also chooses which
features to use.

Use it when there are many features and you want a short list of the ones that carry the
decision, or a simpler model that is easier to explain. The price is a bias, and a stronger
penalty lowers the accuracy. With gradient descent the weights come close to zero and do not
reach it, so the examples below call a weight "dropped" when its size is below a small limit.
The last section shows why.
"""

# %%
# Parameters
# ----------
#
# * ``alpha`` is the strength of the penalty. With ``alpha=0`` the model is logistic regression,
#   and the larger it is, the more weights go to zero.
# * ``learning_rate``, ``max_iter``, ``tol`` and ``fit_intercept`` are the same as in
#   :class:`~pyml.linear_model.LogisticRegression`. The weights of an L1 model keep moving
#   toward zero slowly, so the number of iterations changes the result more than it does for the
#   other linear models. The examples below fix it with ``tol=0``, which makes the loop always
#   run ``max_iter`` steps. The last section shows why.
#
# The labels must be 0 and 1. The full list of parameters, with the default values, is in the
# :class:`API reference <pyml.linear_model.LassoClassifier>`.

# %%
# Dataset
# -------
#
# The task is to tell a malignant tumor from a benign one using 30 measurements of cell nuclei
# from a breast biopsy. The label ``0`` is malignant and ``1`` is benign. Many of the features
# are near copies of each other, which makes it a good case for choosing a few.
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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_breast_cancer

from pyml.linear_model import LassoClassifier
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


def fit_lasso(alpha: float, features: Any, labels: NDArray[np.int64]) -> LassoClassifier:
    """Fit a lasso classifier, always with 5000 steps."""
    model = LassoClassifier(alpha=alpha, learning_rate=0.1, max_iter=5000, tol=0)
    return model.fit(features, labels)


A_train, A_test, y_train, y_test = split(42)
print(f"{len(y_train)} tumors to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``coef_`` holds one weight per feature. Weights smaller than ``negligible = 0.1`` in size are
# counted as dropped. With ``alpha=0.03`` most of the 30 features are dropped, and the accuracy
# is about that of the plain logistic regression on the same split (0.965). The weights
# are negative because the model estimates the probability of benign, and larger nuclei point to
# malignant.
model = fit_lasso(0.03, A_train, y_train)
kept = np.abs(model.coef_) >= negligible
print(f"iterations: {model.n_iter_}")
print(f"features kept: {kept.sum()} of {len(kept)}")
print(f"test accuracy: {model.score(A_test, y_test):.3f}")

order = np.argsort(model.coef_[kept])
positions = np.arange(kept.sum())
kept_names = [name for name, keep in zip(names, kept, strict=True) if keep]

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.38)
ax.barh(positions, model.coef_[kept][order], color="C0")
ax.set_yticks(positions, [kept_names[i] for i in order])
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("weight (log-odds per standard deviation)")
ax.set_title(f"The {kept.sum()} features the model keeps")
plt.show()

# %%
# Choose ``alpha``
# ----------------
#
# Fitting for a range of ``alpha`` shows the trade. The left figure counts the features that are
# kept and the right figure the accuracy. One split of 113 tumors is a small test, so both are
# averaged over ten different splits. At ``alpha=0.03`` the accuracy is lower by about one
# percentage point than with the weakest penalty, and only about six features are left. At 0.1
# three are left and the accuracy falls further, and at 0.3 no feature is left and the model
# only says the more common class.
alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
splits = [split(seed) for seed in range(10)]
kept_count = np.zeros(len(alphas))
test_accuracy = np.zeros(len(alphas))
for a_train, a_test, labels_train, labels_test in splits:
    for position, alpha in enumerate(alphas):
        fitted = fit_lasso(alpha, a_train, labels_train)
        kept_count[position] += np.sum(np.abs(fitted.coef_) >= negligible) / len(splits)
        test_accuracy[position] += fitted.score(a_test, labels_test) / len(splits)

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
left.plot(alphas, kept_count, marker="o", color="C0")
left.set_xscale("log")
left.set_xlabel("alpha")
left.set_ylabel("features kept (mean of 10 splits)")
left.set_title("The features go one by one")
right.plot(alphas, test_accuracy, marker="o", color="C1")
right.set_xscale("log")
right.set_xlabel("alpha")
right.set_ylabel("test accuracy (mean of 10 splits)")
right.set_title("The accuracy falls later")
fig.tight_layout()
plt.show()

# %%
# Which features stay
# -------------------
#
# When the penalty is strong, the features that remain are the ones the model cannot do without.
# At ``alpha=0.1`` three are left, all of them about the size and the shape of the nuclei. Three
# measurements are enough for a test accuracy of about 92 percent on this split.
strong = fit_lasso(0.1, A_train, y_train)
for index in np.argsort(strong.coef_):
    if abs(strong.coef_[index]) >= negligible:
        print(f"{names[index]:<22} {strong.coef_[index]:>6.2f}")
print(f"test accuracy with these features: {strong.score(A_test, y_test):.3f}")

# %%
# A pitfall: dropped is not zero, and the count depends on the steps
# -------------------------------------------------------------------
#
# Gradient descent with an L1 penalty moves a weight toward zero and then steps back over it, so
# the weights of the dropped features are small and not exactly zero. Code that counts the
# nonzero weights, such as ``np.count_nonzero(model.coef_)``, finds all 30 of them. Decide with
# a limit, as above.
#
# The weights also keep shrinking for a long time. The loss is almost flat by then, so the
# default stopping rule ends the loop early, with more features still above the limit than a
# longer run would leave. The table runs the same model for a different number of steps. The
# accuracy hardly changes, but the list of kept features does, and the weights near the limit
# can fall on either side of it. Fix the number of steps, as the examples here do, and report it
# with the result.
print("nonzero weights:", np.count_nonzero(model.coef_), "of", len(model.coef_))
print("steps   kept  test accuracy")
for steps in (100, 300, 1000, 5000, 20000):
    shorter = LassoClassifier(alpha=0.03, learning_rate=0.1, max_iter=steps, tol=0)
    shorter.fit(A_train, y_train)
    count = np.sum(np.abs(shorter.coef_) >= negligible)
    print(f"{steps:<7} {count:>4}  {shorter.score(A_test, y_test):.3f}")
