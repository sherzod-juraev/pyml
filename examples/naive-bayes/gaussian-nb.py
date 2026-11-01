"""
Gaussian Naive Bayes
====================

:class:`~pyml.naive_bayes.GaussianNB` classifies with Bayes' rule. For a new sample it asks, for
every class, how likely the class is before looking at the sample (the prior), and how likely
the measured values are if the sample belongs to that class (the likelihood). It picks the class
where the product is the largest. To make the likelihood simple it makes two assumptions: inside
a class every feature follows a normal distribution, and the features are independent of each
other. The second one is the "naive" part, and it is rarely true.

Use it as a fast baseline. Fitting is one pass that computes a mean and a variance per class and
feature, there is nothing to iterate or to tune, and it works with few samples. The last
sections show what the independence assumption costs.
"""

# %%
# Parameters
# ----------
#
# :class:`~pyml.naive_bayes.GaussianNB` has no parameters to set. The fit is a closed formula, so
# there is no ``learning_rate``, ``max_iter`` or ``tol``, and the features do not have to be
# standardized, because each feature is modeled on its own. All the work is in the assumptions.
#
# The :class:`API reference <pyml.naive_bayes.GaussianNB>` has the details.

# %%
# Dataset
# -------
#
# The task is to tell the species of an iris flower from the length and the width of its sepals
# and petals.
#
# .. dataset-info:: iris

# %%
# Split the data
# --------------
#
# The model is judged on flowers it has not seen, so 20 percent of the data is kept for the
# test. That is only 30 flowers, and one of them is more than three percentage points of
# accuracy, so the later sections repeat the experiment on several splits where the number needs
# it.
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_iris

from pyml.model_selection import train_test_split
from pyml.naive_bayes import GaussianNB

data = load_iris()
names = data.feature_names
species = tuple(data.target_names or ())


def split(
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Return the train and test features and labels of one split."""
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        data.X, data.y, test_size=0.2, random_state=seed
    )
    # train_test_split is typed for both kinds of target; say that this one is a class label.
    y_train = np.asarray(y_train_raw, dtype=np.int64)
    y_test = np.asarray(y_test_raw, dtype=np.int64)
    return np.asarray(X_train), np.asarray(X_test), y_train, y_test


X_train, X_test, y_train, y_test = split(42)
print(f"{len(y_train)} flowers to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``class_log_prior_`` holds the logarithm of the prior of each class, the share of the training
# flowers that belong to it. The three species are about equally common, so the priors are about
# a third. ``score`` is the accuracy.
model = GaussianNB().fit(X_train, y_train)
print("priors:", np.round(np.exp(model.class_log_prior_), 3))
print(f"train accuracy: {model.score(X_train, y_train):.3f}")
print(f"test accuracy:  {model.score(X_test, y_test):.3f}")

# %%
# What the model learned
# ----------------------
#
# For every class and feature the model keeps a mean and a variance, nothing else. The model
# does not offer them as attributes, so they are computed here with NumPy from the training
# flowers, exactly as the fit does. The figure draws the resulting normal curve over the
# histogram of each feature. The petal features separate the species best: setosa stands alone
# and the other two overlap only a little. The sepal width is the weak one, the curves of all
# three species lie on top of each other.


def fit_gaussians(
    features: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the prior, the mean and the variance of every class (one row per class)."""
    classes = np.unique(labels)
    prior = np.array([np.mean(labels == c) for c in classes])
    mean = np.array([features[labels == c].mean(axis=0) for c in classes])
    variance = np.array([features[labels == c].var(axis=0) for c in classes])
    return prior, mean, variance


prior, mean, variance = fit_gaussians(X_train, y_train)

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
for feature, ax in enumerate(axes.ravel()):
    low, high = X_train[:, feature].min(), X_train[:, feature].max()
    grid = np.linspace(low - 0.5, high + 0.5, 200)
    for label in range(3):
        ax.hist(
            X_train[y_train == label, feature],
            bins=12,
            range=(low, high),
            density=True,
            alpha=0.35,
            color=f"C{label}",
        )
        sigma = np.sqrt(variance[label, feature])
        curve = np.exp(-((grid - mean[label, feature]) ** 2) / (2 * sigma**2))
        ax.plot(grid, curve / (sigma * np.sqrt(2 * np.pi)), color=f"C{label}", label=species[label])
    ax.set_xlabel(names[feature])
    ax.set_yticks([])
axes[0, 0].legend(loc="upper right")
fig.suptitle("One normal curve per class and feature")
fig.tight_layout()
plt.show()

# %%
# Check it by hand
# ----------------
#
# The model is small enough to write out. The score of a class is the logarithm of its prior plus
# the sum of the logarithms of the normal densities of the features, and the prediction is the
# class with the largest score. The scores can also be turned into probabilities, which the
# model itself does not return. ``probabilities`` below does both, and its predictions match the
# model on every test flower.


def probabilities(
    features: NDArray[np.float64],
    prior: NDArray[np.float64],
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the probability of every class for every sample, from Bayes' rule."""
    score = np.empty((len(features), len(prior)))
    for label in range(len(prior)):
        # A tiny constant keeps a variance of zero from dividing, as the model does.
        var = variance[label] + 1e-10
        log_density = -0.5 * np.log(2 * np.pi * var) - (features - mean[label]) ** 2 / (2 * var)
        score[:, label] = np.log(prior[label]) + log_density.sum(axis=1)
    score -= score.max(axis=1, keepdims=True)
    weight = np.exp(score)
    return weight / weight.sum(axis=1, keepdims=True)


by_hand = probabilities(X_test, prior, mean, variance)
print(
    "same predictions as the model:", bool(np.all(by_hand.argmax(axis=1) == model.predict(X_test)))
)
print("first test flower:", np.round(by_hand[0], 3), "->", species[by_hand[0].argmax()])

# %%
# The boundary
# ------------
#
# With two features the decision can be drawn. The figure uses a new model trained on only the
# petal length and the petal width, the two features with the cleanest curves. Each region is the
# area where one class has the largest score. The three species have different variances, so the
# boundaries are curved, not straight lines as in the linear models.
pair = [names.index("petal length (cm)"), names.index("petal width (cm)")]
pair_model = GaussianNB().fit(X_train[:, pair], y_train)

xx, yy = np.meshgrid(
    np.linspace(X_train[:, pair[0]].min() - 0.5, X_train[:, pair[0]].max() + 0.5, 300),
    np.linspace(X_train[:, pair[1]].min() - 0.3, X_train[:, pair[1]].max() + 0.3, 300),
)
regions = pair_model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5, 2.5], colors=["C0", "C1", "C2"], alpha=0.2)
for label in range(3):
    points = X_train[y_train == label][:, pair]
    ax.scatter(points[:, 0], points[:, 1], s=18, color=f"C{label}", label=species[label])
ax.set_xlabel(names[pair[0]])
ax.set_ylabel(names[pair[1]])
ax.set_title("Each class owns a region")
ax.legend(loc="upper left")
plt.show()

# %%
# More features are not always better
# -----------------------------------
#
# Ten different splits give a steadier number than one. The petal features alone do about as
# well as all four features, and the sepal features alone are clearly worse. Naive Bayes treats
# the features as separate pieces of evidence and adds them up, so a weak feature that adds
# little is not harmless: it still gets its full say.
subsets = {
    "petal length and width": [2, 3],
    "sepal length and width": [0, 1],
    "all four features": [0, 1, 2, 3],
}
for subset, columns in subsets.items():
    scores = []
    for seed in range(10):
        a_train, a_test, labels_train, labels_test = split(seed)
        fitted = GaussianNB().fit(a_train[:, columns], labels_train)
        scores.append(fitted.score(a_test[:, columns], labels_test))
    print(f"{subset:<24} mean test accuracy {np.mean(scores):.3f}")

# %%
# A pitfall: the independence assumption
# --------------------------------------
#
# The features of a flower are not independent: inside one species a longer petal is usually a
# wider petal too (the correlation is between 0.4 and 0.8). The model ignores that and counts
# the two features as two separate witnesses. The effect is easy to see by giving it the same
# witness several times. The table uses the petal width once, twice, five and ten times (always
# next to the petal length), which adds no information. The accuracy does not improve, but the
# model becomes more and more sure of itself: more flowers get a probability above 0.99. The
# probabilities of Naive Bayes are therefore too extreme, even when its predictions are good,
# and they should be used to rank the classes and not as a measure of confidence.
print("petal width used   test accuracy   flowers with a probability above 0.99")
for times in (1, 2, 5, 10):
    train = np.column_stack([X_train[:, [2]]] + [X_train[:, [3]]] * times)
    test = np.column_stack([X_test[:, [2]]] + [X_test[:, [3]]] * times)
    fitted = GaussianNB().fit(train, y_train)
    chance = probabilities(test, *fit_gaussians(train, y_train))
    sure = int(np.sum(chance.max(axis=1) > 0.99))
    accuracy = fitted.score(test, y_test)
    print(f"{times:>10} times   {accuracy:>13.3f}   {sure:>10} of {len(y_test)}")
