"""
Multinomial Naive Bayes
=======================

:class:`~pyml.naive_bayes.MultinomialNB` is the Naive Bayes classifier for counts. The classic
case is a text: the features are how many times each word appears in a document, and a class
(spam or not, one topic or another) is described by how often it uses each word. For a new
sample the model adds up, for every class, the logarithm of the prior and the logarithm of the
frequency of each feature, weighted by how many times the feature occurs, and picks the class
with the largest sum. The features are assumed to be independent, which is the "naive" part.

Use it for count data with many features, such as word counts, where it is fast and a strong
baseline. This example uses a different kind of count, the brightness of the pixels of a
handwritten digit, which can be read as the amount of ink in a pixel. The last sections show the
two ways to break the model: a smoothing of zero and negative features.
"""

# %%
# Parameters
# ----------
#
# * ``alpha`` is the smoothing. It is added to every count, so a feature that never occurred in
#   a class still gets a small frequency instead of zero. ``alpha=1`` is the usual choice.
#
# The fit is a closed formula, so there is no ``learning_rate`` or ``max_iter``. The features
# must be counts, or at least not negative. Unlike in most other examples, they must **not** be
# standardized, the last section shows why.
#
# The :class:`API reference <pyml.naive_bayes.MultinomialNB>` has the details.

# %%
# Dataset
# -------
#
# The task is to recognize a handwritten digit from an image of 8 by 8 pixels. Each pixel is a
# number from 0 (blank) to 16 (full ink), and the 64 pixels are the features.
#
# .. dataset-info:: digits

# %%
# Split the data
# --------------
#
# The model is judged on digits it has not seen, so 20 percent of the data is kept for the test.
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_digits

from pyml.model_selection import train_test_split
from pyml.naive_bayes import MultinomialNB
from pyml.preprocessing import StandardScaler


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


data = load_digits()
X_train, X_test, y_train, y_test = split(42)
print(f"{len(y_train)} digits to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``class_log_prior_`` holds the logarithm of the prior of each class, the share of the training
# digits that belong to it. The ten digits are about equally common, so each prior is about 0.1.
# ``score`` is the accuracy. Chance is 10 percent, and the model gets nine digits out of ten.
model = MultinomialNB(alpha=1).fit(X_train, y_train)
print("priors:", np.round(np.exp(model.class_log_prior_), 3))
print(f"train accuracy: {model.score(X_train, y_train):.3f}")
print(f"test accuracy:  {model.score(X_test, y_test):.3f}")

# %%
# What the model learned
# ----------------------
#
# For each digit the model keeps one number per pixel: the share of all the ink of that digit
# that lies in the pixel, with ``alpha`` added to every count. The model does not offer these as
# an attribute, so they are computed here from the training digits as the fit does. Drawn as
# images, they are the places where a digit puts its ink. The zero is a ring and the one a thin
# vertical stroke. Three pixels are blank in every training digit, and they get the same tiny
# share in every class.
alpha = 1.0
ink = np.array([X_train[y_train == label].sum(axis=0) for label in range(10)])
share = (ink + alpha) / (ink.sum(axis=1, keepdims=True) + alpha * ink.shape[1])

fig, axes = plt.subplots(2, 5, figsize=(9, 4.6), layout="constrained")
for label, ax in enumerate(axes.ravel()):
    ax.imshow(share[label].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"digit {label}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Where each digit puts its ink")
plt.show()

# %%
# Check it by hand
# ----------------
#
# The score of a digit is the logarithm of its prior plus the sum over the pixels of the
# brightness times the logarithm of the share, which is one matrix product. The prediction is the
# digit with the largest score. It matches the model on every test digit.
score = model.class_log_prior_ + X_test @ np.log(share).T
print("same predictions as the model:", bool(np.all(score.argmax(axis=1) == model.predict(X_test))))

# %%
# Which mistakes
# --------------
#
# Of the 359 test digits the model gets 34 wrong. The figure shows ten of them, with the
# true digit and the model's answer. Some digits look alike when drawn in 64 pixels, such as a
# 5 and a 9, and some are written in an unusual way, such as a 1 with a foot. The model sees only
# how much ink is in each pixel, not the shape of a stroke, and the independence assumption
# cannot learn that two pixels next to each other make a line.
predicted = model.predict(X_test)
wrong = np.flatnonzero(predicted != y_test)
print(f"{len(wrong)} of {len(y_test)} test digits are wrong")

fig, axes = plt.subplots(2, 5, figsize=(9, 4.6), layout="constrained")
for ax, index in zip(axes.ravel(), wrong[:10], strict=False):
    ax.imshow(X_test[index].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"is {y_test[index]}, said {predicted[index]}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Digits the model gets wrong")
plt.show()

# %%
# Choose ``alpha``
# ----------------
#
# The smoothing hardly matters here. A pixel that is blank in every training digit of a class is
# rare, and each of the other pixels is seen many times, so adding 1 changes the shares by very
# little. The accuracy stays the same over a wide range and falls only when ``alpha`` is so large
# that it drowns the data. Smoothing matters more for text, where most of the many words are
# rare. The accuracy is the mean of ten splits.
print("alpha    mean test accuracy")
for value in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
    scores = []
    for seed in range(10):
        a_train, a_test, labels_train, labels_test = split(seed)
        fitted = MultinomialNB(alpha=value).fit(a_train, labels_train)
        scores.append(fitted.score(a_test, labels_test))
    print(f"{value:<8} {np.mean(scores):.3f}")

# %%
# A pitfall: smoothing of zero
# ----------------------------
#
# Without smoothing, a pixel that is blank in every training digit of a class gets the share 0,
# and its logarithm is minus infinity. Any test digit that has ink there is then impossible for
# that class, and where the pixel is blank the product of zero and infinity is not a number. No
# error is raised. The scores become useless and the model says the same class for every digit,
# so the accuracy falls to chance.
with np.errstate(all="ignore"):
    unsmoothed = MultinomialNB(alpha=0).fit(X_train, y_train)
    print(f"alpha=0: test accuracy {unsmoothed.score(X_test, y_test):.3f}")

# %%
# A pitfall: negative features
# ----------------------------
#
# Almost every other model in these examples wants standardized features, and standardizing is a
# habit. Here it breaks the model. A standardized feature is negative below its mean, so it is no
# longer a count, and the share of a pixel can become negative, with an undefined logarithm.
# Again no error is raised, and the accuracy is chance. Scaling every feature into 0 to 1 keeps
# them non-negative, and it works.
scaler = StandardScaler().fit(X_train)
with np.errstate(all="ignore"):
    standardized = MultinomialNB().fit(scaler.transform(X_train), y_train)
    accuracy = standardized.score(scaler.transform(X_test), y_test)
print(f"standardized features: test accuracy {accuracy:.3f}")

low = X_train.min(axis=0)
span = np.maximum(X_train.max(axis=0) - low, 1e-12)
scaled = MultinomialNB().fit((X_train - low) / span, y_train)
print(f"features in 0 to 1:    test accuracy {scaled.score((X_test - low) / span, y_test):.3f}")
