"""
Logistic Regression
===================

:class:`~pyml.linear_model.LogisticRegression` decides between two classes. It computes a
weighted sum of the features, as linear regression does, and squeezes the result through the
sigmoid function into a number between 0 and 1. That number is read as the probability of the
positive class, and the class is chosen by comparing it with 0.5. The weights are found by
gradient descent on the cross-entropy loss.

Use it as the first model for a yes-or-no question, and as a baseline for the classifiers that
come after it. It gives a probability and not only a label, and its weights can be read. Its
boundary between the classes is a straight line (or a flat plane), and the last sections show
what that means. This model has no penalty on the weights. When the features are strongly
correlated, see :class:`~pyml.linear_model.RidgeClassifier`,
:class:`~pyml.linear_model.LassoClassifier` and
:class:`~pyml.linear_model.ElasticNetClassifier`.
"""

# %%
# Parameters
# ----------
#
# * ``learning_rate`` is the size of one gradient descent step. A small one is safe and slow,
#   a large one is fast until it overshoots.
# * ``max_iter`` and ``tol`` decide when to stop. The loop runs at most ``max_iter`` steps and
#   stops earlier when the loss changes by less than ``tol``.
# * ``fit_intercept`` says whether to learn the constant. Leave it on unless the data is
#   already centered.
#
# The labels must be 0 and 1. The full list of parameters, with the default values, is in the
# :class:`API reference <pyml.linear_model.LogisticRegression>`.

# %%
# Dataset
# -------
#
# The task is to tell a malignant tumor from a benign one using measurements of cell nuclei
# from a breast biopsy. The label ``0`` is malignant and ``1`` is benign, so the model
# estimates the probability that a tumor is benign.
#
# .. dataset-info:: breast_cancer

# %%
# Split and standardize
# ---------------------
#
# The model is judged on tumors it has not seen, so the data is split first, and 20 percent is
# kept for the test. Gradient descent needs features on a similar scale, so each feature is
# standardized. The scaler learns its mean and spread from the training part only, otherwise
# information from the test part leaks into the model. The last section shows what happens
# without the scaling.
import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_breast_cancer

from pyml.linear_model import LogisticRegression
from pyml.model_selection import train_test_split
from pyml.preprocessing import StandardScaler

data = load_breast_cancer()
names = data.feature_names
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    data.X, data.y, test_size=0.2, random_state=42
)
# train_test_split is typed for both kinds of target; say that this one is a class label.
y_train = np.asarray(y_train_raw, dtype=np.int64)
y_test = np.asarray(y_test_raw, dtype=np.int64)

scaler = StandardScaler().fit(X_train)
A_train, A_test = scaler.transform(X_train), scaler.transform(X_test)
print(f"{len(y_train)} tumors to train on, {len(y_test)} to test on")
print(f"malignant in the test part: {np.sum(y_test == 0)}, benign: {np.sum(y_test == 1)}")

# %%
# Fit the model
# -------------
#
# ``n_iter_`` is how many steps the loop needed. ``score`` is the accuracy, the share of
# tumors that got the right label. The training and test accuracy are close, so the model has
# not just memorized the training tumors.
model = LogisticRegression(learning_rate=0.1, max_iter=1000).fit(A_train, y_train)
print("iterations:", model.n_iter_)
print(f"train accuracy: {model.score(A_train, y_train):.3f}")
print(f"test accuracy:  {model.score(A_test, y_test):.3f}")

# %%
# Probabilities
# -------------
#
# ``predict_proba`` returns the estimated probability that each tumor is benign. The
# histogram shows these numbers for the test tumors, separately for the two true classes. Most
# tumors are given a probability close to 0 or 1, which means the model is confident, and the
# few in the middle are the ones it is unsure about. ``predict`` is the same probability cut at
# 0.5.
proba = model.predict_proba(A_test)

fig, ax = plt.subplots()
bins = [i / 20 for i in range(21)]
ax.hist(proba[y_test == 0], bins=bins, alpha=0.7, color="C3", label="Malignant")
ax.hist(proba[y_test == 1], bins=bins, alpha=0.7, color="C0", label="Benign")
ax.axvline(0.5, color="gray", linestyle="--")
ax.set_xlabel("predicted probability of benign")
ax.set_ylabel("number of test tumors")
ax.set_title("Confident on most tumors")
ax.legend(loc="upper left")
plt.show()

# %%
# Which mistakes
# --------------
#
# Accuracy hides the kind of mistake. The confusion matrix counts every combination of the
# true class and the predicted class, with the true class in the rows. Here the two mistakes are
# not equal: calling a malignant tumor benign (top right) can delay a treatment, and calling a
# benign tumor malignant (bottom left) leads to further tests.
predicted = model.predict(A_test)
matrix = np.array([[np.sum((y_test == t) & (predicted == p)) for p in (0, 1)] for t in (0, 1)])
print(matrix)

fig, ax = plt.subplots()
ax.imshow(matrix, cmap="Blues")
for row in range(2):
    for column in range(2):
        shade = "white" if matrix[row, column] > matrix.max() / 2 else "black"
        ax.text(column, row, matrix[row, column], ha="center", va="center", color=shade)
ax.set_xticks([0, 1], labels=["malignant", "benign"])
ax.set_yticks([0, 1], labels=["malignant", "benign"])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Confusion matrix of the test tumors")
plt.show()

# %%
# Moving the threshold
# --------------------
#
# The 0.5 is a convention and not a law. Because the model gives a probability, the cut can be
# moved. A higher cut calls fewer tumors benign: fewer malignant ones are missed, and more
# benign ones are sent for further tests. The test part has only 38 malignant tumors, so the
# counts are small and should be read as a direction and not as exact rates.
for cut in (0.5, 0.7, 0.9, 0.95):
    called_benign = proba >= cut
    missed = np.sum((y_test == 0) & called_benign)
    false_alarms = np.sum((y_test == 1) & ~called_benign)
    print(f"cut {cut}: {missed} malignant missed, {false_alarms} benign sent for tests")

# %%
# Read the weights
# ----------------
#
# A weight is the change of the log-odds of "benign" when the feature grows by one standard
# deviation, so a negative weight pushes toward malignant. The largest weights belong to the
# size of the nuclei: the larger the cells, the more likely the tumor is malignant. Take the
# exact values with care. Radius, perimeter and area measure almost the same thing, so the
# model can split the credit between them in many ways and a single weight says little by
# itself.
top = np.argsort(np.abs(model.coef_))[-8:]
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.38)
positions = np.arange(len(top))
ax.barh(positions, model.coef_[top], color="C0")
ax.set_yticks(positions, [names[i] for i in top])
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("weight (log-odds per standard deviation)")
ax.set_title("Larger nuclei point to malignant")
plt.show()

# %%
# The boundary
# ------------
#
# With two features the boundary can be drawn. This is a new model trained on only
# ``worst concave points`` and ``worst texture``, since the model above works in 30 dimensions
# and has no boundary to draw on a page. The shading is the predicted probability, and the
# line is where it equals 0.5. The boundary is a straight line, whatever the data looks like.
pair = [names.index("worst concave points"), names.index("worst texture")]
pair_scaler = StandardScaler().fit(X_train[:, pair])
P_train = pair_scaler.transform(X_train[:, pair])
P_test = pair_scaler.transform(X_test[:, pair])
pair_model = LogisticRegression(learning_rate=0.1, max_iter=1000).fit(P_train, y_train)
print(f"test accuracy with 2 features:  {pair_model.score(P_test, y_test):.3f}")
print(f"test accuracy with 30 features: {model.score(A_test, y_test):.3f}")

lower = P_train.min(axis=0) - 0.5
upper = P_train.max(axis=0) + 0.5
xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 200), np.linspace(lower[1], upper[1], 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
surface = pair_model.predict_proba(grid).reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.35)
ax.contour(xx, yy, surface, levels=[0.5], colors="black", linewidths=1.5)
for label, color, name in ((0, "C3", "Malignant"), (1, "C0", "Benign")):
    points = P_train[y_train == label]
    ax.scatter(points[:, 0], points[:, 1], s=15, color=color, label=name)
ax.set_xlabel("worst concave points (standardized)")
ax.set_ylabel("worst texture (standardized)")
ax.set_title("Probability of benign, with the 0.5 boundary")
ax.legend(loc="lower right")
plt.show()

# %%
# A pitfall: skipping the scaling
# -------------------------------
#
# Without the standardizing, ``worst area`` is in the hundreds and ``worst smoothness`` is
# below one. The same step size is far too large for the big features, and the weights jump
# back and forth with each step. The fit does not fail with an error. It reports a handful of
# iterations and an accuracy that is hardly better than guessing, because the loss alternated
# between two values and two neighboring steps happened to give the same number, which looks
# like convergence to the stopping rule.
with np.errstate(all="ignore"):
    raw_model = LogisticRegression(learning_rate=0.1, max_iter=1000).fit(X_train, y_train)
print("iterations:", raw_model.n_iter_)
print(f"test accuracy without scaling: {raw_model.score(X_test, y_test):.3f}")
print(f"share of the larger class:     {max(np.mean(y_test), 1 - np.mean(y_test)):.3f}")
