"""
Ridge
=====

:class:`~pyml.linear_model.Ridge` is :class:`~pyml.linear_model.LinearRegression` with a
penalty. Besides the squared error, it adds ``alpha`` times the sum of the squared weights to
the loss, so large weights are expensive and the fit prefers small ones.

Use it when the features are strongly correlated. Then many different sets of weights predict
almost equally well, plain least squares picks one of them more or less by chance, and the
weights change more than they should with a few different rows in the data. The penalty settles
that choice.
It shrinks every weight toward zero but never to exactly zero; for a model that drops features,
see :class:`~pyml.linear_model.Lasso`.
"""

# %%
# Parameters
# ----------
#
# * ``alpha`` is the strength of the penalty. ``alpha=0`` is ordinary least squares, and a larger
#   value shrinks the weights more.
# * ``learning_rate``, ``max_iter`` and ``tol`` control the gradient descent, as in
#   :class:`~pyml.linear_model.LinearRegression`. The penalty makes the loss steeper, so a
#   larger ``alpha`` needs a smaller ``learning_rate``.
# * ``fit_intercept`` says whether to learn the constant. The constant is never penalized.
#
# The full list, with the default values, is in the :class:`API reference
# <pyml.linear_model.Ridge>`.

# %%
# Dataset
# -------
#
# The task is to predict the heating load of a building from its shape. The dataset has two
# targets, the heating load and the cooling load, and this example uses the first one. Several
# features describe the same thing in different ways, for example the surface area and the
# relative compactness of the building.
#
# .. dataset-info:: energy_efficiency

# %%
# Split and standardize
# ---------------------
#
# The data is split first, and the scaler learns from the training part only. Standardizing
# matters twice here: gradient descent needs it, and the penalty treats all weights the same, so
# the features must be on the same scale for the penalty to be fair.
import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_energy_efficiency

from pyml.linear_model import LinearRegression, Ridge
from pyml.model_selection import train_test_split
from pyml.preprocessing import StandardScaler

data = load_energy_efficiency()
names = data.feature_names
heating = np.asarray(data.y[:, 0], dtype=np.float64)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    data.X, heating, test_size=0.2, random_state=42
)
# train_test_split is typed for both kinds of target; say that this one is a float target.
y_train = np.asarray(y_train_raw, dtype=np.float64)
y_test = np.asarray(y_test_raw, dtype=np.float64)

scaler = StandardScaler().fit(X_train)
A_train, A_test = scaler.transform(X_train), scaler.transform(X_test)

correlation = np.asarray(np.corrcoef(A_train, rowvar=False), dtype=np.float64)
pair = (names.index("relative_compactness"), names.index("surface_area"))
print(f"correlation of relative_compactness and surface_area: {correlation[pair[0], pair[1]]:.2f}")

# %%
# Fit the model
# -------------
#
# Ridge with a small penalty, next to the plain least squares fit on the same data. The
# weights of Ridge are smaller, and the two features that move together share the work in a
# different way. The price is a little accuracy, because the penalty pulls the fit away from
# the best fit of the training data.
ols = LinearRegression(learning_rate=0.1, max_iter=5000, tol=1e-7).fit(A_train, y_train)
ridge = Ridge(alpha=0.1, learning_rate=0.05, max_iter=5000, tol=1e-7).fit(A_train, y_train)
print(f"test R squared, least squares: {ols.score(A_test, y_test):.3f}")
print(f"test R squared, Ridge:         {ridge.score(A_test, y_test):.3f}")

positions = np.arange(len(names))
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.38, bottom=0.32)
ax.barh(positions + 0.2, ols.coef_, height=0.4, color="C0", label="Least squares")
ax.barh(positions - 0.2, ridge.coef_, height=0.4, color="C1", label="Ridge")
ax.set_yticks(positions, names)
ax.invert_yaxis()
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("weight")
ax.set_title("Ridge shrinks the weights")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.show()

# %%
# The weights are steadier
# ------------------------
#
# What the penalty buys is stability. The same two models are fitted on 30 random resamples of
# the training data, and the two correlated weights are compared. The weights of least squares
# vary several times more from one resample to the next than the weights of Ridge, as the
# spreads below and the plot show.
rng = np.random.default_rng(42)
ols_weights, ridge_weights = [], []
for _ in range(30):
    rows = rng.integers(0, len(A_train), len(A_train))
    ols_fit = LinearRegression(learning_rate=0.1, max_iter=5000, tol=1e-7)
    ridge_fit = Ridge(alpha=0.1, learning_rate=0.05, max_iter=5000, tol=1e-7)
    ols_weights.append(ols_fit.fit(A_train[rows], y_train[rows]).coef_[list(pair)])
    ridge_weights.append(ridge_fit.fit(A_train[rows], y_train[rows]).coef_[list(pair)])
spread_ols = np.std(ols_weights, axis=0)
spread_ridge = np.std(ridge_weights, axis=0)
for column, name in enumerate(("relative_compactness", "surface_area")):
    print(f"{name}: spread {spread_ols[column]:.2f} (least squares), ", end="")
    print(f"{spread_ridge[column]:.2f} (Ridge)")

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
for column, ax in enumerate(axes):
    for position, weights, color in ((0, ols_weights, "C0"), (1, ridge_weights, "C1")):
        values = np.array(weights)[:, column]
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(position + jitter, values, s=18, alpha=0.7, color=color)
    ax.set_xticks([0, 1], ["Least squares", "Ridge"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(names[pair[column]])
axes[0].set_ylabel("weight in 30 resamples")
plt.show()

# %%
# Choose ``alpha``
# ----------------
#
# The path of the weights shows what ``alpha`` does. A larger value pulls all weights toward
# zero. A weight can even change its sign on the way, as ``relative_compactness`` does, because
# at a small ``alpha`` it cancels a part of ``surface_area``. The printed scores show the price:
# the larger the ``alpha``, the lower the test score.
#
# The penalty makes the loss steeper, so the step has to shrink as ``alpha`` grows. Here
# ``learning_rate = 0.1 / (1 + alpha)`` is stable for every value in the table.
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
path = []
for alpha in alphas:
    fitted = Ridge(alpha=alpha, learning_rate=0.1 / (1 + alpha), max_iter=5000).fit(
        A_train, y_train
    )
    path.append(fitted.coef_)
    print(f"alpha={alpha}: test R squared {fitted.score(A_test, y_test):.2f}")

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.6)
for column, name in enumerate(names):
    ax.plot(alphas, np.array(path)[:, column], marker="o", markersize=3, label=name)
ax.set_xscale("log")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("alpha")
ax.set_ylabel("weight")
ax.set_title("Every weight shrinks toward zero")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6)
plt.show()

# %%
# A pitfall: a step that is too large for ``alpha``
# -------------------------------------------------
#
# The penalty adds to the steepness of the loss. The step ``learning_rate=0.1`` that is fine for
# ``alpha=0.1`` is too large for ``alpha=10``, the weights explode, and the fit stops with an
# error.
try:
    with np.errstate(all="ignore"):
        Ridge(alpha=10, learning_rate=0.1, max_iter=5000).fit(A_train, y_train)
except ValueError as error:
    print("fit failed:", error)
