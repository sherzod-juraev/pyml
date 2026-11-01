"""
Linear Regression
=================

:class:`~pyml.linear_model.LinearRegression` predicts a number as a weighted sum of the
features plus a constant. It looks for the weights that make the squared error of the
predictions as small as possible, and it does so by gradient descent: start from zero weights
and move them a little in the direction that lowers the error, again and again.

Use it as the first model for a number you want to predict, and as a baseline that the other
models must beat. Its weights can be read directly, which makes it easy to explain. It can only
draw straight relations, and the last sections show what that costs. When the features are
strongly correlated, see :class:`~pyml.linear_model.Ridge` and
:class:`~pyml.linear_model.Lasso`.
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
# The full list, with the default values, is in the :class:`API reference
# <pyml.linear_model.LinearRegression>`.

# %%
# Dataset
# -------
#
# The task is to predict the compressive strength of a concrete mixture from its ingredients
# and its age.
#
# .. dataset-info:: concrete_strength

# %%
# Split and standardize
# ---------------------
#
# The model is judged on mixtures it has not seen, so the data is split first, and 20 percent is
# kept for the test. Gradient descent needs features on a similar scale, so each feature is
# standardized. The scaler learns its mean and spread from the training part only, otherwise
# information from the test part leaks into the model. The last section shows what happens
# without the scaling.
import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_concrete_strength

from pyml.linear_model import LinearRegression
from pyml.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from pyml.model_selection import train_test_split
from pyml.preprocessing import StandardScaler

data = load_concrete_strength()
names = data.feature_names
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    data.X, data.y, test_size=0.2, random_state=42
)
# train_test_split is typed for both kinds of target; say that this one is a float target.
y_train = np.asarray(y_train_raw, dtype=np.float64)
y_test = np.asarray(y_test_raw, dtype=np.float64)

scaler = StandardScaler().fit(X_train)
A_train, A_test = scaler.transform(X_train), scaler.transform(X_test)
print(f"{len(y_train)} mixtures to train on, {len(y_test)} to test on")

# %%
# Fit the model
# -------------
#
# ``coef_`` holds one weight per feature. The features are standardized, so a weight is the
# change of the strength, in MPa, when that feature grows by one standard deviation, and the
# weights can be compared with each other. ``n_iter_`` is how many steps the loop needed.
model = LinearRegression(learning_rate=0.1, max_iter=1000).fit(A_train, y_train)
print("iterations:", model.n_iter_)
print(f"intercept: {model.intercept_:.1f} MPa")

order = np.argsort(model.coef_)
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.38)
positions = np.arange(len(order))
ax.barh(positions, model.coef_[order], color="C0")
ax.set_yticks(positions, [names[i] for i in order])
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("weight (MPa per standard deviation)")
ax.set_title("Most features raise the strength, water lowers it")
plt.show()

# %%
# Predicted against actual
# ------------------------
#
# On the test mixtures the prediction is compared with the measured strength. A perfect model
# puts every point on the dashed line. The strongest mixtures are predicted too low and the
# weakest too high, which already hints at what a straight line misses. ``score`` is the R
# squared: 1 is perfect, and 0 is no better than always predicting the mean.
y_pred = model.predict(A_test)
print(f"R squared:  {model.score(A_test, y_test):.2f}")
print(f"RMSE:       {root_mean_squared_error(y_test, y_pred):.1f} MPa")
print(f"MAE:        {mean_absolute_error(y_test, y_pred):.1f} MPa")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, s=20, alpha=0.7)
limits = [float(y_test.min()), float(y_test.max())]
ax.plot(limits, limits, color="C3", linestyle="--", label="Perfect prediction")
ax.set_xlabel("actual strength (MPa)")
ax.set_ylabel("predicted strength (MPa)")
ax.set_title("The extremes are pulled toward the middle")
ax.legend(loc="upper left")
plt.show()

# %%
# Gradient descent against the exact solution
# -------------------------------------------
#
# Least squares also has an exact solution, which NumPy computes directly with
# ``np.linalg.lstsq``. It is a good check of the model: both should find nearly the same
# weights. Gradient descent stops when the loss changes by less than ``tol``, so it stops a
# little short of the exact answer.
design = np.column_stack([A_train, np.ones(len(A_train))])
exact = np.linalg.lstsq(design, y_train, rcond=None)[0]
print(f"largest weight difference: {np.abs(model.coef_ - exact[:-1]).max():.2f} MPa")
print(f"test R squared, exact: {r2_score(y_test, A_test @ exact[:-1] + exact[-1]):.4f}")
print(f"test R squared, model: {model.score(A_test, y_test):.4f}")

# %%
# Choose ``learning_rate``
# ------------------------
#
# The step size decides how fast the loop gets there, or whether it gets there at all. A run
# that needed ``max_iter`` steps was stopped before it settled. A step that is too large makes
# the loss grow with every step, and the weights explode.
for rate in (0.001, 0.01, 0.1, 0.4, 0.5):
    with np.errstate(all="ignore"):
        fitted = LinearRegression(learning_rate=rate, max_iter=1000).fit(A_train, y_train)
        score = fitted.score(A_test, y_test)
    print(f"learning_rate={rate}: {fitted.n_iter_} iterations, test R squared {score:.3g}")

# %%
# Where a straight line is not enough
# -----------------------------------
#
# The residual is the actual strength minus the prediction. A model that fits well leaves
# residuals that scatter around zero with no pattern. Here they do not: young mixtures are
# predicted too high and middle-aged ones too low. Strength grows quickly in the first days
# and slowly later, and a straight line in ``age`` cannot follow that.
residual = y_test - y_pred
age = names.index("age")

fig, ax = plt.subplots()
ax.scatter(X_test[:, age], residual, s=20, alpha=0.7)
ax.axhline(0, color="C3", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("age (days, log scale)")
ax.set_ylabel("residual (MPa)")
ax.set_title("The residuals follow the age")
plt.show()

# %%
# The model is linear in its weights, not in the features, so the fix is to give it a feature
# that grows the way the strength does. The logarithm of the age does that. Everything else
# stays the same:
X_log = data.X.copy()
X_log[:, age] = np.log(X_log[:, age])
L_train, L_test, _, _ = train_test_split(X_log, data.y, test_size=0.2, random_state=42)
log_scaler = StandardScaler().fit(L_train)
log_model = LinearRegression(learning_rate=0.1, max_iter=1000).fit(
    log_scaler.transform(L_train), y_train
)
print(f"test R squared with age: {model.score(A_test, y_test):.2f}")
print(f"test R squared with log(age): {log_model.score(log_scaler.transform(L_test), y_test):.2f}")

# %%
# A pitfall: skipping the scaling
# -------------------------------
#
# Without the standardizing, ``coarse_aggregate`` is in the hundreds and ``superplasticizer`` is
# in the single digits. The same step size is far too large for the big features, the weights
# explode, and the fit stops with an error.
try:
    with np.errstate(all="ignore"):
        LinearRegression(learning_rate=0.1, max_iter=1000).fit(X_train, y_train)
except ValueError as error:
    print("fit failed:", error)
