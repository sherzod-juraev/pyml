"""
Lasso
=====

:class:`~pyml.linear_model.Lasso` is :class:`~pyml.linear_model.LinearRegression` with a
different penalty: ``alpha`` times the sum of the absolute values of the weights. The penalty
pushes every weight toward zero with the same force, whatever its size, so the weights of
features that add little to the prediction end up at zero, or very close to it. In this way the
model chooses its features.

Use it when there are many features and you expect only some of them to matter, or when you want
a model that is short enough to read. When the features are correlated, Lasso keeps some of them
and drops the others, and which one it keeps can be a matter of chance. For a model that keeps all
the features, see :class:`~pyml.linear_model.Ridge`. For a mix of the two, see
:class:`~pyml.linear_model.ElasticNet`.
"""

# %%
# Parameters
# ----------
#
# * ``alpha`` is the strength of the penalty. ``alpha=0`` is ordinary least squares, and a
#   larger value drops more features.
# * ``learning_rate``, ``max_iter`` and ``tol`` control the gradient descent, as in
#   :class:`~pyml.linear_model.LinearRegression`. A weight that should be zero moves back and
#   forth around it with a step that grows with ``learning_rate``, so a smaller value settles
#   closer to zero, but needs more iterations.
# * ``fit_intercept`` says whether to learn the constant. The constant is never penalized.
#
# The full list, with the default values, is in the :class:`API reference
# <pyml.linear_model.Lasso>`.

# %%
# Dataset
# -------
#
# The same buildings as in the :class:`~pyml.linear_model.Ridge` example: predict the heating
# load from the shape of a building. Several of the eight features describe the same thing, so
# some of them are not needed.
#
# .. dataset-info:: energy_efficiency

# %%
# Split and standardize
# ---------------------
#
# The data is split first, and the scaler learns from the training part only. The penalty
# treats all weights the same, so the features must be on the same scale for it to be fair.
import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_energy_efficiency

from pyml.linear_model import Lasso, LinearRegression
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

# %%
# Fit the model
# -------------
#
# Lasso with ``alpha=0.3``, next to least squares. Several weights of Lasso are almost zero. The
# weights come from gradient descent and never land on zero exactly, they move around it, so a
# weight below 0.1 is counted as zero here. That is a change of less than 0.1 in the predicted
# load per standard deviation of the feature, small against weights of 2 and 8 of the others.
negligible = 0.1
ols = LinearRegression(learning_rate=0.1, max_iter=5000, tol=1e-7).fit(A_train, y_train)
lasso = Lasso(alpha=0.3, learning_rate=0.01, max_iter=10000, tol=1e-7).fit(A_train, y_train)

dropped = np.abs(lasso.coef_) < negligible
print("weights exactly zero:", int((lasso.coef_ == 0).sum()))
print("kept:   ", ", ".join(np.array(names)[~dropped]))
print("dropped:", ", ".join(np.array(names)[dropped]))
print(f"test R squared, least squares: {ols.score(A_test, y_test):.3f}")
print(f"test R squared, Lasso:         {lasso.score(A_test, y_test):.3f}")

positions = np.arange(len(names))
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.38, bottom=0.32)
ax.barh(positions + 0.2, ols.coef_, height=0.4, color="C0", label="Least squares")
ax.barh(positions - 0.2, lasso.coef_, height=0.4, color="C1", label="Lasso")
ax.set_yticks(positions, names)
ax.invert_yaxis()
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("weight")
ax.set_title("Lasso drops the weights it does not need")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.show()

# %%
# Which features stay
# -------------------
#
# Of the correlated features, Lasso keeps ``wall_area`` and drops ``relative_compactness``,
# ``surface_area`` and ``roof_area``, which carry much of the same information. It also drops
# ``orientation``, whose weight was small already. The test score loses very little. Lasso could
# just as well have kept another one of the correlated group, and a different split of the data
# may give another choice, so the dropped features are not necessarily the unimportant ones.
#
# The path shows what ``alpha`` does. A larger value drops more features, and at some point it
# starts to drop features that matter and the score falls. The step is kept small, so that the
# dropped weights settle close to zero.
alphas = [0.001, 0.01, 0.1, 0.3, 1, 2]
path = []
for alpha in alphas:
    fitted = Lasso(alpha=alpha, learning_rate=0.01, max_iter=10000, tol=1e-7).fit(A_train, y_train)
    path.append(fitted.coef_)
    kept = int((np.abs(fitted.coef_) >= negligible).sum())
    print(
        f"alpha={alpha}: {kept} of {len(names)} features kept, "
        f"test R squared {fitted.score(A_test, y_test):.3f}"
    )

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.6)
for column, name in enumerate(names):
    ax.plot(alphas, np.array(path)[:, column], marker="o", markersize=3, label=name)
ax.set_xscale("log")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("alpha")
ax.set_ylabel("weight")
ax.set_title("The weights leave one by one")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6)
plt.show()

# %%
# A pitfall: the size of the step
# -------------------------------
#
# A weight that should be zero is pushed by the penalty at every step, and by the same
# amount, so it overshoots zero and comes back. The larger the ``learning_rate``, the wider it
# swings. A smaller step settles closer to zero, but it needs more iterations to get there. The
# table shows the largest weight among the four features that Lasso dropped above. With
# ``learning_rate=0.1`` the weights still swing by about 0.05. With ``0.01`` they settle at
# 0.003. With ``0.001`` the loop runs out of ``max_iter`` before the weights have reached zero,
# and a dropped feature still has a large weight.
for rate in (0.1, 0.01, 0.001):
    noisy = Lasso(alpha=0.3, learning_rate=rate, max_iter=10000, tol=1e-9).fit(A_train, y_train)
    spread = np.abs(noisy.coef_[dropped]).max()
    print(f"learning_rate={rate}: largest dropped weight {spread:.3f}, {noisy.n_iter_} iterations")
