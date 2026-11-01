"""
Elastic Net
===========

:class:`~pyml.linear_model.ElasticNet` mixes the two penalties of the earlier models. The
:class:`~pyml.linear_model.Lasso` part pushes the weights of unneeded features toward zero, and
the :class:`~pyml.linear_model.Ridge` part shrinks the weights smoothly. The parameter
``l1_ratio`` says how much of each: ``1`` is Lasso, ``0`` is Ridge, and the values between are
the mix.

Use it when the features come in correlated groups. Lasso tends to keep one feature of a group
and drop the others. The Ridge part makes Elastic Net keep the members of a group together and
share the work between them. Elastic Net also has one more knob to tune than Lasso or Ridge,
and that is the price.
"""

# %%
# Parameters
# ----------
#
# * ``alpha`` is the strength of the whole penalty. ``alpha=0`` is ordinary least squares.
# * ``l1_ratio`` is the share of the Lasso penalty, between ``0`` and ``1``.
# * ``learning_rate``, ``max_iter`` and ``tol`` control the gradient descent, as in the other
#   linear models. A weight that should be zero swings around it by about
#   ``learning_rate * alpha * l1_ratio``, so a smaller ``learning_rate`` settles closer to zero.
# * ``fit_intercept`` says whether to learn the constant. The constant is never penalized.
#
# The full list, with the default values, is in the :class:`API reference
# <pyml.linear_model.ElasticNet>`.

# %%
# Dataset
# -------
#
# The same buildings as in the Ridge and Lasso examples: predict the heating load from the shape
# of a building. Several of the features are correlated, which is where Elastic Net differs from
# Lasso.
#
# .. dataset-info:: energy_efficiency

# %%
# Split and standardize
# ---------------------
#
# The data is split first, and the scaler learns from the training part only. The penalty treats
# all weights the same, so the features must be on the same scale for it to be fair.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pyml_datasets import load_energy_efficiency

from pyml.linear_model import ElasticNet, Lasso, Ridge
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
# Elastic Net with half of the penalty from Lasso, next to Lasso with the same ``alpha``. As in
# the Lasso example, the weights come from gradient descent and swing around zero instead of
# landing on it, so a weight below 0.1 is counted as zero.
negligible = 0.1
settings: dict[str, Any] = {"learning_rate": 0.01, "max_iter": 10000, "tol": 1e-7}
lasso = Lasso(alpha=0.3, **settings).fit(A_train, y_train)
elastic = ElasticNet(alpha=0.3, l1_ratio=0.5, **settings).fit(A_train, y_train)

for label, fitted in (("Lasso", lasso), ("Elastic Net", elastic)):
    kept = np.abs(fitted.coef_) >= negligible
    print(
        f"{label}: {int(kept.sum())} features kept, test R squared "
        f"{fitted.score(A_test, y_test):.3f}"
    )
    print("  dropped:", ", ".join(np.array(names)[~kept]))

positions = np.arange(len(names))
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.38, bottom=0.32)
ax.barh(positions + 0.2, lasso.coef_, height=0.4, color="C0", label="Lasso")
ax.barh(positions - 0.2, elastic.coef_, height=0.4, color="C1", label="Elastic Net")
ax.set_yticks(positions, names)
ax.invert_yaxis()
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("weight")
ax.set_title("Elastic Net keeps more of the correlated group")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.show()

# %%
# Lasso keeps ``wall_area`` and drops ``surface_area`` and ``roof_area``, which move together
# with it. Elastic Net keeps them too, and the weight of ``overall_height`` is smaller, because
# the group now carries a part of the prediction. In exchange it gives up a little test score.
# Which one is better depends on what the model is for: a short model, or one that does not hide
# a feature just because a neighbor was chosen first.

# %%
# The two ends of ``l1_ratio``
# ----------------------------
#
# At the ends, Elastic Net is one of the other two models. With ``l1_ratio=1`` it gives the
# same weights as Lasso. With ``l1_ratio=0`` it gives the weights of Ridge, but with ``alpha``
# halved, because the L2 part of the Elastic Net penalty carries a factor of one half and the
# penalty of Ridge does not.
same_as_lasso = ElasticNet(alpha=0.3, l1_ratio=1.0, **settings).fit(A_train, y_train)
ridge_end = ElasticNet(alpha=0.3, l1_ratio=0.0, **settings).fit(A_train, y_train)
ridge = Ridge(alpha=0.15, **settings).fit(A_train, y_train)
print("l1_ratio=1 against Lasso:", float(np.abs(same_as_lasso.coef_ - lasso.coef_).max()))
print("l1_ratio=0 against Ridge(alpha/2):", float(np.abs(ridge_end.coef_ - ridge.coef_).max()))

# %%
# Choose ``l1_ratio``
# -------------------
#
# Between the ends the weights move from the Ridge shape to the Lasso shape. A larger
# ``l1_ratio`` drops more features, and a smaller one keeps them and spreads the weight over
# the group. The test score is the price of the choice.
ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
path = []
for ratio in ratios:
    fitted = ElasticNet(alpha=0.3, l1_ratio=ratio, **settings).fit(A_train, y_train)
    path.append(fitted.coef_)
    kept_count = int((np.abs(fitted.coef_) >= negligible).sum())
    print(
        f"l1_ratio={ratio}: {kept_count} of {len(names)} features kept, "
        f"test R squared {fitted.score(A_test, y_test):.3f}"
    )

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.6)
for column, name in enumerate(names):
    ax.plot(ratios, np.array(path)[:, column], marker="o", markersize=3, label=name)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("l1_ratio")
ax.set_ylabel("weight")
ax.set_title("From Ridge (0) to Lasso (1)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6)
plt.show()

# %%
# A pitfall: stopping too early
# -----------------------------
#
# The loop stops when the loss changes by less than ``tol``. With features that move together
# the prediction settles long before the weights do: many different splits of the work between
# ``surface_area`` and ``roof_area`` predict almost the same. The table shows the same fit with
# four values of ``tol``. The test score hardly moves, but the weight of ``surface_area`` goes
# from about -1.1 to -0.6 as the loop is allowed to run longer. Do not read the single weights
# of correlated features too literally, and loosen or tighten ``tol`` to see how much they
# depend on where the loop stopped.
column = names.index("surface_area")
for tol in (1e-3, 1e-4, 1e-6, 1e-8):
    stopped = ElasticNet(alpha=0.3, l1_ratio=0.5, learning_rate=0.01, max_iter=20000, tol=tol).fit(
        A_train, y_train
    )
    print(
        f"tol={tol}: {stopped.n_iter_} iterations, test R squared "
        f"{stopped.score(A_test, y_test):.3f}, weight of surface_area {stopped.coef_[column]:.2f}"
    )
