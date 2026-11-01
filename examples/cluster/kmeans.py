"""
K Means
=======

:class:`~pyml.cluster.KMeans` splits the samples into ``n_clusters`` groups. It puts a centroid
in each group, assigns every sample to its nearest centroid, moves each centroid to the mean of
its samples, and repeats until the centroids stop moving.

Use it when you expect roughly round, similarly sized groups and you know how many to look for.
It always assigns every sample to a cluster, even an outlier. For groups of any shape, and for
noise, see :class:`~pyml.cluster.DBSCAN`.
"""

# %%
# Parameters
# ----------
#
# * ``n_clusters`` is how many groups to form. It must be chosen in advance.
# * ``max_iter`` and ``tol`` decide when to stop. The loop runs at most ``max_iter`` rounds,
#   and stops earlier when the centroids move less than ``tol`` in total.
# * ``random_state`` seeds the k-means++ start, so the same seed gives the same result.
#
# The full list, with the default values, is in the :class:`API reference
# <pyml.cluster.KMeans>`.

# %%
# Dataset
# -------
#
# The model never sees the cultivar of a wine. It is only used at the end to check what K Means
# found.
#
# .. dataset-info:: wine

# %%
# Standardize the features
# ------------------------
#
# K Means measures distances, so a feature with large numbers decides the result. Every feature
# is first scaled to zero mean and unit variance. The last section shows what happens without
# this step.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyml_datasets import load_wine

from pyml.cluster import KMeans
from pyml.preprocessing import StandardScaler

data = load_wine()
X, y = data.X, data.y
X_scaled = StandardScaler().fit_transform(X)


# %%
# Fit the model
# -------------
#
# ``fit`` finds the centroids and ``labels_`` holds the cluster of every sample. The model works
# on all 13 features, the plot shows two of them.
model = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
print("iterations:", model.n_iter_)

names = data.feature_names
columns = [names.index("flavanoids"), names.index("color_intensity")]

fig, ax = plt.subplots()
for cluster in range(3):
    points = X_scaled[model.labels_ == cluster][:, columns]
    ax.scatter(
        points[:, 0], points[:, 1], s=30, alpha=0.8, color=f"C{cluster}", label=f"Cluster {cluster}"
    )
ax.scatter(
    model.centroids_[:, columns[0]],
    model.centroids_[:, columns[1]],
    marker="X",
    s=200,
    color="C3",
    label="Centroids",
)
ax.set_xlabel("flavanoids (standardized)")
ax.set_ylabel("color_intensity (standardized)")
ax.set_title("Three clusters on two of the 13 features")
ax.legend(loc="lower right")
plt.show()

# %%
# Compare with the cultivars
# --------------------------
#
# K Means never saw the cultivar of a wine. Each row below is a cluster, each column a cultivar.
# A cluster that holds one cultivar only is a perfect group. The purity is the share of samples
# that belong to the most common cultivar of their cluster.


def cluster_class_counts(labels: NDArray[np.integer[Any]], y: NDArray[Any]) -> NDArray[np.int_]:
    """Count the samples of every cultivar in every cluster."""
    counts = np.zeros((labels.max() + 1, y.max() + 1), dtype=np.int_)
    np.add.at(counts, (labels, y), 1)
    return counts


counts = cluster_class_counts(model.labels_, y)
print(counts)
print(f"purity: {counts.max(axis=1).sum() / len(y):.2f}")

# %%
# Choose ``n_clusters``
# ---------------------
#
# The number of clusters is not known in real problems. One way to look for it is to fit the
# model for several values and plot the within-cluster sum of squares, the quantity that K Means
# minimizes. It always falls as clusters are added, so look for the point where the fall slows
# down. Here that is at three, the number of cultivars.
candidates = list(range(1, 9))
spread = []
for k in candidates:
    fitted = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    spread.append(float(((X_scaled - fitted.centroids_[fitted.labels_]) ** 2).sum()))

fig, ax = plt.subplots()
ax.plot(candidates, spread, marker="o")
ax.set_xlabel("n_clusters")
ax.set_ylabel("within-cluster sum of squares")
ax.set_title("The fall slows down after three clusters")
plt.show()

# %%
# A pitfall: skipping the scaling
# -------------------------------
#
# The same model on the raw features finds worse groups. ``proline`` ranges over more than a
# thousand units while most other features stay below twenty, so it alone decides which centroid
# is the nearest.
raw_model = KMeans(n_clusters=3, random_state=42).fit(X)
raw_counts = cluster_class_counts(raw_model.labels_, y)
print(raw_counts)
print(f"purity without scaling: {raw_counts.max(axis=1).sum() / len(y):.2f}")
