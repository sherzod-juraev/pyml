"""
DBSCAN
======

:class:`~pyml.cluster.DBSCAN` finds groups as dense regions. A sample with enough neighbors
close to it is a core point, and the core points that touch each other form one cluster, with
the samples at its edge. A sample that belongs to no dense region is not forced into a cluster,
it is labelled noise (``-1``).

Use it when you do not know how many groups to look for, when the groups can have any shape, or
when some samples are outliers. It struggles when the groups have very different densities,
because one distance ``eps`` has to suit all of them. For round groups and a known number of
groups, see :class:`~pyml.cluster.KMeans`.
"""

# %%
# Parameters
# ----------
#
# * ``eps`` is the radius of a neighborhood. Two samples closer than ``eps`` are neighbors.
# * ``min_samples`` is how many samples (the sample itself included) a neighborhood needs to
#   count as dense. A sample with that many neighbors is a core point.
# * ``metric`` is how the distance is measured.
#
# There is no ``n_clusters``: the number of clusters is a result, not a setting. The full list,
# with the default values, is in the :class:`API reference <pyml.cluster.DBSCAN>`.

# %%
# Dataset
# -------
#
# The model never sees the species of a penguin. It is only used at the end to check what
# DBSCAN found. The two penguins with missing measurements are left out, and so are ``island``
# and ``sex``, which are categories stored as integer codes, not distances. The four body
# measurements remain.
#
# .. dataset-info:: penguins

# %%
# Standardize the features
# ------------------------
#
# DBSCAN measures distances, so a feature with large numbers decides the result. Every feature
# is first scaled to zero mean and unit variance. The last section shows what happens without
# this step.
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pyml_datasets import load_penguins
from scipy.spatial.distance import cdist

from pyml.cluster import DBSCAN
from pyml.preprocessing import StandardScaler

data = load_penguins()
names = data.feature_names
measurements = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = data.X[:, [names.index(name) for name in measurements]]
complete = ~np.isnan(X).any(axis=1)
X, y = X[complete], data.y[complete]
X_scaled = StandardScaler().fit_transform(X)
print(f"{X.shape[0]} penguins, {X.shape[1]} measurements")

# %%
# Fit the model
# -------------
#
# ``labels_`` holds the cluster of every sample, and ``-1`` marks the noise. The model works on
# all four measurements, the plot shows two of them.
model = DBSCAN(eps=0.6, min_samples=10).fit(X_scaled)
labels = model.labels_
n_clusters = int(labels.max()) + 1
print("clusters:", n_clusters)
print("noise points:", int((labels == -1).sum()))


def plot_clusters(ax: Axes, x_name: str, y_name: str) -> None:
    """Draw the clusters and the noise on two of the standardized measurements."""
    x, y_ = measurements.index(x_name), measurements.index(y_name)
    noise = labels == -1
    ax.scatter(
        X_scaled[noise][:, x], X_scaled[noise][:, y_], marker="x", s=30, color="gray", label="Noise"
    )
    for cluster in range(n_clusters):
        points = X_scaled[labels == cluster]
        ax.scatter(
            points[:, x],
            points[:, y_],
            s=30,
            alpha=0.8,
            color=f"C{cluster}",
            label=f"Cluster {cluster}",
        )
    ax.set_xlabel(f"{x_name} (standardized)")
    ax.set_ylabel(f"{y_name} (standardized)")


fig, ax = plt.subplots()
plot_clusters(ax, "bill_length_mm", "flipper_length_mm")
ax.set_title("Dense regions and the points left out")
ax.legend(loc="lower right")
plt.show()

# %%
# Why some noise looks misplaced
# ------------------------------
#
# Some gray crosses sit in the middle of a cluster. That is the plot, not the model: DBSCAN
# measures the distance in all four measurements, and a plot shows two of them. Below is the
# same fit on the other two, with the same colors. A point that looks inside a cluster in one
# view can sit at the edge in the other, and neither view shows the distance DBSCAN uses.
fig, ax = plt.subplots()
plot_clusters(ax, "bill_depth_mm", "body_mass_g")
ax.set_title("The same fit, seen on the other two measurements")
plt.show()

# %%
# The numbers say the same. ``eps=0.6`` and ``min_samples=10`` ask for ten samples within
# reach. A noise point has far fewer, and none of its neighbors is a core point, so no cluster
# can reach it:
pairwise = cdist(X_scaled, X_scaled)
neighbors = (pairwise <= model.eps).sum(axis=1)
print("samples within eps, noise points (median):", int(np.median(neighbors[labels == -1])))
print("samples within eps, clustered points (median):", int(np.median(neighbors[labels >= 0])))

# %%
# Compare with the species
# ------------------------
#
# Each row below is a cluster, and the last row is the noise. Each column is a species. A row
# that holds one species only is a clean group.


def cluster_species_counts(labels: NDArray[np.integer[Any]], y: NDArray[Any]) -> NDArray[np.int_]:
    """Count the samples of every species in every cluster, the noise in the last row."""
    n_clusters = int(labels.max()) + 1
    rows = np.where(labels < 0, n_clusters, labels)
    counts = np.zeros((n_clusters + 1, y.max() + 1), dtype=np.int_)
    np.add.at(counts, (rows, y), 1)
    return counts


print(data.target_names)
print(cluster_species_counts(labels, y))

# %%
# Gentoo is found as one cluster and Adelie as another. Chinstrap is split in two and
# partly left as noise, because its measurements overlap with those of Adelie and the density
# between them is not low enough to tell them apart. That is a property of the data, not
# a failure of the model: DBSCAN only sees where the points are dense.

# %%
# Choose ``eps``
# --------------
#
# ``eps`` has to match the scale of the data, and the scale is known after the standardizing.
# A way to look for it: for every sample, take the distance at which ``min_samples`` samples
# (itself included) are within reach, and sort these distances. Samples in a dense region have
# a small one, noise has a large one. The curve rises slowly and then steeply, and the usual
# advice is to take ``eps`` near the bend. Here the bend is close to 1.0. It is a starting point,
# not the answer: the next step shows what a smaller value does.
min_samples = 10
distances = np.sort(pairwise, axis=1)[:, min_samples - 1]

fig, ax = plt.subplots()
ax.plot(np.sort(distances))
ax.axhline(0.6, color="C3", linestyle="--", label="eps = 0.6")
ax.axhline(1.0, color="C1", linestyle="--", label="eps = 1.0")
ax.set_xlabel("samples, sorted by distance")
ax.set_ylabel(f"distance that reaches {min_samples} samples")
ax.set_title("The curve bends upwards close to 1.0")
ax.legend(loc="upper left")
plt.show()

# %%
# The choice matters. At the bend, ``eps=1.0``, almost nothing is noise but Adelie and Chinstrap
# are one cluster. The ``eps=0.6`` used above separates more, and pays for it with noise. A
# still smaller ``eps`` leaves most samples as noise:
for eps in (0.4, 0.5, 0.6, 0.8, 1.0):
    fitted = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    print(
        f"eps={eps}: {int(fitted.labels_.max()) + 1} clusters, "
        f"{int((fitted.labels_ == -1).sum())} noise points"
    )

# %%
# A pitfall: skipping the scaling
# -------------------------------
#
# The same model on the raw measurements finds nothing. ``body_mass_g`` is in thousands of grams
# while the other three stay below 250 millimeters, so it alone decides which samples are
# neighbors. With ``eps=0.6`` no sample has enough neighbors, and everything is noise.
raw_model = DBSCAN(eps=0.6, min_samples=10).fit(X)
print("clusters without scaling:", int(raw_model.labels_.max()) + 1)
print("noise points without scaling:", int((raw_model.labels_ == -1).sum()))
