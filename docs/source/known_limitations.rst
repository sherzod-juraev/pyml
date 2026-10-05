Known Limitations
=================

This page documents deliberate trade-offs in pyml's tooling and
documentation — cases where the obvious fix wasn't taken, and why.

Documentation and site
----------------------

.. _limitation-plots-dark-mode:

Plots don't adapt to dark mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots throughout this documentation are static matplotlib images.
They render with a light background and dark text even when the site
is in dark mode. Inverting them with CSS was considered, but breaks
the color meaning in scatter plots (e.g. class colors get reassigned
unpredictably). Left as-is for now — everything else on the site
(page background, text, the logo, the 404 illustration) switches
normally between light and dark.

.. _limitation-branding-duplicated:

Branding files are duplicated, not shared
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``logo.svg`` and ``logo-dark.svg`` exist both in ``assets/branding/``
(used by the GitHub README's light/dark ``<picture>`` switch) and in
``docs/source/_static/branding/`` (used by Sphinx). This isn't an
oversight: pydata-sphinx-theme checks that the logo file exists
directly under Sphinx's own ``_static/`` before ``html_static_path``
extra entries are resolved, so a single shared location across
GitHub and Sphinx isn't possible. The two copies are kept in sync by
hand when the logo changes.

.. _limitation-api-reference-order:

API Reference isn't alphabetical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pages under :doc:`/api/pyml/index` are ordered
supervised-before-unsupervised-before-utilities (Core, Linear Model, Naive Bayes,
Neighbors, Tree, Cluster, Preprocessing, Model Selection, Metrics),
not alphabetically. The same order is used in :doc:`/quick_start/examples/index`
for consistency. Use the sidebar search or :kbd:`Ctrl+K` to jump
directly to a specific model instead of scanning the list.
