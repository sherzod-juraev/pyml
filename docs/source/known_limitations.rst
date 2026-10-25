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

Library behaviour
-----------------

.. _limitation-param-validation:

Hyper-parameters are checked for range, not for type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fit`` rejects out-of-range hyper-parameters with ``InvalidParameterError``, for example a
negative ``alpha`` or ``n_clusters=0``. It does not check types, so ``n_clusters=2.5`` or
``max_iter="10"`` is not reported by a dedicated error. Adding type checks to every
constructor was left out to keep the estimators small. ``fit_intercept`` and ``outlier_label``
are not validated either, since any value is meaningful for them.

.. _limitation-split-errors:

``train_test_split`` raises ``ValueError``, not ``InvalidParameterError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``train_test_split`` is a function, not an estimator, so it does not use the validation hook.
It validates its own arguments and raises ``ValueError`` (and ``TypeError``) for bad input, not
``InvalidParameterError``. ``InvalidParameterError`` is not a subclass of ``ValueError``, so
code that handles errors from both an estimator's ``fit`` and ``train_test_split`` has to catch
both. A NaN ``test_size`` or a negative ``random_state`` is also reported by a plain
``ValueError``. This may be unified in a later release.
