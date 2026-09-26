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

Development CLI
---------------

.. _limitation-docs-live-experimental:

``pyml docs live`` is experimental
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pyml docs live`` (``sphinx-autobuild``) is marked experimental, not
stable. One real problem was fixed during local testing — on
Windows, ``Ctrl+C`` could leave ``sphinx-autobuild`` running in the
background and holding port 8000, which the command now handles by
force-killing the whole process tree on interrupt. That fix has been
verified on Windows and should hold on Unix-like systems too, but
other environment-specific issues may still surface that haven't
been caught yet. If the terminal doesn't return promptly after
``Ctrl+C``, check for an orphaned ``sphinx-autobuild`` process still
holding the port.

.. _limitation-source-checkout-required:

Dev commands require a source checkout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pyml docs check``, ``pyml docs build``, ``pyml docs live``, and
``pyml tests check`` only work from a source checkout of the
repository — ``docs/`` and ``tests/`` are excluded from the
installed package, so these commands exit with a clear error if run
after a plain ``pip install``. ``pyml docs live`` additionally
requires the ``docs`` extras (``pip install -e ".[docs]"``) for
``sphinx-autobuild``.

.. _limitation-cli-inside-package:

Why the CLI lives inside the ``pyml`` package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dev CLI (code/docs/tests checks, docs build, live-reload) is
implemented inside ``pyml/_cli/`` and shipped as part of the package,
rather than as separate scripts outside it. This was a deliberate
choice, not an oversight: this is a personal, single-maintainer
project, so the usual argument for keeping tooling decoupled from the
library (avoiding bloat for downstream consumers, supporting multiple
tooling setups) doesn't carry much weight here. Coupling everything
into one ``pyml`` command instead means one entry point to remember,
one place to check when something's wrong, and no separate tooling
repo or script collection to keep in sync. It trades a small amount
of package purity for a meaningfully simpler day-to-day workflow —
the right trade for this project's scale.
