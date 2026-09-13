Development CLI
================

.. code-block:: bash

    git clone https://github.com/sherzod-juraev/pyml.git
    cd pyml
    pip install -e ".[dev]"

pyml ships with a small ``pyml`` command for running its own quality
checks — the same checks that run in CI.

.. code-block:: bash

    pyml --version
    pyml --help

.. code-block:: text

    pyml check code    # ruff, mypy, interrogate
    pyml check docs    # sphinx-lint, doc8, rstcheck, linkcheck, sphinx-build
    pyml check tests   # ruff, mypy, pytest
    pyml check all     # everything above
    pyml check         # same as `pyml check all`

Each subcommand prints a PASS/FAIL summary; re-run the specific
underlying tool directly (e.g. ``ruff check pyml``) to see a failing
step's full output.

.. note::
    ``pyml check docs`` and ``pyml check tests`` only work from a
    source checkout of the repository — ``docs/`` and ``tests/`` are
    excluded from the installed package, so these subcommands exit
    with a clear error if run after a plain ``pip install``.

.. only:: html

    .. dropdown:: Building the documentation locally

        .. code-block:: bash

            pip install -e ".[docs]"
            sphinx-build -b html docs/source docs/build/html

        For live-reloading while editing:

        .. code-block:: bash

            pip install -e ".[live]"
            sphinx-autobuild docs/source docs/build/html
