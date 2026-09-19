Development CLI
================

.. code-block:: bash

    git clone https://github.com/sherzod-juraev/pyml.git
    cd pyml
    pip install -e ".[dev]"

pyml ships with a small ``pyml`` command for running its own quality
checks and building the docs — the same checks that run in CI.

.. code-block:: bash

    pyml --version    # or: pyml -V
    pyml --help       # or: pyml -h

.. code-block:: text

    pyml code check    # ruff, mypy, interrogate
    pyml docs check    # sphinx-lint, doc8, rstcheck, linkcheck, sphinx-build
    pyml docs build    # build the HTML docs
    pyml docs live     # build and serve the docs with live-reload
    pyml tests check   # ruff, mypy, pytest
    pyml check         # code + docs + tests checks, in sequence

Each ``check`` subcommand prints a PASS/FAIL summary; re-run the
specific underlying tool directly (e.g. ``ruff check pyml``) to see a
failing step's full output. ``pyml docs build`` and ``pyml docs live``
stream their own output live instead, since they aren't checks.

.. code-block:: bash

    pyml docs build --fresh   # discard the cached environment and
                              # rewrite every output file (-E -a)

.. note::
    ``pyml docs check``, ``pyml docs build``, ``pyml docs live``, and
    ``pyml tests check`` only work from a source checkout of the
    repository — ``docs/`` and ``tests/`` are excluded from the
    installed package, so these commands exit with a clear error if
    run after a plain ``pip install``. ``pyml docs live`` additionally
    requires the ``docs`` extras (``pip install -e ".[docs]"``) for
    ``sphinx-autobuild``.

.. note::
    ``pyml docs live`` is marked experimental, not stable. We fixed
    one real problem during local testing — on Windows, ``Ctrl+C``
    could leave ``sphinx-autobuild`` running in the background and
    holding port 8000, which the command now handles by force-killing
    the whole process tree on interrupt. That fix has been verified
    on Windows and should hold on Unix-like systems too, but other
    environment-specific issues may still surface that haven't been
    caught yet. If the terminal doesn't return promptly after
    ``Ctrl+C``, check for an orphaned ``sphinx-autobuild`` process
    still holding the port.
