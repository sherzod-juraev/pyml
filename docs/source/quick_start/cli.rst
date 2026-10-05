Nox sessions
============

This project uses `Nox <https://nox.thea.codes/>`_ to run all quality
checks in isolated environments — the same checks that run in CI.

Nox creates a separate virtual environment for each session, installs
the required dependencies, and runs the configured commands. This
guarantees reproducibility across machines and Python versions.

Setup
-----

Install the ``nox`` extra to get Nox:

.. code-block:: bash

    git clone https://github.com/sherzod-juraev/pyml.git
    cd pyml
    pip install -e ".[nox]"

Run all default sessions
------------------------

.. code-block:: bash

    nox

This runs every session listed in ``nox.options.sessions``.

List all available sessions
---------------------------

.. code-block:: bash

    nox -l

This prints every session Nox can run, including parametrized
combinations such as ``check-3.13(pyml)`` and ``check-3.13(tests)``.

Run a specific session
----------------------

.. code-block:: bash

    nox -s <session-name>

Replace ``<session-name>`` with one of the sessions listed by
``nox -l``.
