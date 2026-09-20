Development CLI
================

.. code-block:: bash

    git clone https://github.com/sherzod-juraev/pyml.git
    cd pyml
    pip install -e ".[dev]"

pyml ships with a small ``pyml`` command for running its own quality
checks and building the docs — the same checks that run in CI.

Overview
--------

.. code-block:: bash

    pyml --version # or pyml -V

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml --version

.. code-block:: bash

    pyml --help # or pyml -h

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml --help

Documentation commands
-----------------------

.. code-block:: bash

    pyml docs --help

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml docs --help

.. code-block:: bash

    pyml docs build --help

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml docs build --help

.. code-block:: bash

    pyml docs check --help

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml docs check --help

.. code-block:: bash

    pyml docs live --help

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml docs live --help

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

Code quality commands
----------------------

.. code-block:: bash

    pyml code --help

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml code --help

Test suite commands
---------------------

.. code-block:: bash

    pyml tests --help

.. dropdown:: Output
    :icon: terminal
    :color: light

    .. program-output:: pyml tests --help

.. note::
    ``pyml docs check``, ``pyml docs build``, ``pyml docs live``, and
    ``pyml tests check`` only work from a source checkout of the
    repository — ``docs/`` and ``tests/`` are excluded from the
    installed package, so these commands exit with a clear error if
    run after a plain ``pip install``. ``pyml docs live`` additionally
    requires the ``docs`` extras (``pip install -e ".[docs]"``) for
    ``sphinx-autobuild``.
