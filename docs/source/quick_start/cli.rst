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

.. warning::
    ``pyml docs live`` is experimental — see :ref:`limitation-docs-live-experimental`.

.. warning::
    ``docs check``/``build``/``live`` require a source checkout — see
    :ref:`limitation-source-checkout-required`.

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

.. warning::
    Requires a source checkout — see :ref:`limitation-source-checkout-required`.
