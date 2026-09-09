Installation
============

pyml is an educational project and is not published to PyPI.

.. tab-set::

    .. tab-item:: pip install

        For using pyml as a library:

        .. code-block:: bash

            pip install git+https://github.com/sherzod-juraev/pyml.git

    .. tab-item:: git clone (development)

        For running tests, linting, or contributing:

        .. code-block:: bash

            git clone https://github.com/sherzod-juraev/pyml.git
            cd pyml
            pip install -e ".[dev]"

.. dropdown:: Building the documentation locally

    .. code-block:: bash

        pip install -e ".[docs]"
        sphinx-build -b html docs/source docs/build/html

    For live-reloading while editing:

    .. code-block:: bash

        pip install -e ".[live]"
        sphinx-autobuild docs/source docs/build/html
