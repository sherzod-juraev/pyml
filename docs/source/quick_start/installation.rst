Installation
============

Requirements
------------

* Python 3.12, 3.13 or 3.14
* NumPy 1.26 or newer and SciPy 1.11.2 or newer (installed automatically)

.. note::

   pyml is a personal project and is not published on PyPI, so it is installed directly from
   GitHub.

Install from GitHub
-------------------

.. code-block:: console

   $ pip install git+https://github.com/sherzod-juraev/pyml.git

To install a fixed release, add its tag after ``@``, replacing ``vX.Y.Z`` with a tag from the
`Releases <https://github.com/sherzod-juraev/pyml/releases>`_ page. Pinning a tag keeps your
results reproducible, because the code no longer changes under you.

.. code-block:: console

   $ pip install git+https://github.com/sherzod-juraev/pyml.git@vX.Y.Z

Example datasets
----------------

The examples in this documentation use the classic datasets from the companion package
`pyml-datasets <https://pyml-datasets.readthedocs.io/en/latest/>`_. It is not needed to use
pyml itself, only to run the examples as they are written:

.. code-block:: console

   $ pip install git+https://github.com/sherzod-juraev/pyml-datasets.git

Check the installation
----------------------

.. code-block:: console

   $ python -c "import pyml; print(pyml.__version__)"

This prints the version of the installed release.
