"""Root exception hierarchy for pyml.

This module defines :class:`PymlError`, the base class from which all
other exceptions in the library derive. Catching ``PymlError`` allows
consumers to handle any pyml-specific failure without accidentally
suppressing unrelated exceptions.
"""


class PymlError(Exception):
    """Base class for all exceptions raised by pyml.

    This allows users to catch any library-specific error with a
    single ``except PymlError`` clause, without accidentally catching
    unrelated exceptions (e.g. from NumPy).
    """
