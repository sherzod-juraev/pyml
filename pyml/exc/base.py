r"""Base exception class for the pyml library.

This module defines the root exception that all custom pyml
exceptions inherit from, allowing users to catch all library-specific
errors with a single except clause.

Classes
-------
PymlError
    Base exception for all pyml-specific errors.

Examples
--------
>>> from pyml.exc import PymlError
>>>
>>> try:
...     raise PymlError("Something went wrong")
... except PymlError as e:
...     print(e)
Something went wrong
"""


class PymlError(Exception):
    r"""Base class for all pyml exceptions.

    All custom exceptions in the pyml library inherit from this class,
    enabling users to catch any library-specific error with:

    .. code-block:: python

        try:
            model.fit(X, y)
        except PymlError as e:
            print(f"pyml error: {e}")

    This follows the convention established by standard libraries
    where a root exception provides a single catch point for all
    package-specific errors.
    """

    pass
