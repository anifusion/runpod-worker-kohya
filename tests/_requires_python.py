"""Fail with a clear message when the tests run on an unsupported Python.

The suite follows the worker's minimum supported Python version. Failing early
keeps local checks aligned with the interpreter used by training dependencies.
"""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "These tests require Python 3.11 or later, the version used by the worker image. "
        f"Found {sys.version_info.major}.{sys.version_info.minor}."
    )
