"""Fail with a clear message when the tests run on an unsupported Python.

The worker image runs Python 3.10, and the handler modules use annotation syntax
that older interpreters reject. Importing this module first turns a confusing
TypeError into an explicit requirement.
"""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError(
        "These tests require Python 3.10 or later, the version used by the worker image. "
        f"Found {sys.version_info.major}.{sys.version_info.minor}."
    )
