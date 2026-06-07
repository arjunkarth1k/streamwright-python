"""Bootstrap smoke test: the package imports and exposes a version.

A minimal guard so the suite is green from the first commit; the full
test suite lands module by module as the build progresses.
"""

from __future__ import annotations

import streamwright


def test_package_exposes_version() -> None:
    assert streamwright.__version__
