# -*- coding: utf-8 -*-
"""
pytest configuration for PRyMordial-nu regression tests.

Ensures the project root is on sys.path and the working directory is set
so tests can be invoked from anywhere (e.g. `pytest tests/`).

Adds a custom marker `slow` for tests that exercise the full Boltzmann/QKE
pipeline (~130 s each). Skip these during fast iteration with
`pytest -m "not slow"`.
"""
import os
import sys

# Resolve project root regardless of pytest invocation directory
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Some of the PRyMordial modules open data files with relative paths at
# import time; make sure CWD is the repo root during the test session.
os.chdir(_root)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tests that take >30 seconds (Boltzmann/QKE runs)"
    )
