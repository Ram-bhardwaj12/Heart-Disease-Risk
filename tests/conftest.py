"""Pytest configuration for test suite stability on Windows."""

import gc
import matplotlib

matplotlib.use("Agg")

import pytest


@pytest.fixture(autouse=True)
def sync_gc():
    """Disable automatic asynchronous cyclic GC during test execution to prevent Matplotlib C-extension collisions."""
    gc.disable()
    yield
    gc.collect()

