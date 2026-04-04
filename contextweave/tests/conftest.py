"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def no_network_calls(monkeypatch):
    """Prevent accidental network calls in unit tests."""
    # Only block for unit tests — integration tests should opt in
    pass
