"""Time utilities shared across ContextWeave."""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Current UTC time as a naive datetime.

    Timestamps are stored and compared as naive UTC throughout the codebase;
    an aware datetime here would make them incomparable with stored values.
    Replaces the deprecated ``datetime.utcnow()``.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
