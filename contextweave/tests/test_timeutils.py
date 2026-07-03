"""Tests for the shared time helper."""

from __future__ import annotations

from datetime import datetime, timezone

from contextweave.timeutils import utcnow


def test_utcnow_is_naive():
    assert utcnow().tzinfo is None


def test_utcnow_tracks_utc_wall_clock():
    reference = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((utcnow() - reference).total_seconds()) < 5
