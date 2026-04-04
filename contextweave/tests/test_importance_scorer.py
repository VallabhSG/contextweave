"""Tests for the temporal decay importance scorer."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from contextweave.processing.importance_scorer import ImportanceScorer


@pytest.fixture
def scorer():
    return ImportanceScorer(
        half_life_days=30.0,
        access_boost_factor=1.2,
        connection_density_weight=0.3,
    )


class TestImportanceScorer:
    def test_recent_memory_scores_high(self, scorer):
        now = datetime.utcnow()
        score = scorer.score(base_importance=0.8, timestamp=now)
        assert score > 0.7

    def test_old_memory_decays(self, scorer):
        now = datetime.utcnow()
        old = now - timedelta(days=90)
        recent_score = scorer.score(base_importance=0.8, timestamp=now)
        old_score = scorer.score(base_importance=0.8, timestamp=old, now=now)
        assert old_score < recent_score

    def test_half_life_at_30_days(self, scorer):
        now = datetime.utcnow()
        thirty_days_ago = now - timedelta(days=30)
        score = scorer.score(base_importance=1.0, timestamp=thirty_days_ago, now=now)
        # At half-life, decay should be ~0.5 (before any boosts)
        assert 0.4 <= score <= 0.6

    def test_access_boost_increases_score(self, scorer):
        now = datetime.utcnow()
        base = scorer.score(base_importance=0.5, timestamp=now, access_count=0)
        boosted = scorer.score(base_importance=0.5, timestamp=now, access_count=10)
        assert boosted > base

    def test_connection_boost_increases_score(self, scorer):
        now = datetime.utcnow()
        base = scorer.score(base_importance=0.5, timestamp=now, connection_count=0)
        connected = scorer.score(base_importance=0.5, timestamp=now, connection_count=5)
        assert connected > base

    def test_score_capped_at_one(self, scorer):
        now = datetime.utcnow()
        score = scorer.score(
            base_importance=1.0,
            timestamp=now,
            access_count=100,
            connection_count=50,
        )
        assert score <= 1.0

    def test_score_non_negative(self, scorer):
        old = datetime(2000, 1, 1)
        score = scorer.score(base_importance=0.1, timestamp=old)
        assert score >= 0.0

    def test_estimate_base_importance_note_higher_than_browser(self, scorer):
        note_score = scorer.estimate_base_importance("Important project update", "note")
        browser_score = scorer.estimate_base_importance("visited google.com", "browser")
        assert note_score > browser_score

    def test_importance_signals_boost_score(self, scorer):
        plain = scorer.estimate_base_importance("Had a meeting today", "note")
        signal = scorer.estimate_base_importance(
            "IMPORTANT: action item - follow up on deadline", "note"
        )
        assert signal > plain

    def test_short_content_slightly_lower(self, scorer):
        short = scorer.estimate_base_importance("ok", "note")
        long = scorer.estimate_base_importance("word " * 250, "note")
        assert short < long
