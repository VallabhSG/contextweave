"""Tests for shared query-intent detection."""

from __future__ import annotations

from contextweave.reasoning.query_intent import detect_query_type


class TestDetectQueryType:
    def test_temporal(self):
        assert detect_query_type("how has my thinking evolved over time?") == "temporal"

    def test_patterns(self):
        assert detect_query_type("what recurring pattern do you notice?") == "patterns"

    def test_gaps(self):
        assert detect_query_type("what am I avoiding or overlooking?") == "gaps"

    def test_cross_reference(self):
        assert (
            detect_query_type("what is the relationship between Alice and Project Alpha?")
            == "cross_reference"
        )

    def test_priorities(self):
        assert detect_query_type("what should I focus on this week?") == "priorities"

    def test_defaults_to_general(self):
        assert detect_query_type("what did I have for lunch on Tuesday") == "general"

    def test_empty_is_general(self):
        assert detect_query_type("") == "general"

    def test_strongest_signal_wins_on_tie_break(self):
        # Contains a temporal hint ("changed") and a priorities hint ("focus");
        # two temporal hints ("changed", "over time") should win.
        assert detect_query_type("how has my focus changed over time?") == "temporal"
