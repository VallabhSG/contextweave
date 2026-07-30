"""Unit tests for hybrid-retriever helpers."""

from __future__ import annotations

from contextweave.retrieval.hybrid_retriever import _normalize_fts_rank


class TestFtsNormalization:
    def test_zero_and_positive_rank_are_zero(self):
        # bm25 of 0 (or a positive/non-matching value) is no relevance.
        assert _normalize_fts_rank(0.0) == 0.0
        assert _normalize_fts_rank(3.0) == 0.0

    def test_midpoint_at_saturation_constant(self):
        # A match strength equal to k maps to 0.5 (rank == -k).
        assert _normalize_fts_rank(-5.0) == 0.5

    def test_monotonic_in_match_strength(self):
        weak = _normalize_fts_rank(-1.0)
        mid = _normalize_fts_rank(-5.0)
        strong = _normalize_fts_rank(-20.0)
        assert weak < mid < strong

    def test_bounded_below_one(self):
        # Even a very strong match saturates toward, but never reaches, 1.0 —
        # no hard clip cliff where many strong matches collapse to the same score.
        assert _normalize_fts_rank(-1000.0) < 1.0
        assert _normalize_fts_rank(-1000.0) > 0.99
