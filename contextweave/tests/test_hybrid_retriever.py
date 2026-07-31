"""Unit tests for hybrid-retriever helpers."""

from __future__ import annotations

import pytest

from contextweave.retrieval.hybrid_retriever import _normalize_fts_rank, fusion_weights


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


class TestFusionWeights:
    def test_weights_sum_to_one(self):
        # Scores must stay comparable across intents, so weights always sum to 1.
        for intent in ("general", "cross_reference", "patterns", "temporal", "priorities"):
            assert sum(fusion_weights(intent)) == pytest.approx(1.0)

    def test_connection_intents_weight_graph_more(self):
        _, _, general_graph = fusion_weights("general")
        for intent in ("cross_reference", "patterns"):
            _, _, graph_w = fusion_weights(intent)
            assert graph_w > general_graph, f"{intent} should lean on the graph"

    def test_unknown_intent_falls_back_to_general(self):
        assert fusion_weights("nonsense") == fusion_weights("general")

    def test_a_graph_only_hit_ranks_higher_under_cross_reference(self):
        # Same signals, different intent: a purely graph-connected chunk earns a
        # higher fused score when the query is about connections.
        def fuse(intent, vector, fts, graph):
            wv, wf, wg = fusion_weights(intent)
            return wv * vector + wf * fts + wg * graph

        graph_only = (0.0, 0.0, 1.0)
        assert fuse("cross_reference", *graph_only) > fuse("general", *graph_only)
