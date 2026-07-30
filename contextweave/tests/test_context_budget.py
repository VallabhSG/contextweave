"""Tests for token-budgeted, redundancy-aware context assembly.

These assert the *properties* of context selection — budget respected,
near-duplicates suppressed, the top memory always survives, and confidence
tracks relevance rather than raw result count — not the plumbing.
"""

from __future__ import annotations

from contextweave.reasoning.context_budget import (
    ContextBudgeter,
    _jaccard,
    _tokenize_words,
    estimate_tokens,
)
from contextweave.schemas import QueryResult, SourceType
from contextweave.timeutils import utcnow


def make_result(
    chunk_id: str, content: str, score: float = 0.5, relevance: float | None = None
) -> QueryResult:
    return QueryResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        relevance=score if relevance is None else relevance,
        source=SourceType.NOTE,
        timestamp=utcnow(),
        entities=[],
    )


def distinct(prefix: str, words: int) -> str:
    """Content whose words share nothing with a different prefix (never redundant)."""
    return " ".join(f"{prefix}w{j}" for j in range(words))


class TestTokenHelpers:
    def test_estimate_tokens_scales_with_length(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("a") == 1
        assert estimate_tokens("x" * 400) == 100

    def test_jaccard_bounds(self):
        a = _tokenize_words("the quarterly budget review")
        assert _jaccard(a, a) == 1.0
        assert _jaccard(a, _tokenize_words("sourdough starter rye flour")) == 0.0
        assert 0.0 < _jaccard(a, _tokenize_words("the quarterly numbers")) < 1.0


class TestBudget:
    def test_packs_within_budget_and_drops_the_rest(self):
        results = [make_result(f"c{i}", distinct(f"t{i}", 45), score=0.9) for i in range(12)]
        budgeter = ContextBudgeter(token_budget=200)

        assembled = budgeter.assemble(results)

        assert 0 < len(assembled.results) < len(results), "some, not all, should be packed"
        assert assembled.dropped_for_budget > 0
        assert assembled.token_estimate <= 200 + 8, "budget respected (slack for truncation marker)"

    def test_top_memory_survives_even_when_it_exceeds_budget(self):
        huge = make_result("big", distinct("huge", 2000), score=0.7)
        budgeter = ContextBudgeter(token_budget=50)

        assembled = budgeter.assemble([huge])

        assert len(assembled.results) == 1, "the single best memory must never be dropped whole"
        assert assembled.results[0].content.endswith("…[truncated]")
        assert "big" in assembled.truncated_ids

    def test_empty_input_is_empty_output(self):
        assembled = ContextBudgeter().assemble([])
        assert assembled.results == []
        assert assembled.confidence == 0.0
        assert assembled.token_estimate == 0


class TestRedundancy:
    def test_near_duplicate_is_suppressed(self):
        base = "The quarterly budget review covers cloud spend and headcount planning for Q3"
        results = [
            make_result("a", base, score=0.9),
            make_result("b", base + " as well", score=0.8),  # near-identical
            make_result("c", distinct("sourdough", 12), score=0.7),  # distinct topic
        ]
        budgeter = ContextBudgeter(token_budget=5000, redundancy_threshold=0.82)

        assembled = budgeter.assemble(results)

        packed_ids = {r.chunk_id for r in assembled.results}
        assert assembled.dropped_for_redundancy == 1
        assert packed_ids == {"a", "c"}, "the near-duplicate of 'a' should be dropped, 'c' kept"

    def test_distinct_memories_all_survive(self):
        results = [make_result(f"c{i}", distinct(f"topic{i}", 10), score=0.8) for i in range(5)]
        assembled = ContextBudgeter(token_budget=5000).assemble(results)
        assert len(assembled.results) == 5
        assert assembled.dropped_for_redundancy == 0


class TestConfidenceCalibration:
    def test_confidence_tracks_relevance_not_count(self):
        """The core anti-regression: eight weak hits must not read as certain.

        The old formula returned min(1.0, len(results)/8 * 0.8 + 0.2) == 1.0 for
        eight results regardless of relevance. Confidence must now be low.
        """
        weak = [make_result(f"w{i}", distinct(f"weak{i}", 8), score=0.02) for i in range(8)]
        assembled = ContextBudgeter(token_budget=5000).assemble(weak)
        assert len(assembled.results) == 8, "all distinct, so all packed"
        assert assembled.confidence < 0.15, "eight irrelevant memories are not high confidence"

    def test_stronger_context_is_more_confident(self):
        strong = [make_result(f"s{i}", distinct(f"strong{i}", 8), score=0.9) for i in range(5)]
        weak = [make_result(f"w{i}", distinct(f"weak{i}", 8), score=0.2) for i in range(5)]
        budgeter = ContextBudgeter(token_budget=5000)

        strong_conf = budgeter.assemble(strong).confidence
        weak_conf = budgeter.assemble(weak).confidence

        assert strong_conf > weak_conf
        assert strong_conf > 0.8

    def test_single_strong_hit_is_reasonably_confident(self):
        one = [make_result("s", distinct("solo", 20), score=0.85)]
        conf = ContextBudgeter(token_budget=5000).assemble(one).confidence
        assert 0.4 < conf < 0.85, "a lone strong match is moderately, not maximally, confident"

    def test_confidence_uses_relevance_not_decayed_score(self):
        """Confidence must reflect match quality, not the decay-tuned ranking score.

        A memory that is highly relevant but heavily decayed (low `score`, e.g. an
        old note under default decay) should still read as confident context —
        otherwise confidence swings purely on how ranking was tuned or how a query
        was classified.
        """
        decayed_but_relevant = [
            make_result(f"d{i}", distinct(f"topic{i}", 8), score=0.05, relevance=0.9)
            for i in range(3)
        ]
        conf = ContextBudgeter(token_budget=5000).assemble(decayed_but_relevant).confidence
        assert conf > 0.7, "confidence should track pre-decay relevance, not the decayed score"

    def test_breadth_raises_confidence_at_equal_peak(self):
        budgeter = ContextBudgeter(token_budget=5000)
        one = [make_result("s", distinct("solo", 8), score=0.8)]
        many = [make_result(f"s{i}", distinct(f"many{i}", 8), score=0.8) for i in range(5)]
        assert budgeter.assemble(many).confidence > budgeter.assemble(one).confidence
