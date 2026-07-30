"""Tests for confidence-aware answer guidance in the reasoning engine.

All LLM-free: they exercise prompt construction directly and the no-API-key
fallback path, so no network or Groq client is involved.
"""

from __future__ import annotations

from contextweave.reasoning.engine import ReasoningEngine
from contextweave.reasoning.prompts import LOW_CONFIDENCE_GUIDANCE, LOW_CONFIDENCE_NOTE
from contextweave.schemas import QueryResult, SourceType
from contextweave.timeutils import utcnow


def result(
    chunk_id: str, relevance: float, content: str = "A note about the budget review."
) -> QueryResult:
    return QueryResult(
        chunk_id=chunk_id,
        content=content,
        score=relevance,
        relevance=relevance,
        source=SourceType.NOTE,
        timestamp=utcnow(),
        entities=[],
    )


class TestPromptGuidance:
    def test_low_confidence_prompt_gets_hedge(self):
        engine = ReasoningEngine(api_key="unused")
        prompt = engine._build_prompt("what's my plan?", [result("a", 0.05)], "general", 0.1)
        assert LOW_CONFIDENCE_GUIDANCE.strip() in prompt

    def test_high_confidence_prompt_has_no_hedge(self):
        engine = ReasoningEngine(api_key="unused")
        prompt = engine._build_prompt("what's my plan?", [result("a", 0.9)], "general", 0.9)
        assert LOW_CONFIDENCE_GUIDANCE.strip() not in prompt


class TestContextFormatting:
    def test_shows_match_relevance_not_decayed_score(self):
        # An older-but-on-topic memory: low decayed score, high match relevance.
        # The model must see the relevance, or it wrongly distrusts good context.
        r = QueryResult(
            chunk_id="a",
            content="The cloud budget was cut by twenty percent.",
            score=0.05,
            relevance=0.9,
            source=SourceType.NOTE,
            timestamp=utcnow(),
            entities=["Budget"],
        )
        out = ReasoningEngine._format_context([r])
        assert "Relevance: 0.90" in out
        assert "Relevance: 0.05" not in out


class TestFallbackGuidance:
    def test_fallback_flags_weak_context(self):
        # No API key => fallback path; weak relevance => low confidence => caution.
        engine = ReasoningEngine(api_key="")
        resp = engine.reason("what's my plan?", [result("a", 0.05)])
        assert resp.confidence < 0.35
        assert LOW_CONFIDENCE_NOTE.strip() in resp.answer

    def test_fallback_confident_answer_has_no_caution(self):
        engine = ReasoningEngine(api_key="")
        strong = [result(f"c{i}", 0.9) for i in range(5)]
        resp = engine.reason("what's my plan?", strong)
        assert resp.confidence >= 0.35
        assert LOW_CONFIDENCE_NOTE.strip() not in resp.answer
