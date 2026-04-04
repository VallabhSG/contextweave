"""LLM reasoning engine that synthesizes answers from retrieved context."""

from __future__ import annotations

import logging

from contextweave.config import settings
from contextweave.reasoning.prompts import QUERY_TYPE_PROMPTS, SYSTEM_PROMPT
from contextweave.schemas import QueryResult, ReasoningResponse

logger = logging.getLogger(__name__)

# Keywords that hint at specific query types
QUERY_TYPE_HINTS = {
    "patterns": ["pattern", "trend", "recurring", "often", "usually", "tend to"],
    "gaps": ["avoiding", "missing", "overlooking", "neglecting", "forgot"],
    "temporal": ["evolved", "changed", "over time", "progression", "shift"],
    "cross_reference": ["think about", "opinion on", "what does", "relationship between"],
    "priorities": ["focus", "prioritize", "this week", "next", "should I", "what's important"],
}


class ReasoningEngine:
    """Synthesizes answers from retrieved context using Groq LLM."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or settings.groq_api_key
        self._model_name = settings.reasoning_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def reason(
        self,
        query: str,
        results: list[QueryResult],
        query_type: str | None = None,
    ) -> ReasoningResponse:
        """Generate a reasoned answer from query and retrieved context."""
        if not results:
            return ReasoningResponse(
                answer="I don't have enough context to answer this question. Try ingesting more data sources.",
                confidence=0.0,
                query_type=query_type or "general",
            )

        detected_type = query_type or self._detect_query_type(query)

        if not self._api_key:
            return self._fallback_response(query, results, detected_type)

        prompt_template = QUERY_TYPE_PROMPTS.get(detected_type, QUERY_TYPE_PROMPTS["general"])
        context_str = self._format_context(results)
        prompt = prompt_template.format(context=context_str, query=query)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )

            answer = response.choices[0].message.content

            confidence = min(1.0, len(results) / 8 * 0.8 + 0.2)

            patterns = []
            for line in answer.split("\n"):
                line = line.strip()
                if line.startswith("- **") or line.startswith("**Pattern"):
                    patterns.append(line.lstrip("- "))

            return ReasoningResponse(
                answer=answer,
                cited_memories=[r.chunk_id for r in results[:5]],
                confidence=confidence,
                patterns=patterns[:10],
                query_type=detected_type,
            )

        except Exception as e:
            logger.error("Reasoning failed: %s", e)
            return self._fallback_response(query, results, detected_type)

    def _fallback_response(
        self,
        query: str,
        results: list[QueryResult],
        query_type: str,
    ) -> ReasoningResponse:
        """Return structured context when LLM is unavailable."""
        lines = [f"Found {len(results)} relevant memories for: **{query}**\n"]
        for i, r in enumerate(results[:5], 1):
            ts = r.timestamp.strftime("%Y-%m-%d")
            lines.append(f"{i}. [{r.source.value} · {ts}] {r.content[:200]}…")

        return ReasoningResponse(
            answer="\n".join(lines),
            cited_memories=[r.chunk_id for r in results[:5]],
            confidence=min(1.0, len(results) / 8 * 0.6),
            patterns=[],
            query_type=query_type,
        )

    def _detect_query_type(self, query: str) -> str:
        """Classify query into a reasoning type based on keyword hints."""
        query_lower = query.lower()
        scores = {}

        for qtype, keywords in QUERY_TYPE_HINTS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[qtype] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    @staticmethod
    def _format_context(results: list[QueryResult]) -> str:
        """Format retrieved results into a readable context block."""
        lines = []
        for i, r in enumerate(results, 1):
            timestamp = r.timestamp.strftime("%Y-%m-%d %H:%M")
            entities = ", ".join(r.entities) if r.entities else "none"
            lines.append(
                f"### Memory {i} [{r.source.value}: {timestamp}]\n"
                f"Entities: {entities}\n"
                f"Relevance: {r.score:.2f}\n\n"
                f"{r.content}\n"
            )
        return "\n---\n".join(lines)
