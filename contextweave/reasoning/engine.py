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
    """Synthesizes answers from retrieved context using Gemini."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or settings.gemini_api_key
        self._model_name = settings.reasoning_model
        self._model = None

    def _get_client(self):
        if self._model is None:
            from google import genai
            self._model = genai.Client(api_key=self._api_key)
        return self._model

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

        # Auto-detect query type if not specified
        detected_type = query_type or self._detect_query_type(query)
        prompt_template = QUERY_TYPE_PROMPTS.get(detected_type, QUERY_TYPE_PROMPTS["general"])

        # Format context
        context_str = self._format_context(results)

        prompt = prompt_template.format(context=context_str, query=query)

        from google.genai import types
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )

            answer = response.text

            # Unused: re.findall(r"\[([^\]]+)\]", answer) — citation parsing reserved

            # Estimate confidence from context coverage
            confidence = min(1.0, len(results) / 8 * 0.8 + 0.2)

            # Extract pattern bullets if present
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
            return ReasoningResponse(
                answer=f"Reasoning error: {e}. Retrieved {len(results)} relevant memories.",
                confidence=0.0,
                query_type=detected_type,
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
