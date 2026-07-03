"""Proactive digest engine — turns recent memory into a daily nudge."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from contextweave.config import settings
from contextweave.schemas import Memory
from contextweave.timeutils import utcnow

logger = logging.getLogger(__name__)

COMMITMENT_SIGNALS = re.compile(
    r"action item|follow[- ]?up|deadline|due\b|i('|\s+wi)ll\b|promised|agreed"
    r"|by (mon|tues|wednes|thurs|fri|satur|sun)day|remind",
    re.IGNORECASE,
)

STALE_DAYS = 7

DIGEST_PROMPT = """You are a proactive personal chief-of-staff. From the recent memories below, \
produce a short daily nudge.

Return ONLY valid JSON (no markdown fences) with exactly these keys:
- "headline": one warm, specific sentence nudging the user toward the most important thing right now
- "focus": array of up to 3 short strings — current priorities inferred from the memories
- "commitments": array of up to 4 short strings — concrete commitments or action items the user made
- "gaps": array of up to 3 short strings — things the user seems to be avoiding or letting slip

Recent memories (newest first):
{context}
"""


class Digest(BaseModel):
    """A proactive nudge synthesized from recent memories."""

    generated_at: datetime
    headline: str = ""
    focus: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    top_entities: list[str] = Field(default_factory=list)
    memory_count: int = 0
    llm_generated: bool = False


class DigestEngine:
    """Generates digests via Groq when a key is set, else deterministically."""

    def generate(self, memories: list[Memory]) -> Digest:
        if not memories:
            return Digest(
                generated_at=utcnow(),
                headline=(
                    "Your memory is empty — capture a thought, a meeting, or a "
                    "conversation to get your first nudge."
                ),
            )

        top_entities = self._top_entities(memories)
        api_key = settings.groq_api_key
        if api_key:
            digest = self._llm_generate(memories, top_entities, api_key)
            if digest is not None:
                return digest
        return self._fallback_generate(memories, top_entities)

    # ── LLM path ────────────────────────────────────────────

    def _llm_generate(
        self, memories: list[Memory], top_entities: list[str], api_key: str
    ) -> Digest | None:
        context = "\n".join(
            f"- [{m.source.value} · {m.timestamp.strftime('%Y-%m-%d')}] {m.summary}"
            for m in memories
        )
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=settings.reasoning_model,
                messages=[{"role": "user", "content": DIGEST_PROMPT.format(context=context)}],
                temperature=0.4,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            return Digest(
                generated_at=utcnow(),
                headline=str(data.get("headline", ""))[:300],
                focus=[str(x) for x in data.get("focus", [])][:3],
                commitments=[str(x) for x in data.get("commitments", [])][:4],
                gaps=[str(x) for x in data.get("gaps", [])][:3],
                top_entities=top_entities,
                memory_count=len(memories),
                llm_generated=True,
            )
        except Exception as e:
            logger.warning("LLM digest failed, using fallback: %s", e)
            return None

    # ── Deterministic fallback ──────────────────────────────

    def _fallback_generate(self, memories: list[Memory], top_entities: list[str]) -> Digest:
        commitments = [
            m.summary.strip()[:140] for m in memories if COMMITMENT_SIGNALS.search(m.summary)
        ][:4]
        gaps = self._stale_threads(memories)

        if commitments:
            headline = (
                f"You have {len(commitments)} open commitment"
                f"{'s' if len(commitments) != 1 else ''} — start with: {commitments[0]}"
            )
        elif top_entities:
            headline = (
                f"Your recent thread centers on {top_entities[0]} — "
                "worth a deliberate next step today."
            )
        else:
            headline = "You've been capturing memories — ask your memory what to focus on next."

        return Digest(
            generated_at=utcnow(),
            headline=headline,
            focus=top_entities[:3],
            commitments=commitments,
            gaps=gaps,
            top_entities=top_entities,
            memory_count=len(memories),
            llm_generated=False,
        )

    @staticmethod
    def _top_entities(memories: list[Memory], limit: int = 5) -> list[str]:
        counts: dict[str, int] = {}
        for m in memories:
            for e in m.entities:
                if e:
                    counts[e] = counts.get(e, 0) + 1
        return sorted(counts, key=lambda k: counts[k], reverse=True)[:limit]

    @staticmethod
    def _stale_threads(memories: list[Memory]) -> list[str]:
        """Entities mentioned more than once whose last mention has gone quiet."""
        newest = max(m.timestamp for m in memories)
        last_seen: dict[str, datetime] = {}
        mentions: dict[str, int] = {}
        for m in memories:
            for e in m.entities:
                if not e:
                    continue
                mentions[e] = mentions.get(e, 0) + 1
                if e not in last_seen or m.timestamp > last_seen[e]:
                    last_seen[e] = m.timestamp

        stale = [
            e
            for e, seen in last_seen.items()
            if mentions[e] >= 2 and (newest - seen) > timedelta(days=STALE_DAYS)
        ]
        stale.sort(key=lambda e: last_seen[e])
        return [f"No recent activity around {e}" for e in stale[:3]]
