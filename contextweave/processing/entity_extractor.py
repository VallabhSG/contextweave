"""Entity extraction using Groq LLM for structured NER."""

from __future__ import annotations

import json
import logging
import re

from contextweave.config import settings
from contextweave.schemas import Chunk, Entity

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract named entities from the following text. Return a JSON array of objects with these fields:
- "name": the entity name (normalized, e.g., "John Smith" not "john")
- "type": one of "person", "place", "project", "topic", "organization", "event"

Only extract meaningful, specific entities. Skip generic words.

Text:
{text}

Return ONLY valid JSON array, no markdown fences or explanation."""


class EntityExtractor:
    """Extracts named entities from chunks using Groq LLM."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or settings.groq_api_key
        self._model_name = settings.extraction_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def extract_from_chunk(self, chunk: Chunk) -> list[Entity]:
        """Extract entities from a single chunk."""
        if not self._api_key:
            return self._fallback_extract(chunk)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "user", "content": EXTRACTION_PROMPT.format(text=chunk.content[:2000])}
                ],
                temperature=0.1,
                max_tokens=1024,
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            entities_data = json.loads(raw)
            now = chunk.timestamp

            return [
                Entity(
                    name=e["name"],
                    entity_type=e.get("type", "topic"),
                    first_seen=now,
                    last_seen=now,
                )
                for e in entities_data
                if isinstance(e, dict) and e.get("name")
            ]

        except Exception as e:
            logger.warning("Entity extraction failed for chunk %s: %s", chunk.id, e)
            return self._fallback_extract(chunk)

    def extract_from_chunks(self, chunks: list[Chunk]) -> dict[str, list[Entity]]:
        """Extract entities from multiple chunks. Returns {chunk_id: [Entity]}."""
        results = {}
        for chunk in chunks:
            entities = self.extract_from_chunk(chunk)
            results[chunk.id] = entities
        return results

    def _fallback_extract(self, chunk: Chunk) -> list[Entity]:
        """Simple regex-based fallback when LLM extraction is unavailable."""
        entities = []
        now = chunk.timestamp

        # Extract @mentions
        mentions = re.findall(r"@(\w+)", chunk.content)
        for m in mentions:
            entities.append(Entity(name=m, entity_type="person", first_seen=now, last_seen=now))

        # Extract capitalized multi-word names (simple heuristic)
        names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", chunk.content)
        for name in set(names):
            entities.append(Entity(name=name, entity_type="person", first_seen=now, last_seen=now))

        # Extract URLs as topics
        urls = re.findall(r"https?://\S+", chunk.content)
        for url in urls[:5]:
            domain = url.split("://")[1].split("/")[0] if "://" in url else url
            entities.append(Entity(name=domain, entity_type="topic", first_seen=now, last_seen=now))

        return entities
