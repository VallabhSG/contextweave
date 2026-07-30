"""Core data schemas for ContextWeave."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from contextweave.timeutils import utcnow


class SourceType(str, Enum):
    CONVERSATION = "conversation"
    NOTE = "note"
    BROWSER = "browser"
    CALENDAR = "calendar"
    JOURNAL = "journal"
    UNKNOWN = "unknown"


class ContextEvent(BaseModel):
    """A normalized unit of ingested context from any source."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: SourceType
    content: str
    timestamp: datetime
    metadata: dict = Field(default_factory=dict)
    raw_format: str = "text"

    model_config = {"frozen": True}


class Chunk(BaseModel):
    """A semantically coherent piece of a ContextEvent."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    content: str
    start_idx: int
    end_idx: int
    timestamp: datetime
    source: SourceType
    entities: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    metadata: dict = Field(default_factory=dict)


class Memory(BaseModel):
    """A scored, retrievable memory built from one or more chunks."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_ids: list[str]
    summary: str
    entities: list[str] = Field(default_factory=list)
    source: SourceType
    timestamp: datetime
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)


class Entity(BaseModel):
    """A named entity extracted from context."""

    name: str
    entity_type: str  # person, place, project, topic, organization
    first_seen: datetime
    last_seen: datetime
    mention_count: int = 1
    connected_entities: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """A single result from hybrid retrieval.

    ``score`` is the final ranking score (fused relevance × temporal decay ×
    boosts) and determines ordering. ``relevance`` is the pre-decay fused
    relevance — the same memory scores the same regardless of how recency or
    query intent affect ranking — so confidence can be judged on how well the
    context matches the query, not on how the ranking was tuned.
    """

    chunk_id: str
    content: str
    score: float
    relevance: float = 0.0
    source: SourceType
    timestamp: datetime
    entities: list[str] = Field(default_factory=list)
    memory_id: Optional[str] = None


class ReasoningResponse(BaseModel):
    """Response from the LLM reasoning engine."""

    answer: str
    cited_memories: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    patterns: list[str] = Field(default_factory=list)
    query_type: str = "general"
    suggested_queries: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
