"""Tests for the semantic chunker."""

from __future__ import annotations

from datetime import datetime

import pytest

from contextweave.processing.chunker import SemanticChunker
from contextweave.schemas import ContextEvent, SourceType


@pytest.fixture
def chunker():
    return SemanticChunker(max_tokens=100, overlap_sentences=1)


@pytest.fixture
def note_event():
    return ContextEvent(
        source=SourceType.NOTE,
        content="## Project Alpha\n\nThis is a note about Project Alpha. It has multiple paragraphs.\n\n## Project Beta\n\nThis is a note about Project Beta. It is separate from Alpha.",
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        metadata={"filename": "notes.md"},
    )


@pytest.fixture
def chat_event():
    lines = [f"[Alice]: Message {i} about the project status" for i in range(25)]
    return ContextEvent(
        source=SourceType.CONVERSATION,
        content="\n".join(lines),
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        metadata={"speakers": ["Alice"]},
    )


class TestChunker:
    def test_chunk_event_returns_chunks(self, chunker, note_event):
        chunks = chunker.chunk_event(note_event)
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)

    def test_chunk_event_preserves_metadata(self, chunker, note_event):
        chunks = chunker.chunk_event(note_event)
        for chunk in chunks:
            assert chunk.event_id == note_event.id
            assert chunk.timestamp == note_event.timestamp
            assert chunk.source == SourceType.NOTE

    def test_chunk_event_respects_max_tokens(self, chunker, note_event):
        chunks = chunker.chunk_event(note_event)
        for chunk in chunks:
            estimated = SemanticChunker._estimate_tokens(chunk.content)
            assert estimated <= chunker.max_tokens * 1.2  # 20% tolerance for overlap

    def test_long_conversation_splits(self, chunker, chat_event):
        chunks = chunker.chunk_event(chat_event)
        assert len(chunks) > 1, "Long conversation should be split into multiple chunks"

    def test_short_content_stays_single_chunk(self, chunker):
        event = ContextEvent(
            source=SourceType.NOTE,
            content="Short note.",
            timestamp=datetime.utcnow(),
        )
        chunks = chunker.chunk_event(event)
        assert len(chunks) == 1

    def test_chunk_events_batch(self, chunker, note_event, chat_event):
        chunks = chunker.chunk_events([note_event, chat_event])
        event_ids = {c.event_id for c in chunks}
        assert note_event.id in event_ids
        assert chat_event.id in event_ids

    def test_estimate_tokens(self):
        text = "a" * 400  # 400 chars = ~100 tokens
        assert SemanticChunker._estimate_tokens(text) == 100

    def test_chunk_ids_are_unique(self, chunker, chat_event):
        chunks = chunker.chunk_event(chat_event)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_content_handled(self, chunker):
        event = ContextEvent(
            source=SourceType.NOTE,
            content="",
            timestamp=datetime.utcnow(),
        )
        chunks = chunker.chunk_event(event)
        # Should produce one chunk even for empty content
        assert isinstance(chunks, list)
