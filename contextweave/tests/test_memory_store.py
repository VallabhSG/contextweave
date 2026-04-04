"""Tests for the SQLite memory store."""

from __future__ import annotations

from datetime import datetime

import pytest

from contextweave.schemas import Chunk, ContextEvent, Memory, SourceType
from contextweave.storage.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return MemoryStore(db_path=db_path)


@pytest.fixture
def sample_event():
    return ContextEvent(
        source=SourceType.NOTE,
        content="This is a test note about project planning and deadlines.",
        timestamp=datetime(2024, 3, 1, 9, 0, 0),
        metadata={"filename": "planning.md"},
    )


@pytest.fixture
def sample_chunk(sample_event):
    return Chunk(
        event_id=sample_event.id,
        content="This is a test note about project planning and deadlines.",
        start_idx=0,
        end_idx=57,
        timestamp=sample_event.timestamp,
        source=SourceType.NOTE,
        entities=["project", "deadlines"],
        metadata={},
    )


@pytest.fixture
def sample_memory(sample_chunk):
    return Memory(
        chunk_ids=[sample_chunk.id],
        summary="Note about project planning",
        entities=["project", "deadlines"],
        source=SourceType.NOTE,
        timestamp=datetime(2024, 3, 1, 9, 0, 0),
        importance=0.7,
    )


class TestMemoryStore:
    def test_save_and_retrieve_event(self, store, sample_event):
        store.save_event(sample_event)
        stats = store.stats()
        assert stats["events"] == 1

    def test_save_and_retrieve_chunk(self, store, sample_event, sample_chunk):
        store.save_event(sample_event)
        store.save_chunk(sample_chunk)
        retrieved = store.get_chunk(sample_chunk.id)
        assert retrieved is not None
        assert retrieved.id == sample_chunk.id
        assert retrieved.content == sample_chunk.content

    def test_save_and_retrieve_memory(self, store, sample_memory):
        store.save_memory(sample_memory)
        retrieved = store.get_memory(sample_memory.id)
        assert retrieved is not None
        assert retrieved.id == sample_memory.id
        assert retrieved.importance == 0.7

    def test_list_memories_default(self, store, sample_memory):
        store.save_memory(sample_memory)
        memories = store.list_memories()
        assert len(memories) == 1

    def test_list_memories_filter_by_source(self, store):
        note_mem = Memory(
            chunk_ids=["c1"],
            summary="note memory",
            source=SourceType.NOTE,
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        conv_mem = Memory(
            chunk_ids=["c2"],
            summary="chat memory",
            source=SourceType.CONVERSATION,
            timestamp=datetime.utcnow(),
            importance=0.5,
        )
        store.save_memory(note_mem)
        store.save_memory(conv_mem)

        notes = store.list_memories(source="note")
        assert len(notes) == 1
        assert notes[0].source == SourceType.NOTE

    def test_list_memories_filter_by_importance(self, store):
        low = Memory(chunk_ids=["c1"], summary="low", source=SourceType.NOTE, timestamp=datetime.utcnow(), importance=0.2)
        high = Memory(chunk_ids=["c2"], summary="high", source=SourceType.NOTE, timestamp=datetime.utcnow(), importance=0.9)
        store.save_memory(low)
        store.save_memory(high)

        filtered = store.list_memories(min_importance=0.5)
        assert len(filtered) == 1
        assert filtered[0].importance == 0.9

    def test_fts_search(self, store, sample_event, sample_chunk):
        store.save_event(sample_event)
        store.save_chunk(sample_chunk)
        results = store.search_fts("planning deadlines")
        assert len(results) >= 1

    def test_record_access_increments_count(self, store, sample_memory):
        store.save_memory(sample_memory)
        store.record_access(sample_memory.id)
        updated = store.get_memory(sample_memory.id)
        assert updated.access_count == 1

    def test_stats_counts_all_types(self, store, sample_event, sample_chunk, sample_memory):
        store.save_event(sample_event)
        store.save_chunk(sample_chunk)
        store.save_memory(sample_memory)
        stats = store.stats()
        assert stats["events"] == 1
        assert stats["chunks"] == 1
        assert stats["memories"] == 1

    def test_upsert_replaces_existing(self, store, sample_memory):
        store.save_memory(sample_memory)
        updated = sample_memory.model_copy(update={"importance": 0.99})
        store.save_memory(updated)
        retrieved = store.get_memory(sample_memory.id)
        assert retrieved.importance == 0.99
