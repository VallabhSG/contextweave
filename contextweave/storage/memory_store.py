"""SQLite-backed store for memories, events, and full-text search."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from contextweave.config import settings
from contextweave.schemas import Chunk, ContextEvent, Memory, SourceType

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    raw_format TEXT DEFAULT 'text'
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    content TEXT NOT NULL,
    start_idx INTEGER,
    end_idx INTEGER,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    entities TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (event_id) REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    chunk_ids TEXT NOT NULL,
    summary TEXT NOT NULL,
    entities TEXT DEFAULT '[]',
    source TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id UNINDEXED,
    content,
    entities
);

CREATE INDEX IF NOT EXISTS idx_chunks_event ON chunks(event_id);
CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
"""


class MemoryStore:
    """SQLite store for events, chunks, memories, and full-text search."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.sqlite_db_path
        self._ensure_schema()

    def _ensure_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Events ──────────────────────────────────────────────

    def save_event(self, event: ContextEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO events (id, source, content, timestamp, metadata, raw_format) VALUES (?, ?, ?, ?, ?, ?)",
                (event.id, event.source.value, event.content, event.timestamp.isoformat(), json.dumps(event.metadata), event.raw_format),
            )

    def save_events(self, events: list[ContextEvent]) -> int:
        for event in events:
            self.save_event(event)
        return len(events)

    # ── Chunks ──────────────────────────────────────────────

    def save_chunk(self, chunk: Chunk) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chunks (id, event_id, content, start_idx, end_idx, timestamp, source, entities, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (chunk.id, chunk.event_id, chunk.content, chunk.start_idx, chunk.end_idx, chunk.timestamp.isoformat(), chunk.source.value, json.dumps(chunk.entities), json.dumps(chunk.metadata)),
            )
            # Update FTS index
            conn.execute("INSERT OR REPLACE INTO chunks_fts (id, content, entities) VALUES (?, ?, ?)", (chunk.id, chunk.content, " ".join(chunk.entities)))

    def save_chunks(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self.save_chunk(chunk)
        return len(chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            if row:
                return self._row_to_chunk(row)
        return None

    def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over chunks."""
        # Escape FTS5 special characters to prevent OperationalError
        safe_query = re.sub(r'["()*:]', " ", query).strip()
        if not safe_query:
            return []

        with self._conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT id, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                    (safe_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                logger.warning("FTS query failed for: %s", safe_query)
                return []

            results = []
            for row in rows:
                chunk = self.get_chunk(row["id"])
                if chunk:
                    results.append({"chunk_id": chunk.id, "content": chunk.content, "fts_rank": row["rank"], "source": chunk.source, "timestamp": chunk.timestamp, "entities": chunk.entities})
            return results

    # ── Memories ────────────────────────────────────────────

    def save_memory(self, memory: Memory) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, chunk_ids, summary, entities, source, timestamp, importance, access_count, last_accessed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (memory.id, json.dumps(memory.chunk_ids), memory.summary, json.dumps(memory.entities), memory.source.value, memory.timestamp.isoformat(), memory.importance, memory.access_count, memory.last_accessed.isoformat() if memory.last_accessed else None, memory.created_at.isoformat()),
            )

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                return self._row_to_memory(row)
        return None

    def list_memories(
        self,
        source: str | None = None,
        min_importance: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        with self._conn() as conn:
            query = "SELECT * FROM memories WHERE importance >= ?"
            params: list = [min_importance]
            if source:
                query += " AND source = ?"
                params.append(source)
            query += " ORDER BY importance DESC, timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def record_access(self, memory_id: str) -> None:
        """Increment access count and update last_accessed."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), memory_id),
            )

    def stats(self) -> dict:
        with self._conn() as conn:
            events = conn.execute("SELECT COUNT(*) as c FROM events").fetchone()["c"]
            chunks = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
            memories = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            return {"events": events, "chunks": chunks, "memories": memories}

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            event_id=row["event_id"],
            content=row["content"],
            start_idx=row["start_idx"],
            end_idx=row["end_idx"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            source=SourceType(row["source"]),
            entities=json.loads(row["entities"]),
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            chunk_ids=json.loads(row["chunk_ids"]),
            summary=row["summary"],
            entities=json.loads(row["entities"]),
            source=SourceType(row["source"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
