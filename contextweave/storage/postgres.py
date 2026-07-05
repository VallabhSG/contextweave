"""Postgres + pgvector backends — one database, multi-tenant by user_id.

Activated when CW_DATABASE_URL is set (e.g. a Supabase connection string).
Replaces SQLite (relational + FTS5), ChromaDB (vectors), and the SQLite
graph tables in one engine: tsvector full-text search, a vector(dim)
column on chunks, and plain entity/edge tables. NetworkX stays as the
in-RAM traversal cache per workspace, loaded from Postgres.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
import uuid
from datetime import datetime

import networkx as nx

from contextweave.config import settings
from contextweave.schemas import Chunk, ContextEvent, Entity, Memory, SourceType
from contextweave.timeutils import utcnow

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()
_schema_ready = False


def _configure(conn):
    # Supabase's transaction pooler (port 6543) rejects server-side prepared
    # statements; disabling them keeps both pooler modes working.
    conn.prepare_threshold = None


def get_pool():
    """Lazily created process-wide connection pool."""
    global _pool
    with _pool_lock:
        if _pool is None:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            _pool = ConnectionPool(
                settings.database_url,
                min_size=1,
                max_size=5,
                kwargs={"row_factory": dict_row},
                configure=_configure,
                open=True,
            )
            _ensure_schema(_pool)
        return _pool


def reset_pool() -> None:
    """Close the pool (tests / config changes)."""
    global _pool, _schema_ready
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
        _schema_ready = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS cw_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB DEFAULT '{}',
    raw_format TEXT DEFAULT 'text'
);
CREATE INDEX IF NOT EXISTS idx_cw_events_user ON cw_events(user_id);

CREATE TABLE IF NOT EXISTS cw_chunks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    content TEXT NOT NULL,
    start_idx INTEGER,
    end_idx INTEGER,
    timestamp TIMESTAMP NOT NULL,
    source TEXT NOT NULL,
    entities JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    embedding vector({dim}),
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS idx_cw_chunks_user ON cw_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_cw_chunks_fts ON cw_chunks USING GIN(fts);

CREATE TABLE IF NOT EXISTS cw_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    chunk_ids JSONB NOT NULL,
    summary TEXT NOT NULL,
    entities JSONB DEFAULT '[]',
    source TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cw_memories_user ON cw_memories(user_id);

CREATE TABLE IF NOT EXISTS cw_digests (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cw_digests_user ON cw_digests(user_id);

CREATE TABLE IF NOT EXISTS cw_entities (
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    mention_count INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, name)
);

CREATE TABLE IF NOT EXISTS cw_entity_edges (
    user_id TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    weight INTEGER DEFAULT 1,
    co_occurrence_chunks JSONB DEFAULT '[]',
    PRIMARY KEY (user_id, source, target)
);

CREATE TABLE IF NOT EXISTS cw_entity_chunks (
    user_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (user_id, entity_name, chunk_id)
);

CREATE TABLE IF NOT EXISTS cw_users (
    id TEXT PRIMARY KEY,
    api_key_hash TEXT NOT NULL UNIQUE,
    name TEXT DEFAULT '',
    created_at TIMESTAMP NOT NULL
);
"""


def _ensure_schema(pool) -> None:
    global _schema_ready
    if _schema_ready:
        return
    with pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # .replace, not .format — the DDL contains literal {} in JSONB defaults
        conn.execute(SCHEMA.replace("{dim}", str(settings.embedding_dimension)))
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cw_chunks_emb "
                "ON cw_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        except Exception as e:  # older pgvector without HNSW — seq scan still works
            logger.warning("HNSW index unavailable, using exact scan: %s", e)
    _schema_ready = True


def _vec(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.8g}" for x in embedding) + "]"


# ── Memory store ─────────────────────────────────────────────


class PgMemoryStore:
    """Postgres implementation of the MemoryStore interface."""

    def __init__(self, user_id: str):
        self._user_id = user_id
        self._pool = get_pool()

    # ── Events ──────────────────────────────────────────────

    def save_event(self, event: ContextEvent) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO cw_events (id, user_id, source, content, timestamp, metadata, raw_format) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata",
                (
                    event.id,
                    self._user_id,
                    event.source.value,
                    event.content,
                    event.timestamp,
                    json.dumps(event.metadata),
                    event.raw_format,
                ),
            )

    def save_events(self, events: list[ContextEvent]) -> int:
        for event in events:
            self.save_event(event)
        return len(events)

    # ── Chunks ──────────────────────────────────────────────

    def save_chunk(self, chunk: Chunk) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO cw_chunks (id, user_id, event_id, content, start_idx, end_idx, timestamp, source, entities, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, entities = EXCLUDED.entities, metadata = EXCLUDED.metadata",
                (
                    chunk.id,
                    self._user_id,
                    chunk.event_id,
                    chunk.content,
                    chunk.start_idx,
                    chunk.end_idx,
                    chunk.timestamp,
                    chunk.source.value,
                    json.dumps(chunk.entities),
                    json.dumps(chunk.metadata),
                ),
            )

    def save_chunks(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self.save_chunk(chunk)
        return len(chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM cw_chunks WHERE id = %s AND user_id = %s",
                (chunk_id, self._user_id),
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search via websearch_to_tsquery.

        Ranks are normalized per result set and emitted on the bm25-like
        negative scale the retriever expects (top hit ≈ -10 → score 1.0).
        """
        if not query.strip():
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, content, source, timestamp, entities, "
                "ts_rank_cd(fts, websearch_to_tsquery('english', %s)) AS rank "
                "FROM cw_chunks "
                "WHERE user_id = %s AND fts @@ websearch_to_tsquery('english', %s) "
                "ORDER BY rank DESC LIMIT %s",
                (query, self._user_id, query, limit),
            ).fetchall()
        if not rows:
            return []
        max_rank = max(float(r["rank"]) for r in rows) or 1.0
        return [
            {
                "chunk_id": r["id"],
                "content": r["content"],
                "fts_rank": -10.0 * (float(r["rank"]) / max_rank),
                "source": SourceType(r["source"]),
                "timestamp": r["timestamp"],
                "entities": r["entities"] or [],
            }
            for r in rows
        ]

    # ── Memories ────────────────────────────────────────────

    def save_memory(self, memory: Memory) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO cw_memories (id, user_id, chunk_ids, summary, entities, source, timestamp, importance, access_count, last_accessed, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET summary = EXCLUDED.summary, importance = EXCLUDED.importance",
                (
                    memory.id,
                    self._user_id,
                    json.dumps(memory.chunk_ids),
                    memory.summary,
                    json.dumps(memory.entities),
                    memory.source.value,
                    memory.timestamp,
                    memory.importance,
                    memory.access_count,
                    memory.last_accessed,
                    memory.created_at,
                ),
            )

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM cw_memories WHERE id = %s AND user_id = %s",
                (memory_id, self._user_id),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(
        self,
        source: str | None = None,
        min_importance: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        query = "SELECT * FROM cw_memories WHERE user_id = %s AND importance >= %s"
        params: list = [self._user_id, min_importance]
        if source:
            query += " AND source = %s"
            params.append(source)
        query += " ORDER BY importance DESC, timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        with self._pool.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def record_access(self, memory_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE cw_memories SET access_count = access_count + 1, last_accessed = %s "
                "WHERE id = %s AND user_id = %s",
                (utcnow(), memory_id, self._user_id),
            )

    def record_chunk_access(self, chunk_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE cw_memories SET access_count = access_count + 1, last_accessed = %s "
                "WHERE user_id = %s AND chunk_ids @> %s::jsonb",
                (utcnow(), self._user_id, json.dumps([chunk_id])),
            )

    def list_most_accessed(self, limit: int = 20) -> list[Memory]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM cw_memories WHERE user_id = %s AND access_count > 0 "
                "ORDER BY access_count DESC, last_accessed DESC LIMIT %s",
                (self._user_id, limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def list_recent(self, limit: int = 20) -> list[Memory]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM cw_memories WHERE user_id = %s ORDER BY timestamp DESC LIMIT %s",
                (self._user_id, limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def access_counts_by_chunk(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT chunk_ids, access_count FROM cw_memories "
                "WHERE user_id = %s AND access_count > 0",
                (self._user_id,),
            ).fetchall()
        for row in rows:
            for chunk_id in row["chunk_ids"]:
                counts[chunk_id] = counts.get(chunk_id, 0) + row["access_count"]
        return counts

    # ── Digests ─────────────────────────────────────────────

    def save_digest(self, payload_json: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO cw_digests (user_id, generated_at, payload) VALUES (%s, %s, %s)",
                (self._user_id, utcnow(), payload_json),
            )

    def latest_digest(self) -> tuple[datetime, str] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT generated_at, payload FROM cw_digests "
                "WHERE user_id = %s ORDER BY id DESC LIMIT 1",
                (self._user_id,),
            ).fetchone()
        return (row["generated_at"], row["payload"]) if row else None

    # ── Data control ────────────────────────────────────────

    def wipe(self) -> None:
        with self._pool.connection() as conn:
            for table in ("cw_events", "cw_chunks", "cw_memories", "cw_digests"):
                conn.execute(
                    f"DELETE FROM {table} WHERE user_id = %s",  # noqa: S608 — fixed table list
                    (self._user_id,),
                )

    def export_data(self) -> dict:
        with self._pool.connection() as conn:
            events = conn.execute(
                "SELECT id, source, content, timestamp, metadata, raw_format "
                "FROM cw_events WHERE user_id = %s",
                (self._user_id,),
            ).fetchall()
            chunks = conn.execute(
                "SELECT id, event_id, content, timestamp, source, entities, metadata "
                "FROM cw_chunks WHERE user_id = %s",
                (self._user_id,),
            ).fetchall()
            memories = conn.execute(
                "SELECT id, chunk_ids, summary, entities, source, timestamp, importance, "
                "access_count, last_accessed, created_at FROM cw_memories WHERE user_id = %s",
                (self._user_id,),
            ).fetchall()
        return {
            "events": [dict(r) for r in events],
            "chunks": [dict(r) for r in chunks],
            "memories": [dict(r) for r in memories],
        }

    def stats(self) -> dict:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM cw_events WHERE user_id = %s) AS events, "
                "(SELECT COUNT(*) FROM cw_chunks WHERE user_id = %s) AS chunks, "
                "(SELECT COUNT(*) FROM cw_memories WHERE user_id = %s) AS memories",
                (self._user_id, self._user_id, self._user_id),
            ).fetchone()
        return {"events": row["events"], "chunks": row["chunks"], "memories": row["memories"]}

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _row_to_chunk(row: dict) -> Chunk:
        return Chunk(
            id=row["id"],
            event_id=row["event_id"],
            content=row["content"],
            start_idx=row["start_idx"],
            end_idx=row["end_idx"],
            timestamp=row["timestamp"],
            source=SourceType(row["source"]),
            entities=row["entities"] or [],
            metadata=row["metadata"] or {},
        )

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        return Memory(
            id=row["id"],
            chunk_ids=row["chunk_ids"],
            summary=row["summary"],
            entities=row["entities"] or [],
            source=SourceType(row["source"]),
            timestamp=row["timestamp"],
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
        )


# ── Vector store ─────────────────────────────────────────────


class PgVectorStore:
    """pgvector implementation of the VectorStore interface.

    Embeddings live on cw_chunks (written by ingestion after save_chunk),
    so 'adding' a vector is an UPDATE against the already-saved row.
    """

    def __init__(self, user_id: str):
        self._user_id = user_id
        self._pool = get_pool()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        added = 0
        with self._pool.connection() as conn:
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning("Skipping chunk %s — no embedding", chunk.id)
                    continue
                try:
                    cur = conn.execute(
                        "UPDATE cw_chunks SET embedding = %s::vector "
                        "WHERE id = %s AND user_id = %s",
                        (_vec(chunk.embedding), chunk.id, self._user_id),
                    )
                    added += cur.rowcount
                except Exception as e:
                    logger.error("Failed to store vector for chunk %s: %s", chunk.id, e)
        return added

    def query(
        self,
        embedding: list[float],
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        k = top_k or settings.retrieval_top_k
        if where:
            logger.debug("PgVectorStore.query ignores 'where' filters: %s", where)
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, event_id, content, source, timestamp, entities, "
                "(embedding <=> %s::vector) AS distance "
                "FROM cw_chunks WHERE user_id = %s AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (_vec(embedding), self._user_id, _vec(embedding), k),
            ).fetchall()
        return [
            {
                "chunk_id": r["id"],
                "content": r["content"],
                "metadata": {
                    "event_id": r["event_id"],
                    "source": r["source"],
                    "timestamp": r["timestamp"].isoformat(),
                    "entities": ",".join(r["entities"] or []),
                },
                "distance": float(r["distance"]),
                "score": 1.0 - float(r["distance"]),
            }
            for r in rows
        ]

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE cw_chunks SET embedding = NULL WHERE user_id = %s AND id = ANY(%s)",
                (self._user_id, chunk_ids),
            )

    def reset(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE cw_chunks SET embedding = NULL WHERE user_id = %s", (self._user_id,)
            )

    def count(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM cw_chunks WHERE user_id = %s AND embedding IS NOT NULL",
                (self._user_id,),
            ).fetchone()
        return row["c"]


# ── Knowledge graph ──────────────────────────────────────────


class PgKnowledgeGraph:
    """Postgres-backed entity graph with the NetworkX in-RAM traversal cache."""

    def __init__(self, user_id: str):
        self._user_id = user_id
        self._pool = get_pool()
        self._graph = nx.Graph()
        self._lock = threading.RLock()
        self._load_graph()

    def _load_graph(self):
        with self._pool.connection() as conn:
            for e in conn.execute(
                "SELECT name, entity_type, mention_count FROM cw_entities WHERE user_id = %s",
                (self._user_id,),
            ).fetchall():
                self._graph.add_node(
                    e["name"], entity_type=e["entity_type"], mention_count=e["mention_count"]
                )
            for edge in conn.execute(
                "SELECT source, target, weight FROM cw_entity_edges WHERE user_id = %s",
                (self._user_id,),
            ).fetchall():
                self._graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    def add_entities(self, entities: list[Entity], chunk_id: str) -> None:
        with self._lock, self._pool.connection() as conn:
            entity_names = []
            for entity in entities:
                conn.execute(
                    "INSERT INTO cw_entities (user_id, name, entity_type, first_seen, last_seen, mention_count) "
                    "VALUES (%s, %s, %s, %s, %s, 1) "
                    "ON CONFLICT (user_id, name) DO UPDATE SET "
                    "last_seen = EXCLUDED.last_seen, mention_count = cw_entities.mention_count + 1",
                    (
                        self._user_id,
                        entity.name,
                        entity.entity_type,
                        entity.first_seen,
                        entity.last_seen,
                    ),
                )
                conn.execute(
                    "INSERT INTO cw_entity_chunks (user_id, entity_name, chunk_id) "
                    "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (self._user_id, entity.name, chunk_id),
                )
                self._graph.add_node(
                    entity.name,
                    entity_type=entity.entity_type,
                    mention_count=self._graph.nodes.get(entity.name, {}).get("mention_count", 0)
                    + 1,
                )
                entity_names.append(entity.name)

            for i, src in enumerate(entity_names):
                for tgt in entity_names[i + 1 :]:
                    a, b = min(src, tgt), max(src, tgt)
                    existing = conn.execute(
                        "SELECT co_occurrence_chunks FROM cw_entity_edges "
                        "WHERE user_id = %s AND source = %s AND target = %s",
                        (self._user_id, a, b),
                    ).fetchone()
                    if existing:
                        chunks = (existing["co_occurrence_chunks"] or []) + [chunk_id]
                        conn.execute(
                            "UPDATE cw_entity_edges SET weight = weight + 1, co_occurrence_chunks = %s "
                            "WHERE user_id = %s AND source = %s AND target = %s",
                            (json.dumps(chunks[-100:]), self._user_id, a, b),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO cw_entity_edges (user_id, source, target, weight, co_occurrence_chunks) "
                            "VALUES (%s, %s, %s, 1, %s)",
                            (self._user_id, a, b, json.dumps([chunk_id])),
                        )
                    self._graph.add_edge(
                        src, tgt, weight=self._graph.edges.get((src, tgt), {}).get("weight", 0) + 1
                    )

    def get_neighbors(self, entity_name: str, hops: int = 1) -> list[str]:
        with self._lock:
            if entity_name not in self._graph:
                return []
            visited: set[str] = set()
            frontier = {entity_name}
            for _ in range(hops):
                next_frontier = set()
                for node in frontier:
                    for neighbor in self._graph.neighbors(node):
                        if neighbor not in visited and neighbor != entity_name:
                            next_frontier.add(neighbor)
                            visited.add(neighbor)
                frontier = next_frontier
        return sorted(visited)

    def get_connected_chunks(self, entity_name: str, hops: int = 1) -> list[str]:
        neighbors = self.get_neighbors(entity_name, hops)
        neighbors.append(entity_name)
        chunk_ids: set[str] = set()
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM cw_entity_chunks "
                "WHERE user_id = %s AND entity_name = ANY(%s)",
                (self._user_id, neighbors),
            ).fetchall()
            chunk_ids.update(r["chunk_id"] for r in rows)
            edges = conn.execute(
                "SELECT co_occurrence_chunks FROM cw_entity_edges "
                "WHERE user_id = %s AND (source = ANY(%s) OR target = ANY(%s))",
                (self._user_id, neighbors, neighbors),
            ).fetchall()
            for edge in edges:
                chunk_ids.update(edge["co_occurrence_chunks"] or [])
        return sorted(chunk_ids)

    def get_entity(self, name: str) -> Entity | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM cw_entities WHERE user_id = %s AND name = %s",
                (self._user_id, name),
            ).fetchone()
        if not row:
            return None
        return Entity(
            name=row["name"],
            entity_type=row["entity_type"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            mention_count=row["mention_count"],
            connected_entities=self.get_neighbors(name, hops=1),
        )

    def list_entities(self, limit: int = 100) -> list[Entity]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM cw_entities WHERE user_id = %s ORDER BY mention_count DESC LIMIT %s",
                (self._user_id, limit),
            ).fetchall()
        return [
            Entity(
                name=r["name"],
                entity_type=r["entity_type"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
                mention_count=r["mention_count"],
                connected_entities=self.get_neighbors(r["name"], hops=1),
            )
            for r in rows
        ]

    def connection_count(self, entity_name: str) -> int:
        with self._lock:
            if entity_name not in self._graph:
                return 0
            return self._graph.degree(entity_name)

    def wipe(self) -> None:
        with self._lock, self._pool.connection() as conn:
            for table in ("cw_entities", "cw_entity_edges", "cw_entity_chunks"):
                conn.execute(
                    f"DELETE FROM {table} WHERE user_id = %s",  # noqa: S608 — fixed table list
                    (self._user_id,),
                )
            self._graph.clear()

    def export_data(self) -> dict:
        with self._pool.connection() as conn:
            entities = conn.execute(
                "SELECT name, entity_type, first_seen, last_seen, mention_count "
                "FROM cw_entities WHERE user_id = %s",
                (self._user_id,),
            ).fetchall()
            edges = conn.execute(
                "SELECT source, target, weight FROM cw_entity_edges WHERE user_id = %s",
                (self._user_id,),
            ).fetchall()
        return {"entities": [dict(r) for r in entities], "edges": [dict(r) for r in edges]}

    def stats(self) -> dict:
        with self._lock:
            return {
                "entities": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
            }


# ── User store ───────────────────────────────────────────────


class PgUserStore:
    """Postgres registry mapping hashed API keys to user ids."""

    def __init__(self):
        self._pool = get_pool()

    def create_user(self, name: str = "") -> tuple[str, str]:
        user_id = uuid.uuid4().hex[:12]
        api_key = f"cw_{secrets.token_hex(20)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO cw_users (id, api_key_hash, name, created_at) VALUES (%s, %s, %s, %s)",
                (user_id, key_hash, name[:80], utcnow()),
            )
        return user_id, api_key

    def verify(self, api_key: str) -> str | None:
        if not api_key:
            return None
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id FROM cw_users WHERE api_key_hash = %s", (key_hash,)
            ).fetchone()
        return row["id"] if row else None
