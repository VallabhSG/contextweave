"""Knowledge graph built on NetworkX + SQLite for entity relationships."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import networkx as nx

from contextweave.config import settings
from contextweave.schemas import Entity

logger = logging.getLogger(__name__)

GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER DEFAULT 1,
    connected_entities TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS entity_edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    weight INTEGER DEFAULT 1,
    co_occurrence_chunks TEXT DEFAULT '[]',
    PRIMARY KEY (source, target),
    FOREIGN KEY (source) REFERENCES entities(name),
    FOREIGN KEY (target) REFERENCES entities(name)
);

CREATE TABLE IF NOT EXISTS entity_chunks (
    entity_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (entity_name, chunk_id),
    FOREIGN KEY (entity_name) REFERENCES entities(name)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON entity_edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON entity_edges(target);
CREATE INDEX IF NOT EXISTS idx_entity_chunks_name ON entity_chunks(entity_name);
"""


class KnowledgeGraph:
    """Entity relationship graph backed by SQLite with NetworkX for traversal."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.sqlite_db_path
        self._graph = nx.Graph()
        self._ensure_schema()
        self._load_graph()

    def _ensure_schema(self):
        with self._conn() as conn:
            conn.executescript(GRAPH_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _load_graph(self):
        """Load the full graph from SQLite into NetworkX."""
        with self._conn() as conn:
            entities = conn.execute("SELECT * FROM entities").fetchall()
            for e in entities:
                self._graph.add_node(
                    e["name"],
                    entity_type=e["entity_type"],
                    mention_count=e["mention_count"],
                )

            edges = conn.execute("SELECT * FROM entity_edges").fetchall()
            for edge in edges:
                self._graph.add_edge(
                    edge["source"],
                    edge["target"],
                    weight=edge["weight"],
                )

    def add_entities(self, entities: list[Entity], chunk_id: str) -> None:
        """Add entities and create co-occurrence edges from a single chunk."""
        with self._conn() as conn:
            entity_names = []

            for entity in entities:
                existing = conn.execute(
                    "SELECT * FROM entities WHERE name = ?", (entity.name,)
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE entities SET last_seen = ?, mention_count = mention_count + 1 WHERE name = ?",
                        (entity.last_seen.isoformat(), entity.name),
                    )
                else:
                    conn.execute(
                        "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) VALUES (?, ?, ?, ?, ?)",
                        (
                            entity.name,
                            entity.entity_type,
                            entity.first_seen.isoformat(),
                            entity.last_seen.isoformat(),
                            1,
                        ),
                    )

                self._graph.add_node(
                    entity.name,
                    entity_type=entity.entity_type,
                    mention_count=self._graph.nodes.get(entity.name, {}).get("mention_count", 0)
                    + 1,
                )
                entity_names.append(entity.name)

                # Direct entity → chunk mapping (always works, even for solo entities)
                conn.execute(
                    "INSERT OR IGNORE INTO entity_chunks (entity_name, chunk_id) VALUES (?, ?)",
                    (entity.name, chunk_id),
                )

            # Create co-occurrence edges between all entities in this chunk
            for i, src in enumerate(entity_names):
                for tgt in entity_names[i + 1 :]:
                    existing_edge = conn.execute(
                        "SELECT * FROM entity_edges WHERE source = ? AND target = ?",
                        (min(src, tgt), max(src, tgt)),
                    ).fetchone()

                    if existing_edge:
                        chunks = json.loads(existing_edge["co_occurrence_chunks"])
                        chunks.append(chunk_id)
                        conn.execute(
                            "UPDATE entity_edges SET weight = weight + 1, co_occurrence_chunks = ? WHERE source = ? AND target = ?",
                            (json.dumps(chunks[-100:]), min(src, tgt), max(src, tgt)),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO entity_edges (source, target, weight, co_occurrence_chunks) VALUES (?, ?, ?, ?)",
                            (min(src, tgt), max(src, tgt), 1, json.dumps([chunk_id])),
                        )

                    self._graph.add_edge(
                        src,
                        tgt,
                        weight=self._graph.edges.get((src, tgt), {}).get("weight", 0) + 1,
                    )

    def get_neighbors(self, entity_name: str, hops: int = 1) -> list[str]:
        """Get entities within N hops of the given entity."""
        if entity_name not in self._graph:
            return []

        visited = set()
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
        """Get chunk IDs connected to an entity and its neighbors."""
        neighbors = self.get_neighbors(entity_name, hops)
        neighbors.append(entity_name)

        chunk_ids = set()
        with self._conn() as conn:
            for name in neighbors:
                # Direct entity→chunk mappings (primary source)
                rows = conn.execute(
                    "SELECT chunk_id FROM entity_chunks WHERE entity_name = ?",
                    (name,),
                ).fetchall()
                for row in rows:
                    chunk_ids.add(row["chunk_id"])

                # Co-occurrence edges (supplementary)
                edges = conn.execute(
                    "SELECT co_occurrence_chunks FROM entity_edges WHERE source = ? OR target = ?",
                    (name, name),
                ).fetchall()
                for edge in edges:
                    chunks = json.loads(edge["co_occurrence_chunks"])
                    chunk_ids.update(chunks)

        return sorted(chunk_ids)

    def get_entity(self, name: str) -> Entity | None:
        """Get entity details."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
            if row:
                neighbors = self.get_neighbors(name, hops=1)
                return Entity(
                    name=row["name"],
                    entity_type=row["entity_type"],
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                    mention_count=row["mention_count"],
                    connected_entities=neighbors,
                )
        return None

    def list_entities(self, limit: int = 100) -> list[Entity]:
        """List entities ordered by mention count."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                Entity(
                    name=r["name"],
                    entity_type=r["entity_type"],
                    first_seen=datetime.fromisoformat(r["first_seen"]),
                    last_seen=datetime.fromisoformat(r["last_seen"]),
                    mention_count=r["mention_count"],
                    connected_entities=self.get_neighbors(r["name"], hops=1),
                )
                for r in rows
            ]

    def connection_count(self, entity_name: str) -> int:
        """Number of direct connections for an entity."""
        if entity_name not in self._graph:
            return 0
        return self._graph.degree(entity_name)

    def stats(self) -> dict:
        return {
            "entities": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }
