"""Guard: the Postgres backend must expose the same interface as SQLite.

The retriever and reasoning layer are backend-agnostic — they call the same
methods whether storage is SQLite or Postgres. A method added to one backend but
not the other is invisible to unit tests (which use SQLite) and blows up only in
production (which uses Postgres). This test makes that drift fail in CI instead.

It is a pure-introspection test: no database, no Docker, runs everywhere.
"""

from __future__ import annotations

import inspect

import pytest

from contextweave.storage.knowledge_graph import KnowledgeGraph
from contextweave.storage.memory_store import MemoryStore
from contextweave.storage.postgres import (
    PgKnowledgeGraph,
    PgMemoryStore,
    PgVectorStore,
)
from contextweave.storage.vector_store import VectorStore


def _public_methods(cls) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


# The interface the retriever + reasoning layer actually depend on, per store.
# (Kept explicit rather than "full parity" so backend-specific helpers don't
# create false positives — but every method here MUST exist on both backends.)
REQUIRED = {
    "knowledge_graph": {
        "add_entities",
        "get_neighbors",
        "get_neighbors_with_distance",
        "get_connected_chunks",
        "get_connected_chunks_ranked",
        "connection_count",
        "get_entity",
        "list_entities",
        "stats",
    },
    "memory_store": {
        "search_fts",
        "get_chunk",
        "save_chunk",
        "access_counts_by_chunk",
        "record_chunk_access",
        "stats",
    },
    "vector_store": {"query", "count", "add_chunks"},
}


@pytest.mark.parametrize(
    "sqlite_cls, pg_cls, key",
    [
        (KnowledgeGraph, PgKnowledgeGraph, "knowledge_graph"),
        (MemoryStore, PgMemoryStore, "memory_store"),
        (VectorStore, PgVectorStore, "vector_store"),
    ],
)
def test_backends_share_required_interface(sqlite_cls, pg_cls, key):
    required = REQUIRED[key]
    sqlite_methods = _public_methods(sqlite_cls)
    pg_methods = _public_methods(pg_cls)

    missing_sqlite = required - sqlite_methods
    assert not missing_sqlite, (
        f"{sqlite_cls.__name__} lost required methods: {sorted(missing_sqlite)}"
    )

    missing_pg = required - pg_methods
    assert not missing_pg, (
        f"{pg_cls.__name__} is missing methods the retriever needs "
        f"(present on {sqlite_cls.__name__}): {sorted(missing_pg)}"
    )
