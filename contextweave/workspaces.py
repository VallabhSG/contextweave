"""Per-user workspaces: isolated stores sharing heavyweight components."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from contextweave.config import settings
from contextweave.processing.chunker import SemanticChunker
from contextweave.processing.embedder import LocalEmbedder
from contextweave.processing.entity_extractor import EntityExtractor
from contextweave.processing.importance_scorer import ImportanceScorer
from contextweave.reasoning.engine import ReasoningEngine
from contextweave.retrieval.hybrid_retriever import HybridRetriever
from contextweave.storage.knowledge_graph import KnowledgeGraph
from contextweave.storage.memory_store import MemoryStore
from contextweave.storage.vector_store import VectorStore

DEMO_USER_ID = "demo"

# KnowledgeGraph keeps its graph in RAM, so bound how many stay cached.
_MAX_CACHED_WORKSPACES = 100


@dataclass
class SharedComponents:
    """User-independent components; the embedder holds the loaded model."""

    embedder: LocalEmbedder
    chunker: SemanticChunker
    extractor: EntityExtractor
    scorer: ImportanceScorer
    reasoning: ReasoningEngine


@dataclass
class Workspace:
    """One user's isolated memory: own SQLite file and Chroma collection."""

    user_id: str
    memory_store: MemoryStore
    vector_store: VectorStore
    knowledge_graph: KnowledgeGraph
    retriever: HybridRetriever

    @property
    def is_demo(self) -> bool:
        return self.user_id == DEMO_USER_ID

    def wipe(self) -> None:
        self.memory_store.wipe()
        self.knowledge_graph.wipe()
        self.vector_store.reset()

    def export_data(self) -> dict:
        data = self.memory_store.export_data()
        data["graph"] = self.knowledge_graph.export_data()
        return data


class WorkspaceManager:
    """Builds and caches workspaces; the demo space uses the legacy paths."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workspaces: dict[str, Workspace] = {}
        self._shared: SharedComponents | None = None

    def shared(self) -> SharedComponents:
        with self._lock:
            if self._shared is None:
                self._shared = SharedComponents(
                    embedder=LocalEmbedder(),
                    chunker=SemanticChunker(),
                    extractor=EntityExtractor(),
                    scorer=ImportanceScorer(),
                    reasoning=ReasoningEngine(),
                )
            return self._shared

    def get(self, user_id: str) -> Workspace:
        shared = self.shared()
        with self._lock:
            ws = self._workspaces.get(user_id)
            if ws is not None:
                return ws

            if settings.database_url:
                # Postgres + pgvector: one external database, tenant per user_id
                from contextweave.storage.postgres import (
                    PgKnowledgeGraph,
                    PgMemoryStore,
                    PgVectorStore,
                )

                memory_store = PgMemoryStore(user_id)
                vector_store = PgVectorStore(user_id)
                knowledge_graph = PgKnowledgeGraph(user_id)
            else:
                if user_id == DEMO_USER_ID:
                    db_path = settings.sqlite_db_path
                    collection = "chunks"
                else:
                    user_dir = Path(settings.data_dir) / "users" / user_id
                    user_dir.mkdir(parents=True, exist_ok=True)
                    db_path = str(user_dir / "contextweave.db")
                    collection = f"u_{user_id}"

                memory_store = MemoryStore(db_path=db_path)
                vector_store = VectorStore(collection_name=collection)
                knowledge_graph = KnowledgeGraph(db_path=db_path)
            ws = Workspace(
                user_id=user_id,
                memory_store=memory_store,
                vector_store=vector_store,
                knowledge_graph=knowledge_graph,
                retriever=HybridRetriever(
                    vector_store=vector_store,
                    memory_store=memory_store,
                    knowledge_graph=knowledge_graph,
                    embedder=shared.embedder,
                    scorer=shared.scorer,
                ),
            )

            if len(self._workspaces) >= _MAX_CACHED_WORKSPACES:
                for key in list(self._workspaces):
                    if key != DEMO_USER_ID:
                        del self._workspaces[key]
                        break
            self._workspaces[user_id] = ws
            return ws

    def reset(self) -> None:
        """Testing hook: drop cached workspaces, shared components, and DB pool."""
        with self._lock:
            self._workspaces.clear()
            self._shared = None
        try:
            from contextweave.storage import postgres

            postgres.reset_pool()
        except ImportError:  # psycopg not installed — SQLite-only environment
            pass


manager = WorkspaceManager()
