"""Hybrid retrieval combining vector similarity, FTS, and graph traversal."""

from __future__ import annotations

import logging
from datetime import datetime

from contextweave.config import settings
from contextweave.processing.embedder import GeminiEmbedder
from contextweave.processing.importance_scorer import ImportanceScorer
from contextweave.schemas import QueryResult, SourceType
from contextweave.storage.knowledge_graph import KnowledgeGraph
from contextweave.storage.memory_store import MemoryStore
from contextweave.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Multi-signal retrieval: vector + FTS + graph, fused and reranked."""

    def __init__(
        self,
        vector_store: VectorStore,
        memory_store: MemoryStore,
        knowledge_graph: KnowledgeGraph,
        embedder: GeminiEmbedder,
        scorer: ImportanceScorer | None = None,
    ):
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.knowledge_graph = knowledge_graph
        self.embedder = embedder
        self.scorer = scorer or ImportanceScorer()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        extra_terms: list[str] | None = None,
    ) -> list[QueryResult]:
        """Execute hybrid retrieval and return ranked results."""
        final_k = top_k or settings.retrieval_final_k

        # 1. Vector similarity search (degrade gracefully if embedding fails or store empty)
        vector_results = []
        try:
            if self.vector_store.count() > 0:
                query_embedding = self.embedder.embed_query(query)
                vector_results = self.vector_store.query(
                    embedding=query_embedding,
                    top_k=settings.retrieval_top_k,
                )
        except Exception as e:
            logger.warning("Vector search skipped: %s", e)

        # 2. Full-text search (primary + expanded terms)
        fts_results = self.memory_store.search_fts(query, limit=settings.retrieval_top_k)
        if extra_terms:
            seen_ids = {r["chunk_id"] for r in fts_results}
            for term in extra_terms[:4]:
                for r in self.memory_store.search_fts(term, limit=10):
                    if r["chunk_id"] not in seen_ids:
                        fts_results.append(r)
                        seen_ids.add(r["chunk_id"])

        # 3. Graph expansion — extract entity names from query results
        graph_chunk_ids = set()
        entity_names = set()

        for vr in vector_results:
            entities = vr["metadata"].get("entities", "").split(",")
            entity_names.update(e.strip() for e in entities if e.strip())

        for entity in entity_names:
            connected = self.knowledge_graph.get_connected_chunks(
                entity, hops=settings.graph_hop_depth
            )
            graph_chunk_ids.update(connected)

        # 4. Merge all results into a unified scoring map
        scored: dict[str, dict] = {}

        # Vector results
        for vr in vector_results:
            chunk_id = vr["chunk_id"]
            scored[chunk_id] = {
                "chunk_id": chunk_id,
                "content": vr["content"],
                "vector_score": vr["score"],
                "fts_score": 0.0,
                "graph_score": 0.0,
                "metadata": vr["metadata"],
            }

        # FTS results
        for fr in fts_results:
            chunk_id = fr["chunk_id"]
            fts_normalized = min(1.0, abs(fr["fts_rank"]) / 10.0)
            if chunk_id in scored:
                scored[chunk_id]["fts_score"] = fts_normalized
            else:
                scored[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": fr["content"],
                    "vector_score": 0.0,
                    "fts_score": fts_normalized,
                    "graph_score": 0.0,
                    "metadata": {
                        "source": fr["source"].value
                        if hasattr(fr["source"], "value")
                        else str(fr["source"]),
                        "timestamp": fr["timestamp"].isoformat()
                        if isinstance(fr["timestamp"], datetime)
                        else str(fr["timestamp"]),
                        "entities": ",".join(fr.get("entities", [])),
                    },
                }

        # Graph boost
        for chunk_id in graph_chunk_ids:
            if chunk_id in scored:
                scored[chunk_id]["graph_score"] = 0.3

        # 5. Compute final scores
        results = []
        for item in scored.values():
            # Weighted fusion: 50% vector + 30% FTS + 20% graph
            combined = (
                0.5 * item["vector_score"] + 0.3 * item["fts_score"] + 0.2 * item["graph_score"]
            )

            # Apply temporal decay
            ts = self._parse_timestamp(item["metadata"].get("timestamp", ""))
            entities = [e for e in item["metadata"].get("entities", "").split(",") if e.strip()]
            conn_count = sum(self.knowledge_graph.connection_count(e) for e in entities)
            importance = self.scorer.score(
                base_importance=combined,
                timestamp=ts,
                connection_count=conn_count,
            )

            source_str = item["metadata"].get("source", "unknown")
            try:
                source = SourceType(source_str)
            except ValueError:
                source = SourceType.UNKNOWN

            results.append(
                QueryResult(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    score=importance,
                    source=source,
                    timestamp=ts,
                    entities=entities,
                )
            )

        # 6. Filter by source if requested
        if source_filter:
            results = [r for r in results if r.source.value == source_filter]

        # 7. Filter by date range if specified
        if date_from or date_to:
            results = [
                r for r in results
                if (date_from is None or r.timestamp >= date_from)
                and (date_to is None or r.timestamp <= date_to)
            ]

        # 8. Sort by score descending and return top K
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:final_k]

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        if not ts_str:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return datetime.utcnow()
