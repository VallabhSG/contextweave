"""Hybrid retrieval combining vector similarity, FTS, and graph traversal."""

from __future__ import annotations

import logging
from datetime import datetime

from contextweave.config import settings
from contextweave.processing.embedder import LocalEmbedder
from contextweave.processing.importance_scorer import ImportanceScorer
from contextweave.reasoning.query_intent import detect_query_type
from contextweave.schemas import QueryResult, SourceType
from contextweave.storage.knowledge_graph import KnowledgeGraph
from contextweave.storage.memory_store import MemoryStore
from contextweave.storage.vector_store import VectorStore
from contextweave.timeutils import utcnow

logger = logging.getLogger(__name__)

# SQLite FTS5 bm25() returns negative scores (more negative = stronger match).
# Normalize to [0, 1) with a smooth saturating curve rather than an arbitrary
# hard clip, so FTS relevance composes predictably with the [0, 1] vector score.
# _FTS_SATURATION_K is the match strength that maps to a relevance of 0.5.
_FTS_SATURATION_K = 5.0


def _normalize_fts_rank(rank: float) -> float:
    """Map an FTS5 bm25 rank (negative = better) to a [0, 1) relevance score."""
    strength = max(0.0, -rank)
    return strength / (strength + _FTS_SATURATION_K)


# Query-adaptive fusion weights (vector, fts, graph), each summing to 1.0 so
# scores stay comparable across intents. Only intents that are *explicitly about
# connections* lean on the graph; everything else keeps the balanced default —
# so behaviour for ordinary queries is unchanged.
_FUSION_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "general": (0.5, 0.3, 0.2),
    "cross_reference": (0.4, 0.2, 0.4),  # "how do X and Y connect?" → trust the graph
    "patterns": (0.45, 0.2, 0.35),  # recurring themes emerge from co-occurrence
}
_DEFAULT_FUSION = _FUSION_WEIGHTS["general"]


def fusion_weights(intent: str) -> tuple[float, float, float]:
    """Return (vector, fts, graph) fusion weights for a query intent."""
    return _FUSION_WEIGHTS.get(intent, _DEFAULT_FUSION)


class HybridRetriever:
    """Multi-signal retrieval: vector + FTS + graph, fused and reranked."""

    def __init__(
        self,
        vector_store: VectorStore,
        memory_store: MemoryStore,
        knowledge_graph: KnowledgeGraph,
        embedder: LocalEmbedder,
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
        query_type: str | None = None,
    ) -> list[QueryResult]:
        """Execute hybrid retrieval and return ranked results."""
        final_k = top_k or settings.retrieval_final_k

        # Intent-aware decay: a temporal query wants the history preserved, so
        # relax the recency half-life instead of burying old memories.
        intent = query_type or detect_query_type(query)
        half_life = settings.temporal_query_half_life_days if intent == "temporal" else None

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
        chunk_distance: dict[str, int] = {}
        entity_names = set()

        for vr in vector_results:
            entities = vr["metadata"].get("entities", "").split(",")
            entity_names.update(e.strip() for e in entities if e.strip())

        # Also seed graph expansion from keyword (FTS) matches. This connects the
        # dots even when a memory surfaces by keyword rather than vector, and
        # keeps the graph working when vector search is unavailable (embedding
        # outage) and returns nothing at all.
        for fr in fts_results:
            entity_names.update(e.strip() for e in fr.get("entities", []) if e and e.strip())

        for entity in entity_names:
            ranked = self.knowledge_graph.get_connected_chunks_ranked(
                entity, hops=settings.graph_hop_depth
            )
            for chunk_id, dist in ranked.items():
                if chunk_id not in chunk_distance or dist < chunk_distance[chunk_id]:
                    chunk_distance[chunk_id] = dist

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
            fts_normalized = _normalize_fts_rank(fr["fts_rank"])
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

        # Graph results — boost chunks other signals found, and pull in
        # connected chunks they missed (this is what "connects the dots":
        # a chunk with no lexical or semantic overlap still surfaces when
        # it shares entities with the ones that matched)
        # Prefer nearer (fewer-hop) connections when the additive cap bites: a
        # 1-hop co-occurrence is more relevant than a distant 2-hop link. Sorting
        # by (distance, id) also keeps the cap deterministic across PYTHONHASHSEED.
        added_from_graph = 0
        for chunk_id in sorted(chunk_distance, key=lambda cid: (chunk_distance[cid], cid)):
            if chunk_id in scored:
                scored[chunk_id]["graph_score"] = 0.3
                continue
            if added_from_graph >= 50:
                continue
            chunk = self.memory_store.get_chunk(chunk_id)
            if chunk is None:
                continue
            added_from_graph += 1
            scored[chunk_id] = {
                "chunk_id": chunk_id,
                "content": chunk.content,
                "vector_score": 0.0,
                "fts_score": 0.0,
                "graph_score": 0.3,
                "metadata": {
                    "source": chunk.source.value,
                    "timestamp": chunk.timestamp.isoformat(),
                    "entities": ",".join(chunk.entities),
                },
            }

        # 5. Compute final scores
        access_counts = self.memory_store.access_counts_by_chunk()

        # Query-adaptive fusion: connection-oriented intents lean on the graph.
        w_vector, w_fts, w_graph = fusion_weights(intent)

        results = []
        for item in scored.values():
            combined = (
                w_vector * item["vector_score"]
                + w_fts * item["fts_score"]
                + w_graph * item["graph_score"]
            )

            # Apply temporal decay, access-frequency boost, and connection boost
            ts = self._parse_timestamp(item["metadata"].get("timestamp", ""))
            entities = [e for e in item["metadata"].get("entities", "").split(",") if e.strip()]
            conn_count = sum(self.knowledge_graph.connection_count(e) for e in entities)
            importance = self.scorer.score(
                base_importance=combined,
                timestamp=ts,
                access_count=access_counts.get(item["chunk_id"], 0),
                connection_count=conn_count,
                half_life_days=half_life,
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
                    # Pre-decay fused relevance (clamped: 1 - cosine_distance can
                    # go slightly negative). Confidence reads this, not `score`,
                    # so it doesn't inherit temporal decay or intent tuning.
                    relevance=max(0.0, min(1.0, combined)),
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
                r
                for r in results
                if (date_from is None or r.timestamp >= date_from)
                and (date_to is None or r.timestamp <= date_to)
            ]

        # 8. Sort by score descending and return top K
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:final_k]

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        if not ts_str:
            return utcnow()
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return utcnow()
