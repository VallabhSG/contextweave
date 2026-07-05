"""Shared ingestion pipeline: chunk → embed → extract entities → store."""

import logging

from pydantic import BaseModel

from contextweave.schemas import Memory
from contextweave.workspaces import Workspace, manager

logger = logging.getLogger(__name__)


class IngestResponse(BaseModel):
    events_created: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    vectors_stored: int = 0
    message: str = ""


def process_events(ws: Workspace, events) -> IngestResponse:
    """Run context events through the full pipeline into a workspace."""
    if not events:
        return IngestResponse(message="No content extracted from input")

    shared = manager.shared()
    store = ws.memory_store
    vstore = ws.vector_store
    graph = ws.knowledge_graph

    # Save raw events
    store.save_events(events)

    # Chunk
    chunks = shared.chunker.chunk_events(events)

    # Embed
    chunks = shared.embedder.embed_chunks(chunks)

    # Extract entities and attach to chunks
    total_entities = 0
    processed_chunks = []
    for chunk in chunks:
        entities = shared.extractor.extract_from_chunk(chunk)
        entity_names = [e.name for e in entities]
        chunk = chunk.model_copy(update={"entities": entity_names})
        total_entities += len(entities)
        processed_chunks.append(chunk)

        # Update knowledge graph
        if entities:
            graph.add_entities(entities, chunk.id)

        # Save chunk to SQLite
        store.save_chunk(chunk)

        # Create memory from chunk
        importance = shared.scorer.estimate_base_importance(chunk.content, chunk.source.value)
        memory = Memory(
            chunk_ids=[chunk.id],
            summary=chunk.content[:200],
            entities=entity_names,
            source=chunk.source,
            timestamp=chunk.timestamp,
            importance=importance,
        )
        store.save_memory(memory)

    # Store embeddings in vector store (with entity metadata attached)
    vectors_stored = vstore.add_chunks(processed_chunks)

    if vectors_stored == 0:
        logger.warning(
            "No vectors stored for %d chunks — local fastembed embedding may be failing. "
            "See /api/debug/status for diagnostics.",
            len(processed_chunks),
        )

    return IngestResponse(
        events_created=len(events),
        chunks_created=len(chunks),
        entities_extracted=total_entities,
        vectors_stored=vectors_stored,
        message=(
            f"Ingested {len(events)} events into {len(chunks)} chunks "
            f"({vectors_stored} vectors stored)"
        ),
    )
