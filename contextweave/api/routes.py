"""FastAPI routes for ContextWeave."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from contextweave.ingestion.calendar_adapter import CalendarAdapter
from contextweave.ingestion.chat_adapter import ChatAdapter
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.processing.chunker import SemanticChunker
from contextweave.processing.embedder import GeminiEmbedder
from contextweave.processing.entity_extractor import EntityExtractor
from contextweave.processing.importance_scorer import ImportanceScorer
from contextweave.reasoning.engine import ReasoningEngine
from contextweave.retrieval.hybrid_retriever import HybridRetriever
from contextweave.schemas import Memory
from contextweave.storage.knowledge_graph import KnowledgeGraph
from contextweave.storage.memory_store import MemoryStore
from contextweave.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Shared instances (initialized lazily) ───────────────────

_instances: dict = {}


def _get(key: str):
    if key not in _instances:
        _instances["memory_store"] = MemoryStore()
        _instances["vector_store"] = VectorStore()
        _instances["knowledge_graph"] = KnowledgeGraph()
        _instances["embedder"] = GeminiEmbedder()
        _instances["chunker"] = SemanticChunker()
        _instances["extractor"] = EntityExtractor()
        _instances["scorer"] = ImportanceScorer()
        _instances["retriever"] = HybridRetriever(
            vector_store=_instances["vector_store"],
            memory_store=_instances["memory_store"],
            knowledge_graph=_instances["knowledge_graph"],
            embedder=_instances["embedder"],
            scorer=_instances["scorer"],
        )
        _instances["reasoning"] = ReasoningEngine()
    return _instances[key]


ADAPTERS = {
    ".txt": TextAdapter(),
    ".md": TextAdapter(),
    ".markdown": TextAdapter(),
    ".json": ChatAdapter(),
    ".ics": CalendarAdapter(),
}


# ── Request/Response Models ─────────────────────────────────


class IngestTextRequest(BaseModel):
    content: str
    source: str = "note"
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    query_type: str | None = None
    top_k: int = 8
    source_filter: str | None = None


class QueryResponse(BaseModel):
    answer: str
    cited_memories: list[str] = []
    confidence: float = 0.0
    patterns: list[str] = []
    query_type: str = "general"
    context_count: int = 0


class IngestResponse(BaseModel):
    events_created: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    events: int = 0
    chunks: int = 0
    memories: int = 0
    vectors: int = 0
    entities: int = 0
    edges: int = 0


# ── Ingestion Endpoints ────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a file (text, markdown, JSON chat, ICS calendar)."""
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    adapter = ADAPTERS.get(suffix)

    if not adapter:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        events = adapter.ingest_file(tmp_path)
        return await _process_events(events)
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest):
    """Ingest raw text content."""
    adapter = TextAdapter()
    events = adapter.ingest_text(req.content, metadata=req.metadata)
    return await _process_events(events)


@router.post("/ingest/batch", response_model=IngestResponse)
async def ingest_batch(files: list[UploadFile] = File(...)):
    """Batch ingest multiple files."""
    total_events = []

    for file in files:
        suffix = Path(file.filename or "upload.txt").suffix.lower()
        adapter = ADAPTERS.get(suffix)
        if not adapter:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            events = adapter.ingest_file(tmp_path)
            total_events.extend(events)
        finally:
            tmp_path.unlink(missing_ok=True)

    return await _process_events(total_events)


async def _process_events(events) -> IngestResponse:
    """Shared pipeline: chunk → embed → extract entities → store."""
    if not events:
        return IngestResponse(message="No content extracted from input")

    store: MemoryStore = _get("memory_store")
    vstore: VectorStore = _get("vector_store")
    graph: KnowledgeGraph = _get("knowledge_graph")
    chunker: SemanticChunker = _get("chunker")
    embedder: GeminiEmbedder = _get("embedder")
    extractor: EntityExtractor = _get("extractor")
    scorer: ImportanceScorer = _get("scorer")

    # Save raw events
    store.save_events(events)

    # Chunk
    chunks = chunker.chunk_events(events)

    # Embed
    chunks = embedder.embed_chunks(chunks)

    # Extract entities and attach to chunks
    total_entities = 0
    processed_chunks = []
    for chunk in chunks:
        entities = extractor.extract_from_chunk(chunk)
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
        importance = scorer.estimate_base_importance(chunk.content, chunk.source.value)
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
    vstore.add_chunks(processed_chunks)

    return IngestResponse(
        events_created=len(events),
        chunks_created=len(chunks),
        entities_extracted=total_entities,
        message=f"Successfully ingested {len(events)} events into {len(chunks)} chunks",
    )


# ── Query Endpoints ─────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query_memories(req: QueryRequest):
    """Natural language query against your memory."""
    retriever: HybridRetriever = _get("retriever")
    reasoning: ReasoningEngine = _get("reasoning")

    results = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        source_filter=req.source_filter,
    )

    response = reasoning.reason(
        query=req.query,
        results=results,
        query_type=req.query_type,
    )

    return QueryResponse(
        answer=response.answer,
        cited_memories=response.cited_memories,
        confidence=response.confidence,
        patterns=response.patterns,
        query_type=response.query_type,
        context_count=len(results),
    )


@router.post("/query/patterns", response_model=QueryResponse)
async def detect_patterns(req: QueryRequest):
    """Detect patterns across recent context."""
    req_with_type = QueryRequest(
        query=req.query,
        query_type="patterns",
        top_k=req.top_k,
        source_filter=req.source_filter,
    )
    return await query_memories(req_with_type)


# ── Memory Endpoints ────────────────────────────────────────


@router.get("/memories")
async def list_memories(
    source: str | None = None,
    min_importance: float = 0.0,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    """List memories filtered by source, importance, with pagination."""
    store: MemoryStore = _get("memory_store")
    memories = store.list_memories(
        source=source,
        min_importance=min_importance,
        limit=limit,
        offset=offset,
    )
    return {"memories": [m.model_dump() for m in memories], "count": len(memories)}


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str):
    """Get a specific memory with its connected entities."""
    store: MemoryStore = _get("memory_store")
    memory = store.get_memory(memory_id)
    if not memory:
        raise HTTPException(404, "Memory not found")

    graph: KnowledgeGraph = _get("knowledge_graph")
    connections = {}
    for entity_name in memory.entities:
        entity = graph.get_entity(entity_name)
        if entity:
            connections[entity_name] = entity.model_dump()

    return {"memory": memory.model_dump(), "connections": connections}


# ── Graph Endpoints ─────────────────────────────────────────


@router.get("/graph/entities")
async def list_entities(limit: int = Query(default=100, le=500)):
    """List all known entities and their connections."""
    graph: KnowledgeGraph = _get("knowledge_graph")
    entities = graph.list_entities(limit=limit)
    return {"entities": [e.model_dump() for e in entities], "count": len(entities)}


@router.get("/graph/entity/{name}")
async def get_entity(name: str):
    """Get all memories connected to an entity."""
    graph: KnowledgeGraph = _get("knowledge_graph")
    entity = graph.get_entity(name)
    if not entity:
        raise HTTPException(404, f"Entity '{name}' not found")

    chunk_ids = graph.get_connected_chunks(name, hops=2)

    store: MemoryStore = _get("memory_store")
    chunks = []
    for cid in chunk_ids[:50]:
        chunk = store.get_chunk(cid)
        if chunk:
            chunks.append(
                {
                    "id": chunk.id,
                    "content": chunk.content[:300],
                    "source": chunk.source.value,
                    "timestamp": chunk.timestamp.isoformat(),
                }
            )

    return {"entity": entity.model_dump(), "connected_chunks": chunks}


# ── Health ──────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health():
    """System health and statistics."""
    store: MemoryStore = _get("memory_store")
    vstore: VectorStore = _get("vector_store")
    graph: KnowledgeGraph = _get("knowledge_graph")

    db_stats = store.stats()
    graph_stats = graph.stats()

    return HealthResponse(
        status="ok",
        events=db_stats["events"],
        chunks=db_stats["chunks"],
        memories=db_stats["memories"],
        vectors=vstore.count(),
        entities=graph_stats["entities"],
        edges=graph_stats["edges"],
    )
