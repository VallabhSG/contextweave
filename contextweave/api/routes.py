"""FastAPI routes for ContextWeave."""

import logging
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from contextweave.api.rate_limit import BATCH_LIMIT, INGEST_LIMIT, QUERY_LIMIT, limiter
from contextweave.ingestion.calendar_adapter import CalendarAdapter
from contextweave.ingestion.chat_adapter import ChatAdapter
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.processing.chunker import SemanticChunker
from contextweave.processing.embedder import LocalEmbedder
from contextweave.processing.entity_extractor import EntityExtractor
from contextweave.processing.importance_scorer import ImportanceScorer
from contextweave.reasoning.engine import ReasoningEngine
from contextweave.retrieval.hybrid_retriever import HybridRetriever
from contextweave.schemas import Memory, SourceType
from contextweave.storage.knowledge_graph import KnowledgeGraph
from contextweave.storage.memory_store import MemoryStore
from contextweave.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Shared instances (initialized lazily) ───────────────────

_instances: dict = {}
_instances_lock = threading.Lock()


def _get(key: str):
    with _instances_lock:
        if key not in _instances:
            _instances["memory_store"] = MemoryStore()
            _instances["vector_store"] = VectorStore()
            _instances["knowledge_graph"] = KnowledgeGraph()
            _instances["embedder"] = LocalEmbedder()
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

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB per file
MAX_BATCH_FILES = 20


# ── Request/Response Models ─────────────────────────────────


class IngestTextRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    source: str = "note"
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    query_type: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    source_filter: str | None = None
    date_from: str | None = None  # ISO date e.g. "2024-01-01"
    date_to: str | None = None  # ISO date e.g. "2024-12-31"


class QueryResponse(BaseModel):
    answer: str
    cited_memories: list[str] = []
    confidence: float = 0.0
    patterns: list[str] = []
    query_type: str = "general"
    context_count: int = 0
    suggested_queries: list[str] = []
    expanded_terms: list[str] = []


class IngestResponse(BaseModel):
    events_created: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    vectors_stored: int = 0
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
@limiter.limit(INGEST_LIMIT)
async def ingest_file(request: Request, file: UploadFile = File(...)):
    """Ingest a file (text, markdown, JSON chat, ICS calendar)."""
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    adapter = ADAPTERS.get(suffix)

    if not adapter:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        events = await run_in_threadpool(adapter.ingest_file, tmp_path)
        return await run_in_threadpool(_process_events, events)
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/ingest/text", response_model=IngestResponse)
@limiter.limit(INGEST_LIMIT)
async def ingest_text(request: Request, req: IngestTextRequest):
    """Ingest raw text content."""
    try:
        source = SourceType(req.source) if req.source else SourceType.NOTE
    except ValueError:
        raise HTTPException(400, f"Unknown source type: {req.source!r}") from None

    adapter = TextAdapter()
    events = adapter.ingest_text(req.content, metadata=req.metadata, source=source)
    return await run_in_threadpool(_process_events, events)


@router.post("/ingest/batch", response_model=IngestResponse)
@limiter.limit(BATCH_LIMIT)
async def ingest_batch(request: Request, files: list[UploadFile] = File(...)):
    """Batch ingest multiple files."""
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f"Too many files (max {MAX_BATCH_FILES} per batch)")

    total_events = []

    for file in files:
        suffix = Path(file.filename or "upload.txt").suffix.lower()
        adapter = ADAPTERS.get(suffix)
        if not adapter:
            continue

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"File {file.filename!r} too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            events = await run_in_threadpool(adapter.ingest_file, tmp_path)
            total_events.extend(events)
        finally:
            tmp_path.unlink(missing_ok=True)

    return await run_in_threadpool(_process_events, total_events)


def _process_events(events) -> IngestResponse:
    """Shared pipeline: chunk → embed → extract entities → store."""
    if not events:
        return IngestResponse(message="No content extracted from input")

    store: MemoryStore = _get("memory_store")
    vstore: VectorStore = _get("vector_store")
    graph: KnowledgeGraph = _get("knowledge_graph")
    chunker: SemanticChunker = _get("chunker")
    embedder: LocalEmbedder = _get("embedder")
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
        message=f"Ingested {len(events)} events into {len(chunks)} chunks ({vectors_stored} vectors stored)",
    )


# ── Query Endpoints ─────────────────────────────────────────


def _parse_query_date(value: str | None, field_name: str) -> datetime | None:
    """Parse an ISO date/datetime filter into naive UTC, or raise 400."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            400, f"Invalid {field_name}: expected ISO format like 2026-01-31"
        ) from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _run_query(req: QueryRequest) -> QueryResponse:
    """Shared retrieval + reasoning pipeline for the query endpoints."""
    retriever: HybridRetriever = _get("retriever")
    reasoning: ReasoningEngine = _get("reasoning")
    store: MemoryStore = _get("memory_store")
    graph: KnowledgeGraph = _get("knowledge_graph")

    # Parse optional date range (fail fast, before spending an LLM call)
    date_from = _parse_query_date(req.date_from, "date_from")
    date_to = _parse_query_date(req.date_to, "date_to")
    if date_to and req.date_to and len(req.date_to) == 10:
        # A bare date means "through the end of that day", not midnight
        date_to = date_to.replace(hour=23, minute=59, second=59)

    # Query expansion
    expanded_terms = reasoning.expand_query(req.query)

    results = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        source_filter=req.source_filter,
        date_from=date_from,
        date_to=date_to,
        extra_terms=expanded_terms,
    )

    response = reasoning.reason(
        query=req.query,
        results=results,
        query_type=req.query_type,
        knowledge_graph=graph,
    )

    # Record access for cited chunks (best-effort, never fails the query)
    for chunk_id in response.cited_memories:
        try:
            store.record_chunk_access(chunk_id)
        except Exception as e:
            logger.debug("Could not record access for chunk %s: %s", chunk_id, e)

    return QueryResponse(
        answer=response.answer,
        cited_memories=response.cited_memories,
        confidence=response.confidence,
        patterns=response.patterns,
        query_type=response.query_type,
        context_count=len(results),
        suggested_queries=response.suggested_queries,
        expanded_terms=expanded_terms,
    )


@router.post("/query", response_model=QueryResponse)
@limiter.limit(QUERY_LIMIT)
def query_memories(request: Request, req: QueryRequest):
    """Natural language query against your memory."""
    return _run_query(req)


@router.post("/query/patterns", response_model=QueryResponse)
@limiter.limit(QUERY_LIMIT)
def detect_patterns(request: Request, req: QueryRequest):
    """Detect patterns across recent context."""
    req_with_type = QueryRequest(
        query=req.query,
        query_type="patterns",
        top_k=req.top_k,
        source_filter=req.source_filter,
    )
    return _run_query(req_with_type)


# ── Memory Endpoints ────────────────────────────────────────


@router.get("/memories")
def list_memories(
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


@router.get("/memories/top/accessed")
def top_accessed_memories(limit: int = Query(default=20, le=100)):
    """List most frequently accessed memories."""
    store: MemoryStore = _get("memory_store")
    memories = store.list_most_accessed(limit=limit)
    return {"memories": [m.model_dump() for m in memories], "count": len(memories)}


@router.get("/memories/{memory_id}")
def get_memory(memory_id: str):
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
def list_entities(limit: int = Query(default=100, le=500)):
    """List all known entities and their connections."""
    graph: KnowledgeGraph = _get("knowledge_graph")
    entities = graph.list_entities(limit=limit)
    return {"entities": [e.model_dump() for e in entities], "count": len(entities)}


@router.get("/graph/entity/{name}")
def get_entity(name: str):
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


# ── Debug ───────────────────────────────────────────────────


@router.get("/debug/status")
def debug_status():
    """Test embedding and generation stack health."""
    from contextweave.config import settings as cfg

    embed_ok, embed_error = False, ""
    gen_ok, gen_error = False, ""
    groq_key_set = bool(cfg.groq_api_key)

    # Test local fastembed
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=cfg.embedding_model)
        embs = list(model.embed(["test"]))
        embed_ok = len(embs[0]) > 0
    except Exception as e:
        embed_error = str(e)

    # Test Groq generation
    if groq_key_set:
        try:
            from groq import Groq

            client = Groq(api_key=cfg.groq_api_key)
            r = client.chat.completions.create(
                model=cfg.reasoning_model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            gen_ok = bool(r.choices[0].message.content)
        except Exception as e:
            gen_error = str(e)
    else:
        gen_error = "CW_GROQ_API_KEY not set — get a free key at console.groq.com"

    return {
        "embedding_backend": "fastembed (local)",
        "embedding_model": cfg.embedding_model,
        "embedding_ok": embed_ok,
        "embedding_error": embed_error,
        "generation_backend": "groq",
        "generation_model": cfg.reasoning_model,
        "groq_key_set": groq_key_set,
        "generation_ok": gen_ok,
        "generation_error": gen_error,
    }


# Keep old path as alias so existing bookmarks still work
@router.get("/debug/gemini")
def debug_gemini():
    """Alias for /debug/status."""
    return debug_status()


# ── Health ──────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
def health():
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
