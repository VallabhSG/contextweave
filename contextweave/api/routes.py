"""FastAPI routes for ContextWeave."""

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from contextweave.api.account import router as account_router
from contextweave.api.audio import router as audio_router
from contextweave.api.deps import get_workspace
from contextweave.api.digest_routes import router as digest_router
from contextweave.api.pipeline import IngestResponse, process_events
from contextweave.api.rate_limit import BATCH_LIMIT, INGEST_LIMIT, QUERY_LIMIT, limiter
from contextweave.ingestion.calendar_adapter import CalendarAdapter
from contextweave.ingestion.chat_adapter import ChatAdapter
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.schemas import SourceType
from contextweave.workspaces import Workspace, manager

logger = logging.getLogger(__name__)

router = APIRouter()

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


class HealthResponse(BaseModel):
    status: str = "ok"
    events: int = 0
    chunks: int = 0
    memories: int = 0
    vectors: int = 0
    entities: int = 0
    edges: int = 0


class VersionResponse(BaseModel):
    """Build + runtime configuration — no secrets, safe to expose publicly.

    Lets anyone confirm what is actually deployed and which features are live
    (health alone can't distinguish builds or verify a config change landed).
    """

    app: str = "contextweave"
    version: str = "0.1.0"
    commit: str = "unknown"
    storage_backend: str = "sqlite"
    reranking_enabled: bool = False
    fallback_configured: bool = False
    reasoning_model: str = ""
    embedding_model: str = ""


# ── Ingestion Endpoints ────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit(INGEST_LIMIT)
async def ingest_file(
    request: Request, file: UploadFile = File(...), ws: Workspace = Depends(get_workspace)
):
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
        return await run_in_threadpool(process_events, ws, events)
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/ingest/text", response_model=IngestResponse)
@limiter.limit(INGEST_LIMIT)
async def ingest_text(
    request: Request, req: IngestTextRequest, ws: Workspace = Depends(get_workspace)
):
    """Ingest raw text content."""
    try:
        source = SourceType(req.source) if req.source else SourceType.NOTE
    except ValueError:
        raise HTTPException(400, f"Unknown source type: {req.source!r}") from None

    adapter = TextAdapter()
    events = adapter.ingest_text(req.content, metadata=req.metadata, source=source)
    return await run_in_threadpool(process_events, ws, events)


@router.post("/ingest/batch", response_model=IngestResponse)
@limiter.limit(BATCH_LIMIT)
async def ingest_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    ws: Workspace = Depends(get_workspace),
):
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

    return await run_in_threadpool(process_events, ws, total_events)


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


def _run_query(ws: Workspace, req: QueryRequest) -> QueryResponse:
    """Shared retrieval + reasoning pipeline for the query endpoints."""
    reasoning = manager.shared().reasoning

    # Parse optional date range (fail fast, before spending an LLM call)
    date_from = _parse_query_date(req.date_from, "date_from")
    date_to = _parse_query_date(req.date_to, "date_to")
    if date_to and req.date_to and len(req.date_to) == 10:
        # A bare date means "through the end of that day", not midnight
        date_to = date_to.replace(hour=23, minute=59, second=59)

    # Query expansion
    expanded_terms = reasoning.expand_query(req.query)

    results = ws.retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        source_filter=req.source_filter,
        date_from=date_from,
        date_to=date_to,
        extra_terms=expanded_terms,
        query_type=req.query_type,
    )

    response = reasoning.reason(
        query=req.query,
        results=results,
        query_type=req.query_type,
        knowledge_graph=ws.knowledge_graph,
    )

    # Record access for cited chunks (best-effort, never fails the query)
    for chunk_id in response.cited_memories:
        try:
            ws.memory_store.record_chunk_access(chunk_id)
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
def query_memories(request: Request, req: QueryRequest, ws: Workspace = Depends(get_workspace)):
    """Natural language query against your memory."""
    return _run_query(ws, req)


@router.post("/query/patterns", response_model=QueryResponse)
@limiter.limit(QUERY_LIMIT)
def detect_patterns(request: Request, req: QueryRequest, ws: Workspace = Depends(get_workspace)):
    """Detect patterns across recent context."""
    req_with_type = QueryRequest(
        query=req.query,
        query_type="patterns",
        top_k=req.top_k,
        source_filter=req.source_filter,
    )
    return _run_query(ws, req_with_type)


# ── Memory Endpoints ────────────────────────────────────────


@router.get("/memories")
def list_memories(
    source: str | None = None,
    min_importance: float = 0.0,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    ws: Workspace = Depends(get_workspace),
):
    """List memories filtered by source, importance, with pagination."""
    memories = ws.memory_store.list_memories(
        source=source,
        min_importance=min_importance,
        limit=limit,
        offset=offset,
    )
    return {"memories": [m.model_dump() for m in memories], "count": len(memories)}


@router.get("/memories/top/accessed")
def top_accessed_memories(
    limit: int = Query(default=20, le=100), ws: Workspace = Depends(get_workspace)
):
    """List most frequently accessed memories."""
    memories = ws.memory_store.list_most_accessed(limit=limit)
    return {"memories": [m.model_dump() for m in memories], "count": len(memories)}


@router.get("/memories/{memory_id}")
def get_memory(memory_id: str, ws: Workspace = Depends(get_workspace)):
    """Get a specific memory with its connected entities."""
    memory = ws.memory_store.get_memory(memory_id)
    if not memory:
        raise HTTPException(404, "Memory not found")

    connections = {}
    for entity_name in memory.entities:
        entity = ws.knowledge_graph.get_entity(entity_name)
        if entity:
            connections[entity_name] = entity.model_dump()

    return {"memory": memory.model_dump(), "connections": connections}


# ── Graph Endpoints ─────────────────────────────────────────


@router.get("/graph/entities")
def list_entities(limit: int = Query(default=100, le=500), ws: Workspace = Depends(get_workspace)):
    """List all known entities and their connections."""
    entities = ws.knowledge_graph.list_entities(limit=limit)
    return {"entities": [e.model_dump() for e in entities], "count": len(entities)}


@router.get("/graph/entity/{name}")
def get_entity(name: str, ws: Workspace = Depends(get_workspace)):
    """Get all memories connected to an entity."""
    entity = ws.knowledge_graph.get_entity(name)
    if not entity:
        raise HTTPException(404, f"Entity '{name}' not found")

    chunk_ids = ws.knowledge_graph.get_connected_chunks(name, hops=2)

    chunks = []
    for cid in chunk_ids[:50]:
        chunk = ws.memory_store.get_chunk(cid)
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


@router.get("/version", response_model=VersionResponse)
def version():
    """Deployed build + which features are live (no secrets)."""
    import os

    from contextweave.config import settings as cfg

    return VersionResponse(
        commit=os.environ.get("CW_COMMIT", "unknown"),
        storage_backend="postgres" if cfg.database_url else "sqlite",
        reranking_enabled=bool(cfg.rerank_model),
        fallback_configured=bool(
            cfg.fallback_base_url and cfg.fallback_api_key and cfg.fallback_model
        ),
        reasoning_model=cfg.reasoning_model,
        embedding_model=cfg.embedding_model,
    )


@router.get("/health", response_model=HealthResponse)
def health(ws: Workspace = Depends(get_workspace)):
    """Health and statistics for the caller's workspace."""
    db_stats = ws.memory_store.stats()
    graph_stats = ws.knowledge_graph.stats()

    return HealthResponse(
        status="ok",
        events=db_stats["events"],
        chunks=db_stats["chunks"],
        memories=db_stats["memories"],
        vectors=ws.vector_store.count(),
        entities=graph_stats["entities"],
        edges=graph_stats["edges"],
    )


# ── Sub-routers ─────────────────────────────────────────────

router.include_router(account_router)
router.include_router(audio_router)
router.include_router(digest_router)
