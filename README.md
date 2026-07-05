---
title: ContextWeave
emoji: 🧠
colorFrom: yellow
colorTo: gray
sdk: docker
pinned: true
---

# ContextWeave

A personal long-term memory and context reasoning engine. Ingests ambient context from conversations, notes, browser history, and calendar events — then lets you query patterns and answers across your entire life data stream.

Built to demonstrate the core technical challenge behind products like [Thine](https://www.thine.com/): not just storing context, but structuring it, scoring it with temporal decay, and reasoning across it to surface insights you didn't know to ask for.

**Live demo:** [vallllllllll-contextweave.hf.space](https://vallllllllll-contextweave.hf.space) · [Interactive docs](https://vallllllllll-contextweave.hf.space/docs) · [Space page](https://huggingface.co/spaces/Vallllllllll/contextweave)

```bash
# Try it now (no auth required)
curl https://vallllllllll-contextweave.hf.space/api/health
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
├───────────────┬──────────────────────┬──────────────────────┤
│   Ingestion   │     Processing       │     Reasoning        │
│               │                      │                      │
│ TextAdapter   │ SemanticChunker      │ ReasoningEngine      │
│ ChatAdapter   │ LocalEmbedder        │   (Groq Llama)       │
│ BrowserAdapter│ EntityExtractor      │                      │
│ CalendarAdapter│ ImportanceScorer    │ 6 Query Types:       │
│               │  (temporal decay)    │  general, patterns,  │
│               │                      │  gaps, temporal,     │
├───────────────┴──────────────────────┤  cross_ref, priority │
│                Storage               │                      │
│                                      │                      │
│  VectorStore (ChromaDB)              │  HybridRetriever     │
│  MemoryStore (SQLite + FTS5)         │  ┌─ Vector sim       │
│  KnowledgeGraph (NetworkX + SQLite)  │  ├─ FTS keyword      │
│                                      │  └─ Graph traversal  │
└──────────────────────────────────────┴──────────────────────┘
```

### Why this architecture matters

Most naive RAG systems do: embed → cosine search → stuff into LLM. That fails for personal memory because:

1. **Important old memories get buried** by recent but trivial noise → solved with temporal decay scoring
2. **Keyword search misses semantic connections** → solved with hybrid retrieval (vector + FTS + graph)
3. **No understanding of relationships** between people, projects, topics → solved with the knowledge graph
4. **Generic retrieval, generic answers** → solved with 6 specialized reasoning query types

---

## Modules

### Ingestion Layer

Pluggable adapters normalize raw data into `ContextEvent` objects:

| Adapter | Handles | Notes |
|---------|---------|-------|
| `TextAdapter` | `.txt`, `.md` | Splits on H1/H2 headings into logical sections |
| `ChatAdapter` | WhatsApp exports, Slack/Telegram JSON | Windows of 10 turns per event |
| `BrowserAdapter` | Chrome history JSON | Groups by browsing sessions of 15 |
| `CalendarAdapter` | `.ics` iCalendar files | Extracts attendees, location, description |

### Processing Layer

**SemanticChunker** — Splits `ContextEvent` into coherent `Chunk` objects:
- Conversation: split by turn windows, never mid-sentence
- Prose: split on paragraph/section breaks, then sentences if needed
- Configurable `max_tokens` (default: 512) and `overlap_sentences` (default: 2)

**LocalEmbedder** — fastembed with `BAAI/bge-small-en-v1.5` (384-dim vectors):
- Runs fully locally — no API key or network call needed for embeddings
- Batch processing with per-item fallback on failure

**EntityExtractor** — LLM-based NER via Groq (`llama-3.1-8b-instant`):
- Extracts: person, place, project, topic, organization, event
- Fallback regex NER when LLM fails

**ImportanceScorer** — Temporal decay with boosts:
```
importance = base × recency_decay × access_boost × connection_boost

recency_decay = exp(-ln(2) × days_elapsed / half_life)   # half-life: 30 days
access_boost  = 1 + log(1 + access_count) × 1.2
connection_boost = 1 + connection_count × 0.3
```

### Storage Layer

**VectorStore** — ChromaDB with cosine distance, persistent on disk

**MemoryStore** — SQLite with:
- `events` table: raw ingested events
- `chunks` table: processed, entity-tagged chunks
- `memories` table: scored, retrievable memories
- `chunks_fts` virtual table: SQLite FTS5 full-text search

**KnowledgeGraph** — NetworkX in-memory graph backed by SQLite:
- Entity nodes: name, type, first/last seen, mention count
- Edges: co-occurrence edges between entities in the same chunk
- Enables N-hop traversal to expand retrieval context

### Retrieval

**HybridRetriever** fuses three signals:

```
final_score = 0.5 × vector_similarity
            + 0.3 × fts_relevance
            + 0.2 × graph_boost
            × importance_decay(timestamp, access_count, connections)
```

1. Vector search (ChromaDB cosine) → top 20 candidates
2. FTS keyword search (SQLite FTS5) → top 20 candidates
3. Graph expansion from matched entities → connected chunk IDs
4. Merge, score, filter by source if requested
5. Return top K by final score

### Reasoning Engine

Six query types automatically detected from query keywords:

| Type | Trigger keywords | What it does |
|------|-----------------|--------------|
| `general` | (default) | Synthesizes answer with citations |
| `patterns` | pattern, trend, recurring | Finds recurring themes and trends |
| `gaps` | avoiding, missing, overlooking | Surfaces what you're not addressing |
| `temporal` | evolved, changed, over time | Tracks how thinking changed |
| `cross_reference` | think about, what does, opinion | Connects info across sources |
| `priorities` | focus, prioritize, this week | Synthesizes action priorities |

---

## Product pillars (mapped to Thine)

| Thine pillar | ContextWeave implementation |
|---|---|
| Everything connects | Knowledge graph + hybrid retrieval (vector 50% · FTS 30% · graph 20%) with temporal decay |
| A companion that remembers | Events → chunks → scored memories, with access-frequency and connection boosts |
| Proactive nudges | `GET /api/digest` — cached daily digest of focus, commitments, and slipping threads, surfaced automatically in the UI |
| Ambient capture | 🎙 web Listen button (Web Speech API), `POST /api/ingest/audio` (Groq Whisper transcription), and an Expo iPhone/Android companion app ([`mobile/`](mobile/)) that records in 45s segments while you live your life |
| Privacy: your data, your control | Private workspaces via `POST /api/auth/register` — isolated SQLite + vector collection per user, full `GET /api/export`, `DELETE /api/memory` |

Without an API key you're in the **shared public demo space**. With one, every request routes to your own isolated workspace.

```bash
# Create a private space (the key is shown exactly once)
curl -X POST https://vallllllllll-contextweave.hf.space/api/auth/register \
  -H "Content-Type: application/json" -d '{"name": "me"}'

# Use it on every call
curl https://vallllllllll-contextweave.hf.space/api/me -H "X-API-Key: cw_..."

# Your daily nudge
curl https://vallllllllll-contextweave.hf.space/api/digest -H "X-API-Key: cw_..."

# Take your data with you / erase it
curl https://vallllllllll-contextweave.hf.space/api/export -H "X-API-Key: cw_..."
curl -X DELETE https://vallllllllll-contextweave.hf.space/api/memory -H "X-API-Key: cw_..."

# Speak instead of type — audio is transcribed, then discarded
curl -X POST https://vallllllllll-contextweave.hf.space/api/ingest/audio \
  -H "X-API-Key: cw_..." -F "file=@voice-note.m4a"
```

**Mobile ambient capture:** the [`mobile/`](mobile/) Expo app is the "phone on the table" experience — tap Listen and it records 45-second segments, ships each to `/api/ingest/audio`, and shows your nudge. See [mobile/README.md](mobile/README.md).

---

## Quickstart

### Try the live API

```bash
# Health check
curl https://vallllllllll-contextweave.hf.space/api/health

# Ingest a note
curl -X POST https://vallllllllll-contextweave.hf.space/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"content": "Met with Alice today about Project Alpha. We decided to prioritize the memory retrieval layer."}'

# Query it
curl -X POST https://vallllllllll-contextweave.hf.space/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What did I decide about Project Alpha?"}'
```

Interactive docs: [https://vallllllllll-contextweave.hf.space/docs](https://vallllllllll-contextweave.hf.space/docs)

### Run locally

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure

```bash
cp .env.example .env
# Edit .env and set CW_GROQ_API_KEY
```

Get a free Groq API key at console.groq.com.

#### 3. Run

```bash
python -m uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## API Reference

### Ingest

```bash
# Ingest a markdown note
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@notes.md"

# Ingest raw text
curl -X POST http://localhost:8000/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"content": "Had a great meeting with Alice about Project Alpha today."}'

# Ingest a WhatsApp export
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@whatsapp_export.txt"

# Ingest calendar
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@calendar.ics"
```

### Query

```bash
# General question
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What have I been working on this month?"}'

# Pattern detection
curl -X POST http://localhost:8000/api/query/patterns \
  -H "Content-Type: application/json" \
  -d '{"query": "What patterns do you see in my conversations?"}'

# Force a specific query type
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How has my thinking about work-life balance evolved?", "query_type": "temporal"}'
```

### Explore Memory

```bash
# List memories (paginated, filterable)
curl "http://localhost:8000/api/memories?min_importance=0.6&limit=20"

# Get entity and all connected memories
curl "http://localhost:8000/api/graph/entity/Alice"

# Full entity graph
curl "http://localhost:8000/api/graph/entities"

# System stats
curl "http://localhost:8000/api/health"
```

---

## Running Tests

```bash
pytest -v
```

Tests cover: chunking, importance scoring, memory store CRUD, FTS search, ingestion adapters.

---

## Configuration Reference

All settings use the `CW_` prefix as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CW_GROQ_API_KEY` | — | **Required.** Groq API key |
| `CW_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local fastembed model |
| `CW_REASONING_MODEL` | `llama-3.1-8b-instant` | Groq LLM for reasoning |
| `CW_EXTRACTION_MODEL` | `llama-3.1-8b-instant` | Groq LLM for entity extraction |
| `CW_DECAY_HALF_LIFE_DAYS` | `30.0` | Memory half-life in days |
| `CW_RETRIEVAL_FINAL_K` | `8` | Max results returned |
| `CW_RETRIEVAL_TOP_K` | `20` | Candidates per retrieval signal |
| `CW_GRAPH_HOP_DEPTH` | `2` | Graph traversal depth |
| `CW_CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage path |
| `CW_SQLITE_DB_PATH` | `./contextweave.db` | SQLite database path |
| `CW_PORT` | `8000` | Server port |

---

## Data Sources to Try

Start with these easy-to-export personal data sources:

- **Notes**: Any `.md` or `.txt` files from Obsidian, Notion export, Apple Notes export
- **WhatsApp**: Settings → Chats → Export Chat → Without Media
- **Telegram**: Settings → Export Telegram Data → JSON format
- **Chrome History**: DevTools → Application → Storage → `history.json` or use export extensions
- **Google Calendar**: calendar.google.com → Settings → Import & Export → Export

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Embeddings | fastembed BAAI/bge-small-en-v1.5 (384-dim, local) |
| LLM Reasoning | Groq llama-3.1-8b-instant (free tier) |
| Vector Store | ChromaDB (local, persistent) |
| Relational + FTS | SQLite + FTS5 |
| Knowledge Graph | NetworkX + SQLite |
| Validation | Pydantic v2 |
| Tests | pytest |

**100% free to run.** No paid services required beyond a Groq API key (free tier: 14,400 req/day).
