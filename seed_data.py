"""Seed ContextWeave with realistic sample data."""

import httpx
import time

BASE = "https://huggingface.co/spaces/Vallllllllll/contextweave"

ENTRIES = [
    # ── MEETINGS ────────────────────────────────────────────────
    {
        "content": """Meeting with Pratyush and the Thine product team — March 12, 2026.
Discussed the core challenge: most AI assistants have no memory of who you actually are.
Pratyush walked through the vision: Thine should feel like a co-founder who's been with you since day one.
Key decisions made:
- Prioritize ambient capture over manual note-taking
- iOS first, Android in Q3
- Memory architecture needs temporal decay — recent context should outweigh older context
- Action item: explore hybrid retrieval combining vector search + knowledge graph
Follow-up: send architecture proposal by March 15.""",
        "metadata": {"source": "meeting", "people": ["Pratyush", "Thine team"], "date": "2026-03-12"},
    },
    {
        "content": """Weekly sync with Alice Chen (eng lead) — March 18, 2026.
Alice flagged that ChromaDB cold-start latency is spiking on Render free tier (~30s first request).
Options discussed:
1. Keep SQLite FTS as a warm fallback while Chroma initialises
2. Move to Render paid tier ($7/mo) — probably worth it for the demo
3. Pre-warm with a health-check ping every 10 minutes via a cron job
Decision: go with option 3 for now, revisit after demo.
Alice also mentioned she's been reading the Memorize paper — the access-frequency boost idea came from there.""",
        "metadata": {"source": "meeting", "people": ["Alice Chen"], "date": "2026-03-18"},
    },
    {
        "content": """Design review with Marcus (product designer) — March 25, 2026.
Marcus showed three directions for the ContextWeave UI:
A) Dark, techy (feels like a dev tool)
B) Light, editorial — closer to Thine's brand language
C) Neutral minimal — safe but forgettable
We all agreed B is the right call. The product is personal and human — the design should match.
Marcus will prototype the Instrument Serif headline treatment and share by EOW.
Key insight from Marcus: "The user shouldn't feel like they're querying a database. They should feel like they're talking to someone who knows them." """,
        "metadata": {"source": "meeting", "people": ["Marcus"], "date": "2026-03-25"},
    },
    {
        "content": """1:1 with Rohan (CTO) — April 1, 2026.
Rohan pushed back on the entity extraction approach — using Gemini for NER adds latency and cost.
Counter-argument: the knowledge graph connections are what differentiate us from plain RAG.
Without entity extraction, you can't do graph traversal, and without graph traversal you lose the 20% retrieval signal that catches cross-document connections.
Rohan agreed to keep it but wants a regex fallback for when the LLM is slow.
Also discussed: open-sourcing the memory core. Rohan thinks it could drive developer adoption.
Decision: ship private demo first, evaluate open-source after Series A.""",
        "metadata": {"source": "meeting", "people": ["Rohan"], "date": "2026-04-01"},
    },

    # ── JOURNAL ENTRIES ─────────────────────────────────────────
    {
        "content": """Journal — February 3, 2026.
Started thinking seriously about the memory problem in AI today. Read three papers:
- MemGPT (hierarchical memory management)
- Generative Agents (Stanford — agents with believable daily routines from memory)
- Memorize (access-frequency weighted recall)
The common thread: naive RAG forgets. Top-K cosine search doesn't care if you accessed a memory 50 times or never. That's wrong.
Your most-recalled memories should get a boost. Your recent decisions should outweigh older ones.
I want to build something that gets smarter the longer you use it.""",
        "metadata": {"source": "journal", "date": "2026-02-03"},
    },
    {
        "content": """Journal — February 17, 2026.
Had a long conversation with Dad tonight. He asked what I was building and I couldn't explain it simply.
Eventually landed on: "It's like if your phone remembered every important conversation you ever had and could tell you what you'd forgotten."
He said: "That's just a really good diary."
Maybe he's right. But a diary doesn't answer back. Doesn't connect your conversation from three months ago to your goal from last week.
The insight I keep coming back to: memory isn't storage. Memory is retrieval. The hard part isn't saving — it's knowing what to surface, and when.""",
        "metadata": {"source": "journal", "date": "2026-02-17"},
    },
    {
        "content": """Journal — March 5, 2026.
Imposter syndrome hit hard today. Looked at the Rewind AI team — 40 engineers, $30M raised.
Then looked at what I've built: 17 Python files, a SQLite database, and a FastAPI wrapper around three Google APIs.
But then I ran a query: "What have I been learning about retrieval systems?"
And it answered. Accurately. With citations from my own notes.
Something clicked. The product actually works. Not at scale, not in production, but the core loop — ingest, store, retrieve, reason — is real.
Reminded myself: Thine started as a weekend project too.""",
        "metadata": {"source": "journal", "date": "2026-03-05"},
    },
    {
        "content": """Journal — March 29, 2026.
Shipped the hybrid retriever today. Three signals fused:
- Vector similarity (ChromaDB cosine): 50%
- Full-text search (SQLite FTS5): 30%
- Knowledge graph traversal (NetworkX): 20%
Then importance-reranked with temporal decay.
Tested with 200 seeded entries. Query: "What did I decide about the architecture?"
It returned the right meeting note from 6 weeks ago — ranked above a more recent but less relevant journal entry.
The temporal decay + access boost is doing real work. Old but frequently recalled memories stay relevant.
This is the part that makes it feel less like a search engine and more like a brain.""",
        "metadata": {"source": "journal", "date": "2026-03-29"},
    },
    {
        "content": """Journal — April 3, 2026.
Applied to Thine today. Felt vulnerable attaching the GitHub link.
The project is rough in places — no auth, cold-start latency, entity extraction is sometimes wrong.
But it demonstrates the thing I believe: that personal memory is a systems problem, not a prompting problem.
You can't prompt your way to good recall. You need the right retrieval architecture.
I genuinely think the hybrid retriever + importance scoring approach is better than what most production memory systems do.
Whether or not I get the job, I'll keep building this.""",
        "metadata": {"source": "journal", "date": "2026-04-03"},
    },

    # ── LEARNING NOTES ──────────────────────────────────────────
    {
        "content": """Learning notes: Vector databases deep-dive — February 8, 2026.
Compared ChromaDB, Qdrant, Weaviate, Pinecone.
ChromaDB: best for local/prototype, simple API, persistent on disk. Latency spikes on cold start.
Qdrant: better production performance, filtering support, gRPC. More complex setup.
Weaviate: hybrid search built-in (BM25 + vector). Interesting for our use case but heavy.
Pinecone: fully managed, lowest operational overhead, but $20/mo minimum.
Decision: ChromaDB for the demo (zero cost, easy setup), Qdrant for production.
Key insight: HNSW indexing (Hierarchical Navigable Small World) is what makes ANN fast.
O(log n) at query time instead of brute-force O(n). The graph structure allows efficient navigation through high-dimensional space.""",
        "metadata": {"source": "note", "date": "2026-02-08"},
    },
    {
        "content": """Learning notes: Temporal reasoning in RAG — February 20, 2026.
Most RAG papers optimise for relevance without considering time.
Problems this causes:
1. A decision you made last week should outweigh a preference you noted last year
2. Frequently-accessed memories signal importance — retrieve them more
3. Context evolves: what you believed about a topic in January might be wrong by April
Approaches reviewed:
- Time-weighted embedding: blend semantic similarity with recency score
- Decay functions: exponential decay (half-life model) vs. linear decay
- Access frequency boosting: multiplicative boost proportional to log(access_count + 1)
Implemented: importance = base × exp(-ln(2) × days/30) × (1 + log(1+access) × 1.2)
Half-life of 30 days means a memory loses half its recency weight in a month. Feels right.""",
        "metadata": {"source": "note", "date": "2026-02-20"},
    },
    {
        "content": """Learning notes: Knowledge graphs for RAG — March 2, 2026.
Standard RAG misses cross-document connections. Example:
- Document A mentions "Alice" and "Project Alpha"
- Document B mentions "Project Alpha" risks
- Query: "What should I know before talking to Alice?"
Plain vector search won't connect Alice → Project Alpha → Project Alpha risks.
Knowledge graph solution:
1. Extract entities (NER) from each chunk
2. Create co-occurrence edges: entities in same chunk get an edge
3. At query time: retrieve via vector, then expand to connected entities, then retrieve their chunks too
Implemented with NetworkX (in-memory) + SQLite (persistent edge store).
2-hop traversal catches most useful connections without exploding the result set.""",
        "metadata": {"source": "note", "date": "2026-03-02"},
    },
    {
        "content": """Learning notes: SQLite FTS5 full-text search — March 10, 2026.
FTS5 is SQLite's built-in full-text search. Faster than LIKE, supports ranking.
Key functions:
- bm25(): BM25 ranking (better than TF-IDF for shorter documents)
- highlight(): returns snippets with matched terms highlighted
- snippet(): similar, more control over fragment length
Gotcha discovered: FTS5 queries crash on special characters like *, (, ", :
Fix: sanitise query with re.sub(r'["()*:]', ' ', query) before passing to FTS5.
Also: MATCH queries are case-insensitive by default — good for user queries.
Performance: FTS5 on 10k rows returns in <5ms. Vector search on same size: ~50ms.
That's why FTS gets 30% weight — it's fast, precise for exact terms, and handles proper nouns well.""",
        "metadata": {"source": "note", "date": "2026-03-10"},
    },
    {
        "content": """Learning notes: Pydantic v2 migration notes — March 16, 2026.
Migrating from Pydantic v1 to v2 — breaking changes:
- @validator → @field_validator (different signature)
- orm_mode = True → model_config = ConfigDict(from_attributes=True)
- .dict() → .model_dump()
- .json() → .model_dump_json()
- frozen=True still works for immutable models (good)
Performance improvement: v2 is ~5-50x faster for validation due to Rust core (pydantic-core).
Gotcha: pydantic-core has no prebuilt wheel for Python 3.14. Must pin to Python 3.11 or 3.12 for production deployments (especially Render, Railway).
Lesson: always pin Python version in .python-version file when deploying.""",
        "metadata": {"source": "note", "date": "2026-03-16"},
    },

    # ── CONVERSATIONS ────────────────────────────────────────────
    {
        "content": """Conversation with Sarah (friend, ML engineer at Google) — March 8, 2026.
Sarah: "Why are you building this instead of just using Notion AI or ChatGPT memory?"
Me: "Because those are prompt-level features. They don't actually think about retrieval."
Sarah: "What do you mean?"
Me: "ChatGPT memory is basically a summary injected into every prompt. It doesn't scale, it doesn't decay, it doesn't distinguish between important decisions and random notes."
Sarah: "Fair. So what makes yours different?"
Me: "The importance scorer. Every memory has a score that decays over time unless it gets accessed. Your brain does the same thing — things you think about often stay sharp."
Sarah: "That's actually interesting. Have you read the Ebbinghaus forgetting curve literature?"
Me: "Not deeply. Should I?"
Sarah: "Yes. And look at spaced repetition too — Anki's algorithm is basically what you're describing but for learning."
Added to reading list: Ebbinghaus forgetting curve, SM-2 spaced repetition algorithm.""",
        "metadata": {"source": "conversation", "people": ["Sarah"], "date": "2026-03-08"},
    },
    {
        "content": """Conversation with James (investor, Sequoia) — March 22, 2026.
James: "What's the moat here? Big tech can build memory into their assistants."
Me: "They will. But they'll do it generically. The moat is personal — the longer you use it, the better it knows you specifically. That graph of entities and relationships is yours."
James: "Privacy concern?"
Me: "Everything on-device or user-controlled cloud. Thine's whole brand is privacy-first."
James: "What's the wedge? Why do I sign up?"
Me: "Professionals who network heavily — founders, investors, executives. People who have 20 meaningful conversations a week and forget 80% of what was said."
James: "Interesting. What's the revenue model?"
Me: "Subscription. $200/mo for the full memory tier, lower tiers for lighter use."
James seemed genuinely interested. He mentioned Rewind as a comp but agreed the mobile-first angle is differentiated.""",
        "metadata": {"source": "conversation", "people": ["James", "Sequoia"], "date": "2026-03-22"},
    },
    {
        "content": """Conversation with Mom — April 2, 2026.
Mom asked how the job application is going.
I explained I'm applying to Thine — a startup building personal AI memory.
She asked what I'd be doing there.
Explained: building the memory and retrieval systems — the part that makes the AI actually remember things accurately rather than hallucinating.
She said: "So like a really good filing system?"
Me: "Kind of. But one that knows which files you need before you ask."
She told me to make sure I show them what I've built. Said: "Don't just apply. Show them you already understand the problem."
That conversation made me realise the demo needs to be live and impressive, not just a GitHub repo.""",
        "metadata": {"source": "conversation", "people": ["Mom"], "date": "2026-04-02"},
    },

    # ── DECISIONS & PRIORITIES ───────────────────────────────────
    {
        "content": """Key architectural decisions — ContextWeave v0.1 — March 14, 2026.
After two weeks of prototyping, locked in the following:

Storage:
- ChromaDB for vectors (cosine similarity, persistent, zero-config)
- SQLite for relational data + FTS5 full-text search
- NetworkX + SQLite for the knowledge graph

Processing:
- Gemini text-embedding-004 for embeddings (768-dim, free tier: 1500 req/day)
- Gemini gemini-2.0-flash for NER and reasoning (fast, cheap)
- SemanticChunker with 512-token max, 2-sentence overlap

Retrieval:
- Hybrid fusion: 50% vector + 30% FTS + 20% graph
- Temporal decay half-life: 30 days
- Final top-K: 8 results

These decisions optimise for: zero infrastructure cost, reasonable latency, and a complete demonstration of the memory pipeline.""",
        "metadata": {"source": "note", "date": "2026-03-14"},
    },
    {
        "content": """Priorities for the week of April 7, 2026.
Critical:
1. Get ContextWeave deployed and live on Render — need the URL for the application
2. Seed with enough data to make the demo impressive (queries should return non-trivial answers)
3. Fix the Python version issue on Render (pin to 3.11)

High:
4. Design the frontend to match Thine's editorial, light aesthetic
5. Make sure the /query endpoint returns structured responses (query_type, confidence, patterns)

Medium:
6. Write a demo script showing the full ingest → query loop in < 2 minutes
7. Add the live URL to the README

Not this week:
- Auth system
- Multi-user support
- Mobile app
- Webhook integrations""",
        "metadata": {"source": "note", "date": "2026-04-07"},
    },
    {
        "content": """Reflection: what I've learned building ContextWeave — April 4, 2026.
Technical learnings:
- Hybrid retrieval is meaningfully better than pure vector search for personal context
- Temporal decay is underused in production RAG systems — the literature supports it but few products implement it
- SQLite FTS5 punches above its weight — it's fast, built-in, and handles proper nouns better than embeddings
- Knowledge graphs for RAG are worth the complexity — 2-hop traversal catches connections pure vector search misses

Product learnings:
- The hardest part is the chunking strategy — wrong chunk boundaries destroy retrieval quality
- Users need to feel the system working — the pipeline animation and live stats matter psychologically
- "Memory" is more relatable than "RAG" or "vector search" — use the right vocabulary

Personal learnings:
- Building in public (GitHub) keeps you honest
- A live demo is worth more than 10 architectural diagrams
- The best way to understand a problem deeply is to build the solution yourself""",
        "metadata": {"source": "journal", "date": "2026-04-04"},
    },
    {
        "content": """Reading list and resources — ongoing, last updated April 2026.
Papers read:
- MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)
- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (2024)
- Lost in the Middle: How LLMs Use Long Contexts (Liu et al., 2023)

Books in progress:
- The Pragmatic Programmer (Hunt & Thomas) — chapter on orthogonality
- Designing Data-Intensive Applications (Kleppmann) — chapter on replication

Resources bookmarked:
- LangChain memory modules documentation
- ChromaDB filtering and metadata guide
- SQLite FTS5 official documentation
- Gemini API rate limits and pricing page

To read:
- Ebbinghaus forgetting curve original paper
- SM-2 spaced repetition algorithm specification
- Anthropic's Constitutional AI paper""",
        "metadata": {"source": "note", "date": "2026-04-06"},
    },
]


def ingest(entry: dict, idx: int) -> bool:
    try:
        r = httpx.post(
            f"{BASE}/api/ingest/text",
            json={"content": entry["content"], "metadata": entry.get("metadata", {})},
            timeout=60,
        )
        if r.status_code == 200:
            data = r.json()
            print(f"  [{idx+1:02d}] ✓  {data['chunks_created']} chunks, {data['entities_extracted']} entities")
            return True
        else:
            print(f"  [{idx+1:02d}] ✗  HTTP {r.status_code}: {r.text[:80]}")
            return False
    except Exception as e:
        print(f"  [{idx+1:02d}] ✗  {e}")
        return False


def main():
    print(f"Seeding {len(ENTRIES)} entries into {BASE}\n")

    # Health check first
    try:
        h = httpx.get(f"{BASE}/api/health", timeout=15).json()
        print(f"Current state: {h['events']} events, {h['memories']} memories, {h['entities']} entities\n")
    except Exception as e:
        print(f"Could not reach API: {e}\nMake sure the server is running.\n")
        return

    ok = 0
    for i, entry in enumerate(ENTRIES):
        success = ingest(entry, i)
        if success:
            ok += 1
        time.sleep(1.2)  # stay within Gemini free-tier rate limits

    print(f"\nDone — {ok}/{len(ENTRIES)} entries ingested successfully.")

    try:
        h = httpx.get(f"{BASE}/api/health", timeout=15).json()
        print(f"New state: {h['events']} events, {h['memories']} memories, {h['entities']} entities, {h['vectors']} vectors")
    except Exception:
        pass


if __name__ == "__main__":
    main()
