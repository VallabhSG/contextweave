# Context Engineering in ContextWeave

> Living document. Each autonomous improvement loop appends to the changelog and
> checks items off the backlog. The goal: make ContextWeave the best possible
> _context engine_ — not just storing memories, but assembling exactly the right
> ones into a language model's finite window.

## What "the context problem" is here

ContextWeave is retrieval-augmented reasoning over a person's own memory stream.
The hard part is not storage and not even retrieval — it is **context assembly**:
given a query and a ranked pile of candidate memories, decide *which* of them,
and *how much* of each, should occupy the model's limited context window so the
answer is accurate, grounded, and honestly qualified.

A naive RAG system does `retrieve top-K → stuff full text into the prompt`. That
fails for personal memory in specific, measurable ways:

| Failure | Symptom | Fix |
|---|---|---|
| No token budget | Unbounded prompt; window blown or tokens wasted on tail memories | Budgeted packing (`reasoning/context_budget.py`) |
| Redundancy | The same recurring thought, captured 5×, crowds out diverse context | MMR-style redundancy suppression |
| Count-based confidence | 8 irrelevant hits report `confidence = 1.0` | Relevance-calibrated confidence |
| Vague grounding | Model can't tell strong context from filler | Rank-ordered, labelled context blocks |

## The pipeline (as of this writing)

```
POST /api/query
  └─ _run_query (api/routes.py)
       ├─ ReasoningEngine.expand_query      → extra search terms
       ├─ HybridRetriever.retrieve          → ranked QueryResult[]  (vector+FTS+graph, decayed)
       ├─ ContextBudgeter.assemble          → packed subset under a token budget  ← context assembly
       └─ ReasoningEngine.reason            → Groq LLM answer + calibrated confidence
```

The retriever's job is **recall + ranking**. The budgeter's job is **selection
under a budget**. Keeping these separate matters: `test_memory_quality.py`
asserts the retriever still *returns* near-duplicate copies (temporal-decay
test), so de-duplication must live in the assembly layer, never in the retriever.

## Backlog (ranked by impact on answer quality)

- [x] **Token-budgeted context assembly** — pack memories by score until a token
      budget is spent; truncate an over-long memory rather than dropping it whole.
      → `reasoning/context_budget.py`
- [x] **Redundancy suppression (MMR-lite)** — skip a candidate that is lexically
      near-identical to something already packed. Diversity beats repetition.
- [x] **Relevance-calibrated confidence** — derive confidence from packed
      relevance scores and breadth, not from raw result count.
- [x] **Intent-aware temporal decay** — a `temporal` query ("how has X evolved
      over time?") now relaxes the recency half-life so old memories, which *are*
      the answer, are not decayed into oblivion. Query-intent detection is shared
      (`reasoning/query_intent.py`) so retriever and reasoning agree.
- [x] **Graph expansion from FTS matches too** — graph traversal now seeds from
      keyword (FTS) hits as well as vector hits, so connections surface even when
      a memory matches by keyword and, crucially, when vector search is
      unavailable (embedding outage) and returns nothing.
- [x] **BM25 score normalization** — replaced the arbitrary `abs(rank)/10` clip
      with a smooth saturating curve (`strength / (strength + k)`), so FTS
      relevance composes predictably with the vector score and strong matches no
      longer collapse to an identical clipped 1.0.
- [x] **Pre-decay relevance channel** — `QueryResult.relevance` now carries the
      un-decayed fused relevance, and confidence reads *that* rather than the
      decay-tuned `score`. Fixes an incoherence introduced by iterations 1+2:
      the same query reported very different confidence depending only on whether
      it was classified `temporal` (which relaxes decay and so inflated `score`).
      Ranking still uses `score`; confidence now judges match quality alone.
- [x] **Query-adaptive fusion weights** — connection-oriented intents now shift
      fusion weight toward the graph: `cross_reference` 0.4/0.2/0.4 and `patterns`
      0.45/0.2/0.35, while every other intent keeps the balanced 0.5/0.3/0.2
      (weights sum to 1.0 so scores stay comparable). Only intents *explicitly*
      about connections deviate, avoiding arbitrary per-type tuning.
      (Note: the BM25 saturating normalization lowered FTS's *effective*
      contribution for strong keyword matches vs. the old hard clip — revisit `k`
      or move to set-relative FTS scaling if fusion needs further tuning.)
- [x] **Graph expansion priority by hop distance** — graph traversal now tracks
      each chunk's minimum hop distance (`get_connected_chunks_ranked`), and the
      retriever's 50-chunk cap keeps the *nearest* connections instead of
      alphabetically-first chunk IDs. 1-hop co-occurrences beat distant 2-hop links.
- [ ] **Backfill after truncation** — `ContextBudgeter.assemble` breaks after
      truncating the last-fitting memory; a shorter later candidate could still
      fill remaining budget. Packing efficiency, not correctness.
- [ ] **Real tokenizer (optional)** — the budgeter uses a chars/4 heuristic;
      allow injecting a true tokenizer for exact accounting without adding a
      hard dependency.
- [x] **Sentence-level context compression** — when the query is known, a memory
      longer than `context_max_tokens_per_memory` is compressed to its most
      query-relevant sentences (first/framing sentence always kept), so the
      budget holds more distinct, on-point context instead of one memory's filler.

## Design principles

1. **Separation of concerns** — recall (retriever) vs. selection (budgeter) vs.
   synthesis (reasoning). Each is independently testable.
2. **No new heavy dependencies** — this project runs 100% free/local. The token
   budgeter estimates tokens with a documented heuristic and accepts an injected
   counter, rather than pulling in a native tokenizer.
3. **Honest over impressive** — confidence should track reality. A confident
   wrong answer is worse than a hedged right one.
4. **Every change is measured** — behavior changes ship with tests that assert
   the property, not the plumbing.

## Changelog

### 2026-07-30 — Confidence-aware answer guidance
- Confidence (built in iteration 4) now *shapes the answer*, not just the
  response metadata. Below `context_low_confidence_threshold` (0.35) the reasoning
  prompt gains an explicit hedge instruction ("answer only what the context
  supports; flag what's missing; don't speculate"), and the no-LLM fallback
  appends a caution note. Serves the product's "be honest when context is
  insufficient" pillar.
- Extracted `ReasoningEngine._build_prompt` for testability.
- Tests: `tests/test_reasoning_engine.py` (LLM-free). Full suite 161 passed.

### 2026-07-30 — Query-adaptive fusion weights
- `fusion_weights(intent)` returns (vector, fts, graph) weights per query intent;
  `cross_reference` (0.4/0.2/0.4) and `patterns` (0.45/0.2/0.35) lean on the
  graph, all others keep 0.5/0.3/0.2. Weights sum to 1.0 so scores stay
  comparable, and only connection-oriented intents deviate.
- `HybridRetriever` fuses with the intent's weights (intent already detected for
  decay). General-intent queries are unchanged.
- Tests: `TestFusionWeights` in `tests/test_hybrid_retriever.py`. Full suite 157.

### 2026-07-30 — Hop-distance graph prioritization
- `KnowledgeGraph.get_neighbors_with_distance` / `get_connected_chunks_ranked`
  expose each connected chunk's minimum hop distance from the query entity.
- `HybridRetriever` uses the distance so the 50-chunk graph cap keeps the
  nearest connections (1-hop before 2-hop) rather than alphabetically-first IDs.
  `get_connected_chunks` now also returns nearest-first.
- Tests: `tests/test_knowledge_graph.py`. Full suite 153 passed.

### 2026-07-30 — Query-aware extractive compression
- `ContextBudgeter.assemble` now takes the `query`; a memory longer than
  `context_max_tokens_per_memory` (default 200) is compressed to its most
  query-relevant sentences before packing (first sentence always kept), so the
  budget holds more on-point context and less filler.
- Truncation is now query-aware too: the budget-boundary memory keeps its
  relevant sentences rather than just its opening.
- Behaviour is gated on a query — `assemble(results)` with no query is byte-for-
  byte unchanged, so all prior tests hold.
- Tests: `TestQueryAwareCompression` in `tests/test_context_budget.py`. Full
  suite 147 passed.

### 2026-07-30 — Confidence coherence (pre-decay relevance channel)
- Added `QueryResult.relevance` (pre-decay fused relevance, clamped to [0,1]).
- `ContextBudgeter._confidence` reads `relevance`, not `score`, so confidence no
  longer swings with temporal decay or query classification — it reflects how
  well context matches the query. (Found via review: temporal-classified queries
  had been reporting up to ~70× higher confidence for identical relevance.)
- Made graph-expansion chunk iteration deterministic (`sorted`) so the 50-chunk
  cap is stable across `PYTHONHASHSEED`.
- Tests: coherence case in `tests/test_context_budget.py`; `make_result` now sets
  `relevance`. Full suite 141 passed.

### 2026-07-30 — Retrieval fusion quality (graph seeding + BM25 normalization)
- `HybridRetriever` now seeds graph expansion from FTS matches too, not only
  vector hits — connections survive an embedding outage.
- Replaced the arbitrary FTS `abs(rank)/10` clip with a smooth saturating
  normalization (`_normalize_fts_rank`, k=5).
- Tests: `tests/test_hybrid_retriever.py` (normalization) and a vector-outage
  graph-expansion case in `tests/test_memory_quality.py`. Full suite 140 passed.

### 2026-07-30 — Intent-aware retrieval decay
- Extracted query-intent detection into `reasoning/query_intent.py`
  (`detect_query_type`), shared by the reasoning engine and the retriever.
- `ImportanceScorer.score` accepts a per-call `half_life_days` override.
- `HybridRetriever.retrieve` accepts `query_type` and, for temporal queries,
  relaxes decay via `context`-configurable `temporal_query_half_life_days`
  (default 365d) so history is preserved rather than buried.
- Threaded the API's explicit `query_type` through to retrieval.
- Tests: `tests/test_query_intent.py`, a scorer-override case, and a real-embedding
  decay case in `tests/test_memory_quality.py`. Full suite 135 passed.

### 2026-07-30 — Context assembly layer
- Added `reasoning/context_budget.py`: `ContextBudgeter` with token-budgeted
  greedy packing, MMR-lite redundancy suppression, and single-memory truncation.
- Wired it into `ReasoningEngine.reason` (both the LLM and no-key fallback paths).
- Replaced count-based confidence with relevance-calibrated confidence.
- Added config knobs `context_token_budget`, `context_redundancy_threshold`.
- Added `tests/test_context_budget.py`.
