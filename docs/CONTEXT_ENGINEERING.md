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
- [ ] **Graph expansion from FTS matches too** — `HybridRetriever` seeds graph
      traversal only from vector hits; keyword-only matches never expand their
      entities. Connect more dots.
- [ ] **BM25 score normalization** — `abs(fts_rank)/10` is an arbitrary clip.
      Normalize FTS relevance against the candidate set instead.
- [ ] **Query-adaptive fusion weights** — decay is now intent-aware (above), but
      the vector/FTS/graph *fusion* weights are still fixed 0.5/0.3/0.2 for every
      query type; a `cross_reference` query should lean harder on the graph.
- [ ] **Pre-decay relevance channel** — expose the un-decayed relevance on
      `QueryResult` so confidence can separate "old" from "irrelevant".
- [ ] **Real tokenizer (optional)** — the budgeter uses a chars/4 heuristic;
      allow injecting a true tokenizer for exact accounting without adding a
      hard dependency.
- [ ] **Sentence-level context compression** — when a memory is long but only
      one passage is on-topic, pack the passage, not the whole memory.

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
