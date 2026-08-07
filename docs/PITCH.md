# ContextWeave — Pre-seed pitch (one-pager + narrative)

> Single source of truth for all fundraising materials. Everything below is
> either **verifiable today** or explicitly marked **[PLACEHOLDER — confirm]**.
> No traction is claimed that doesn't exist. Keep every other doc consistent
> with this file.

---

## Source of truth (canonical facts)

**Verifiable now**
- Live, deployed demo (Hugging Face Space) + interactive API docs.
- Retrieval quality is **measured**, including on a **public benchmark**:
  - Internal eval: cross-encoder reranking lifts **MRR 0.83 → 1.00**.
  - **LoCoMo** (standard multi-session memory benchmark, all **1,977 questions**),
    reproducible harness in `benchmarks/`: reranking measurably lifts retrieval
    **recall@5 0.52 → 0.60 (+15%)**, and **temporal — the category incumbents call
    hardest — is our strongest (0.69)**, validating intent-aware decay. Recall is
    a strict lower bound; on the comparable metric (LLM-judged answer accuracy)
    an early corrected run scores **≈ 0.68 (indicative, n=25)** — above the recall,
    as expected. _(A rigorous full-set answer-accuracy run with a validated judge
    is the next benchmark milestone.)_
- Architecture: hybrid retrieval (vector + full-text + knowledge graph),
  intent-aware temporal decay, token-budgeted context assembly with
  MMR-dedup + query-aware compression, relevance-calibrated confidence with
  explicit hedging on weak context.
- **Local-first / private:** embeddings and reranking run on-box; verified to
  run **100% offline** (local Ollama LLM) with no data leaving the machine.
  Cloud LLMs (Groq primary, Cerebras failover) optional.
- Swappable storage: SQLite (local) or Postgres + pgvector (cloud/Supabase).
- Engineering rigor: **192 automated tests** (incl. real Postgres integration),
  LLM provider failover, live deployment with security headers + rate limiting.

**Not true yet (state honestly)**
- No users, no revenue, no design partners. It is a **working demo**, not a
  company with traction. The raise is to earn the first metrics.

**To finalize — the only inputs I can't produce (≈2 minutes)**
1. **Founder name + one-line bio** (background that makes "this person can build the
   hard part" credible — e.g. prior work, education, shipped projects).
2. **Raise size + instrument** — a **$650K SAFE** default is used below; change it
   to your real target.
3. **Contact email.** (Demo URL and repo are already filled — they're verifiable.)

_Verifiable, already filled:_ Demo `vallllllllll-contextweave.hf.space` · Code
`github.com/VallabhSG/contextweave`.

---

## One-pager

**ContextWeave — the private memory & context layer for AI.**
_Your context, retrieved and ranked to fit the model's window — on your own
infrastructure._

**Why now.** Bigger context windows don't fix memory — they make it worse.
Chroma (2026) showed **all 18 frontier models degrade as input grows** ("context
rot"); the classic *lost-in-the-middle* effect drops accuracy ~70% → ~55% once
~20 docs fill the window. Coatue calls memory *"the new bottleneck"*; NVIDIA is
shipping a tiered agent-memory stack; clouds are now **metering memory**. The fix
isn't a bigger window — it's a layer that **selects, ranks, and budgets** what
reaches the model. Market: **~$6B (2025) → ~$28B (2030), ~35% CAGR** (Mordor;
estimates vary — see MARKET_RESEARCH.md).

**What we do.** ContextWeave ingests a person's or team's context (notes,
conversations, calendar, browsing) and answers questions across it — using a
retrieval stack engineered specifically against context rot: hybrid signals,
temporal decay, a cross-encoder re-ranker, and a **token budget** that packs the
few most-relevant, de-duplicated, compressed passages instead of dumping history.
It reports **honest confidence** and hedges when context is thin.

**Wedge (why us, not Mem0).** The funded leaders (Mem0 $24M, Zep, Letta) are
**cloud-SaaS**. ContextWeave is **local-first and self-hostable** — the context
never has to leave your box — and every quality claim is **measured**, in a
category the buyers themselves say lacks comparable metrics. Privacy/edge and
"which layer is actually more accurate per token" are *named open problems*;
they're our starting point.

**Proof.** Live demo; measured retrieval — **MRR 0.83 → 1.00** internally and
on the public **LoCoMo** benchmark, reranking lifts retrieval **recall@5 0.52 →
0.60** (all 1,977 questions), strongest on *temporal* — the hardest category;
runs **fully offline**; ships an **MCP
server** (private agent memory in one config block); 192 tests. Built by one
engineer — the same hard core (hybrid retrieval, intent-aware decay, reranking,
budgeted assembly, calibrated confidence) that venture-funded teams are building.
<!-- test count: 192 (182 unit/integration + 10 Postgres) as of 2026-08-06 -->

**The ask.** Raising a **[PLACEHOLDER ~$650K] pre-seed (SAFE)** to (1) publish
standard-benchmark numbers (LoCoMo/LongMemEval) vs. incumbents, (2) ship a
3-lines-of-code **SDK/MCP** integration, and (3) land the first **10 design
partners** in privacy-sensitive teams.

**Contact.** [Your name] · [email] · **Demo:** vallllllllll-contextweave.hf.space ·
**Code:** github.com/VallabhSG/contextweave

---

## Pitch narrative (deck flow — talking points per slide)

1. **Company + wedge.** ContextWeave: the *private, self-hostable* memory &
   context layer for AI. One line: "your context, ranked to fit the window,
   on your infrastructure."
2. **Problem.** Agents and personal AIs forget; the naive fix (bigger window /
   stuff everything in) *measurably* fails — context rot, lost-in-the-middle,
   and 20–30× token blow-up per agent session. Memory is now the bottleneck
   (Coatue, NVIDIA, metered by clouds).
3. **Solution.** A dedicated retrieval+assembly layer engineered against rot:
   hybrid signals → re-rank → **budget & compress** → honest confidence.
4. **Product / demo.** Live walkthrough: ingest → ask → see cited, ranked answer
   + confidence; show the same query with reranking on/off (MRR 0.83 → 1.00);
   show it running fully offline (local model) — nothing leaves the box.
5. **Market.** ~$6B → ~$28B AI-memory (est., ~35% CAGR); agentic AI ~$52B by
   2030; 40% of enterprise apps agentic by 2026 (Gartner). Flag estimate spread.
6. **Business model.** Open-core: free/self-hosted OSS layer → paid cloud +
   team features + support/compliance for regulated deployments (mirrors Mem0/
   Zep open-core, but privacy-first). [PLACEHOLDER — pricing to validate.]
7. **Traction.** Honest: pre-traction; working measured demo. Show the eval
   numbers and the deployment as *evidence of build capability*, not adoption.
8. **Team.** [PLACEHOLDER] Solo technical founder who built the full stack +
   live deployment solo. Named hires/advisors: [to add]. Address "why this
   founder can win a crowded category": depth already demonstrated.
9. **Competition / differentiation.** Table: Mem0/Zep/Letta = cloud-SaaS,
   framework-integration wedge, published benchmarks. ContextWeave = local-first
   + measured quality + context-engineering depth (temporal decay, calibrated
   abstention) that maps to incumbents' *open problems*. Consumer (Limitless→
   Meta, Thine $9.36M) is a different, harder lane — we don't fight there head-on.
10. **Ask.** [PLACEHOLDER ~$650K] pre-seed SAFE, ~18 months runway.
11. **Use of funds / milestones** (see table).
12. **Appendix.** Architecture diagram, eval methodology, roadmap, sources.

---

## Use of funds — proposed (sums to 100%) [PLACEHOLDER amounts]

Raise: **$650K** pre-seed · ~18 months runway.

| Bucket | % | $ | Buys |
|---|---:|---:|---|
| Engineering & product (founder + 2) | 55% | $357.5K | Core layer → SDK/MCP, benchmarks, self-host packaging |
| Benchmarks + design partners + DevRel | 20% | $130K | Public LoCoMo/LongMemEval, 10 design partners, docs |
| Infra / compute / tooling | 10% | $65K | Eval compute, hosting, models |
| GTM / community | 10% | $65K | OSS launch, content, early pipeline |
| Legal / buffer | 5% | $32.5K | Entity, contracts, contingency |
| **Total** | **100%** | **$650K** | |

### Milestones this buys (6 / 12 / 18 mo)
- **6 mo:** Published benchmark numbers vs. Mem0/Zep; SDK/MCP alpha; OSS repo public with the eval harness.
- **12 mo:** 10 design partners (privacy-sensitive teams / self-hosters); SDK GA; first paid pilots.
- **18 mo:** Early revenue from cloud/team tier; seed-ready metrics (usage, retention, pilot conversion).

---

## Consistency checklist (before sending anything)
- [ ] Every number here matches deck, model, and emails.
- [ ] No traction claimed beyond "working measured demo."
- [ ] Market numbers labelled as estimates; source cited.
- [ ] Use-of-funds sums to 100% / $650K.
- [ ] Founder bio, raise size, and pricing placeholders filled with real values.
- [ ] Reviewed against `docs/MARKET_RESEARCH.md` (no contradictions).
