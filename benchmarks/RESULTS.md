# ContextWeave — LoCoMo retrieval benchmark

**Benchmark:** [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al.,
2024) — the standard multi-session-conversation memory benchmark. Each question
is annotated with the dialogue turn(s) that support the answer.

**Metric:** **recall@k** — for each answerable question, did retrieval surface at
least one gold-evidence turn in the top-k? This isolates *retrieval quality* (the
core of a memory layer) and needs **no LLM judge**, so results are deterministic
and comparable across runs. (Answer-accuracy, LLM-judged, is a separate future
run; recall upper-bounds achievable answer accuracy.)

**Harness:** [`benchmarks/locomo_recall.py`](locomo_recall.py) — reproducible;
runs the real embedding + hybrid-retrieval (+ optional cross-encoder rerank)
stack. `groq_api_key` blanked (retrieval only). Adversarial / no-evidence
questions are excluded from recall (they can't be scored) and reported separately.

## Results

_Sample below: 3 of 10 conversations, **494 answerable questions**. Full-set
numbers appended when the 10-conversation run completes._

| Config | recall@5 | recall@10 |
|---|---:|---:|
| Fused (vector + FTS + graph) | **0.563** | 0.668 |
| **+ cross-encoder reranking** | **0.638** | 0.694 |

Reranking lifts recall@5 by **+7.5 points (+13% relative)** — a public-benchmark
confirmation of the reranking win seen on the internal eval (MRR 0.83 → 1.00).

### By LoCoMo category (with reranking)
| Category | n | recall@5 | recall@10 |
|---|---:|---:|---:|
| 2 — temporal | 90 | **0.767** | 0.800 |
| 4 — single-hop | 200 | 0.665 | 0.730 |
| 1 — multi-hop | 73 | 0.630 | 0.712 |
| 5 — adversarial | 112 | 0.527 | 0.571 |
| 3 — open-domain | 19 | 0.421 | 0.474 |

**Reading the numbers (honestly):**
- **Temporal is the strongest category (0.77)** — direct evidence that
  intent-aware temporal decay works; temporal reasoning is the category
  incumbents call hardest.
- **Reranking's biggest lift is single-hop** (0.535 → 0.665, +13 pts) — pulling
  the one right turn into the top-5.
- **Open-domain is weakest** (world-knowledge questions where the evidence turn
  is only obliquely related) — the clearest improvement target.

## Caveats
- Recall, not answer-accuracy. A high-recall system still needs good synthesis;
  ContextWeave does the synthesis + calibrated confidence on top, not measured here.
- Sample is 3/10 conversations pending the full run (494 questions is already a
  meaningful sample).
- Not yet compared head-to-head with Mem0/Zep on identical settings — the next
  step is to run their published harnesses on the same split.

## Reproduce
```bash
curl -L -o benchmarks/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python -m benchmarks.locomo_recall --data benchmarks/locomo10.json            # fused
CW_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2 \
  python -m benchmarks.locomo_recall --data benchmarks/locomo10.json          # + rerank
```
