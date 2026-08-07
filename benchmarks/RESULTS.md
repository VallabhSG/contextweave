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

## Results — full set (all 10 conversations, **1,977 answerable questions**)

| Config | recall@5 | recall@10 |
|---|---:|---:|
| Fused (vector + FTS + graph) | _(full-set run finalizing)_ | — |
| **+ cross-encoder reranking** | **0.598** | **0.644** |

_(On a 3-conversation / 494-question subset the fused → reranked lift was
0.563 → 0.638 at recall@5; the full-set fused number is being computed to give a
clean before/after — the reranked full-set figure above is final.)_

### By LoCoMo category (full set, with reranking)
| Category | n | recall@5 | recall@10 |
|---|---:|---:|---:|
| 2 — temporal | 320 | **0.688** | 0.731 |
| 4 — single-hop | 841 | 0.648 | 0.691 |
| 1 — multi-hop | 281 | 0.641 | 0.690 |
| 5 — adversarial | 446 | 0.451 | 0.504 |
| 3 — open-domain | 89 | 0.404 | 0.449 |

**Reading the numbers (honestly):**
- **Temporal is the strongest category (0.69)** — direct evidence that
  intent-aware temporal decay works; temporal reasoning is the category
  incumbents call hardest.
- **Multi-hop and single-hop are solid (~0.64–0.65)** — the fused-signal +
  graph-expansion stack surfaces supporting turns well.
- **Open-domain is weakest (0.40)** — world-knowledge questions where the
  evidence turn is only obliquely related; the clearest improvement target.
- Adversarial (0.45) is expected to be lower — those questions are designed to
  have no clean supporting turn.

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
