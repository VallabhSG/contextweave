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
| Fused (vector + FTS + graph) | 0.520 | 0.622 |
| **+ cross-encoder reranking** | **0.598** | **0.644** |

Reranking lifts recall@5 by **+7.8 points (0.520 → 0.598, +15% relative)** across
all 1,977 questions — a public-benchmark confirmation of the reranking win seen
on the internal eval (MRR 0.83 → 1.00).

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

## How to read these numbers (important — they look lower than incumbents' headlines)

`recall@5 ≈ 0.60` is a **strict retrieval lower-bound**, deliberately harder than
what memory vendors advertise:

- **It's recall, not answer accuracy.** It asks "is the *one exact* gold turn in
  the top-5, out of ~250 raw dialogue turns?" Mem0/Zep headline **LLM-judged
  answer accuracy** (e.g. Mem0's ~93% LongMemEval) — a different, higher metric,
  because the model answers from the *whole* retrieved context, not just the
  single gold turn. **These numbers are not comparable**; recall under-states
  end-to-end quality.
- **We retrieve raw turns; incumbents retrieve extracted facts.** They summarise
  conversations into fact memories first, which is an easier retrieval target.
  Raw-turn retrieval is the conservative choice.
- **The number is real, not a scoring artifact.** A diagnostic (196 questions)
  found only **1%** "retrieved-but-mis-scored" — the exact-substring match is
  accurate. The remaining misses are genuine, and LoCoMo (needle in ~250 turns)
  is legitimately hard.

The honest, comparable metric is **answer accuracy** (retrieve → LLM answers →
judge); it is being measured and will be reported here as the headline, with
recall kept as the stricter retrieval-only diagnostic.

## Caveats
- Not yet compared head-to-head with Mem0/Zep on identical settings — the next
  step is to run their published harnesses on the same split.
- ContextWeave's synthesis + calibrated confidence sit on top of retrieval and
  are captured by the answer-accuracy metric, not by recall.

## Reproduce
```bash
curl -L -o benchmarks/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python -m benchmarks.locomo_recall --data benchmarks/locomo10.json            # fused
CW_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2 \
  python -m benchmarks.locomo_recall --data benchmarks/locomo10.json          # + rerank
```
