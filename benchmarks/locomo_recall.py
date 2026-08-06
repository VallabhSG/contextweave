"""LoCoMo retrieval-recall benchmark for ContextWeave.

LoCoMo (Maharana et al., 2024) is the standard multi-session-conversation memory
benchmark. Each question is annotated with the dialogue turn(s) that support the
answer (`evidence` dia_ids). This harness measures **recall@k**: for each
answerable question, did ContextWeave's retrieval surface at least one gold
evidence turn in the top-k? That isolates retrieval quality — the core of a
memory layer — without an LLM judge, so the number is deterministic and
comparable across runs.

Usage:
    # download data once:
    curl -L -o benchmarks/locomo10.json \\
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    python -m benchmarks.locomo_recall --data benchmarks/locomo10.json --limit 2

Set CW_RERANK_MODEL to measure the reranked configuration.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from uuid import uuid4

from contextweave.api.pipeline import process_events
from contextweave.config import settings
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.schemas import SourceType
from contextweave.timeutils import utcnow
from contextweave.workspaces import manager

KS = (5, 10)


def _sessions(conversation: dict) -> list[tuple[str, list[dict]]]:
    keys = [
        k
        for k in conversation
        if k.startswith("session_") and "date" not in k and "summary" not in k
    ]
    return [(k, conversation[k]) for k in sorted(keys, key=lambda s: int(s.split("_")[1]))]


def _ingest_conversation(ws, conversation: dict) -> dict[str, str]:
    """Ingest each turn as a memory; return dia_id -> turn text."""
    adapter = TextAdapter()
    by_id: dict[str, str] = {}
    for _key, turns in _sessions(conversation):
        for turn in turns:
            dia_id, text = turn.get("dia_id"), turn.get("text", "")
            if not dia_id or not text:
                continue
            by_id[dia_id] = text
            content = f"{turn.get('speaker', '')}: {text}".strip()
            process_events(
                ws, adapter.ingest_text(content, timestamp=utcnow(), source=SourceType.CONVERSATION)
            )
    return by_id


def _recall_for_sample(sample: dict) -> tuple[dict, int, int]:
    ws = manager.get("locomo_" + uuid4().hex[:10])
    by_id = _ingest_conversation(ws, sample["conversation"])

    hits = {k: 0 for k in KS}
    per_cat = defaultdict(lambda: {"n": 0, **{k: 0 for k in KS}})
    answerable = 0
    skipped = 0

    for qa in sample["qa"]:
        evidence = qa.get("evidence") or []
        gold_texts = [by_id[e] for e in evidence if e in by_id]
        if not gold_texts:  # adversarial / no-evidence questions can't score recall
            skipped += 1
            continue
        answerable += 1
        cat = qa.get("category")
        per_cat[cat]["n"] += 1

        results = ws.retriever.retrieve(qa["question"], top_k=max(KS))
        contents = [r.content for r in results]
        for k in KS:
            topk = contents[:k]
            found = any(any(g in c for c in topk) for g in gold_texts)
            hits[k] += int(found)
            per_cat[cat][k] += int(found)

    return {"hits": hits, "answerable": answerable, "per_cat": dict(per_cat)}, answerable, skipped


def run(data_path: str, limit: int | None) -> None:
    samples = json.load(open(data_path, encoding="utf-8"))
    if limit:
        samples = samples[:limit]

    base = tempfile.mkdtemp()
    settings.sqlite_db_path = base + "/d.db"
    settings.chroma_persist_dir = base + "/c"
    settings.data_dir = base + "/data"
    settings.groq_api_key = ""
    manager.reset()

    total_hits = {k: 0 for k in KS}
    total_q = 0
    total_skipped = 0
    cat_totals = defaultdict(lambda: {"n": 0, **{k: 0 for k in KS}})

    for i, sample in enumerate(samples, 1):
        res, answerable, skipped = _recall_for_sample(sample)
        for k in KS:
            total_hits[k] += res["hits"][k]
        total_q += answerable
        total_skipped += skipped
        for cat, d in res["per_cat"].items():
            cat_totals[cat]["n"] += d["n"]
            for k in KS:
                cat_totals[cat][k] += d[k]
        r5 = res["hits"][5] / answerable if answerable else 0
        print(f"  sample {i}/{len(samples)}: recall@5={r5:.3f} ({answerable} answerable)")

    rerank = settings.rerank_model or "(off)"
    print("\n=== LoCoMo retrieval recall ===")
    print(
        f"conversations: {len(samples)} | answerable questions: {total_q} | "
        f"skipped (no evidence): {total_skipped} | reranker: {rerank}"
    )
    for k in KS:
        print(f"recall@{k}: {total_hits[k] / total_q:.3f}" if total_q else f"recall@{k}: n/a")
    print("by category (LoCoMo 1=multi-hop 2=temporal 3=open-domain 4=single-hop 5=adversarial):")
    for cat in sorted(cat_totals):
        d = cat_totals[cat]
        if d["n"]:
            print(
                f"  cat {cat}: n={d['n']:>4}  recall@5={d[5] / d['n']:.3f}  recall@10={d[10] / d['n']:.3f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/locomo10.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.data, args.limit)
