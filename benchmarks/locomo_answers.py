"""LoCoMo answer-accuracy benchmark for ContextWeave.

The comparable, end-to-end metric (what memory vendors headline): retrieve →
an LLM answers from the retrieved context → an LLM judge scores the answer
against the reference. Unlike raw-turn recall, this credits the system for
answering correctly from related/aggregated context.

Two things this harness gets right that a naive script does not:
- **Per-session dates** are attached to every memory, so temporal questions
  ("yesterday" → an absolute date) are answerable.
- The judge is asked for *semantic equivalence*, not string match.

The generator + judge LLM is any OpenAI-compatible endpoint, configured via env
(no secrets in the repo):
    LOCOMO_LLM_URL   e.g. https://api.cerebras.ai/v1   (or Groq, OpenAI, …)
    LOCOMO_LLM_KEY   your key
    LOCOMO_LLM_MODEL e.g. gemma-4-31b

Usage:
    LOCOMO_LLM_URL=... LOCOMO_LLM_KEY=... LOCOMO_LLM_MODEL=... \
      CW_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2 \
      python -m benchmarks.locomo_answers --data benchmarks/locomo10.json --max-questions 100

Numbers from this harness are only as strong as the judge model — report the
model used. For a citable headline, use a strong judge (or the official LoCoMo
eval) over the full set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from uuid import uuid4

import httpx

from contextweave.api.pipeline import process_events
from contextweave.config import settings
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.schemas import SourceType
from contextweave.timeutils import utcnow
from contextweave.workspaces import manager


def _llm(messages: list[dict], max_tokens: int = 60) -> str:
    url = os.environ["LOCOMO_LLM_URL"].rstrip("/") + "/chat/completions"
    key = os.environ["LOCOMO_LLM_KEY"]
    model = os.environ["LOCOMO_LLM_MODEL"]
    for _ in range(6):
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
            timeout=45,
        )
        if r.status_code == 429:  # rate limited — back off
            time.sleep(15)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    return ""


def _ingest_with_dates(ws, conversation: dict) -> None:
    """Ingest each turn tagged with its real session date (temporal resolution)."""
    adapter = TextAdapter()
    session_keys = sorted(
        (
            k
            for k in conversation
            if k.startswith("session_") and "date" not in k and "summary" not in k
        ),
        key=lambda s: int(s.split("_")[1]),
    )
    for key in session_keys:
        date = conversation.get(f"{key}_date_time", "")
        for turn in conversation[key]:
            if turn.get("text"):
                content = f"[{date}] {turn.get('speaker', '')}: {turn['text']}"
                process_events(
                    ws,
                    adapter.ingest_text(
                        content, timestamp=utcnow(), source=SourceType.CONVERSATION
                    ),
                )


def _answer(question: str, contexts: list[str]) -> str:
    ctx = "\n".join(f"- {c}" for c in contexts)
    return _llm(
        [
            {
                "role": "user",
                "content": (
                    f"Dated conversation memories:\n{ctx}\n\nQuestion: {question}\n"
                    "Answer concisely using only the memories; resolve relative times "
                    "('yesterday', 'last week') using the bracketed dates. If the answer "
                    "is not present, say 'No information'."
                ),
            }
        ]
    )


def _judge(question: str, reference: str, candidate: str) -> bool:
    v = _llm(
        [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}\n"
                    "Is the candidate consistent with the reference (correct, even if phrased "
                    "differently or more/less specific)? Reply only yes or no."
                ),
            }
        ],
        max_tokens=4,
    )
    return v.lower().startswith("y")


def run(data_path: str, limit: int | None, max_questions: int | None) -> None:
    for var in ("LOCOMO_LLM_URL", "LOCOMO_LLM_KEY", "LOCOMO_LLM_MODEL"):
        if not os.environ.get(var):
            raise SystemExit(f"Set {var} (see module docstring).")

    samples = json.load(open(data_path, encoding="utf-8"))
    if limit:
        samples = samples[:limit]

    import tempfile

    base = tempfile.mkdtemp(prefix="locomo_ans_")
    settings.sqlite_db_path = base + "/d.db"
    settings.chroma_persist_dir = base + "/c"
    settings.data_dir = base + "/data"
    settings.groq_api_key = ""
    manager.reset()

    correct = n = 0
    per_cat = defaultdict(lambda: {"n": 0, "ok": 0})

    for si, sample in enumerate(samples, 1):
        ws = manager.get("la_" + uuid4().hex[:8])
        _ingest_with_dates(ws, sample["conversation"])
        asked = 0
        for qa in sample["qa"]:
            if not (qa.get("evidence")):
                continue
            if max_questions and n >= max_questions:
                break
            results = ws.retriever.retrieve(qa["question"], top_k=10)
            pred = _answer(qa["question"], [r.content for r in results])
            if not pred:
                continue
            time.sleep(1)
            ok = _judge(qa["question"], str(qa["answer"]), pred)
            time.sleep(1)
            n += 1
            asked += 1
            correct += ok
            per_cat[qa.get("category")]["n"] += 1
            per_cat[qa.get("category")]["ok"] += ok
        print(
            f"  conversation {si}/{len(samples)}: {asked} questions, running acc={correct / max(n, 1):.3f}"
        )
        if max_questions and n >= max_questions:
            break

    print("\n=== LoCoMo answer accuracy ===")
    print(
        f"questions: {n} | judge/generator: {os.environ['LOCOMO_LLM_MODEL']} | "
        f"reranker: {settings.rerank_model or '(off)'}"
    )
    print(f"answer accuracy: {correct / max(n, 1):.3f}")
    for cat in sorted(per_cat):
        d = per_cat[cat]
        if d["n"]:
            print(f"  cat {cat}: n={d['n']:>3}  acc={d['ok'] / d['n']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/locomo10.json")
    ap.add_argument("--limit", type=int, default=None, help="max conversations")
    ap.add_argument(
        "--max-questions", type=int, default=None, help="cap total questions (rate limits)"
    )
    args = ap.parse_args()
    run(args.data, args.limit, args.max_questions)
