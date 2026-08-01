"""Retrieval quality evaluation harness.

Unlike the property tests, this measures *systemic* retrieval quality across a
small labelled query set — hit@k and MRR — and asserts a baseline. It exists so
that tuning (fusion weights, BM25 k, budget, decay) can be judged by a number,
and so a future change that quietly degrades retrieval fails here instead of in
production. Runs the real embedding model end-to-end.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from contextweave.api.pipeline import process_events
from contextweave.config import settings
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.schemas import SourceType
from contextweave.timeutils import utcnow
from contextweave.workspaces import manager

# A shared, diverse personal corpus. Each eval case queries it and names the
# substring that a correct top-k result must contain.
CORPUS = [
    "Quarterly budget review with the finance team: we agreed to cut cloud spend by 20%.",
    "My sourdough starter finally doubled overnight after switching to rye flour.",
    "Long run went well — 18 kilometres at a steady pace before sunrise.",
    "Spoke with Dana Whitfield about the funding options for Project Falcon.",
    "The prototype for Project Falcon needs a dedicated hardware fund.",
    "Sketch for the reading tracker app: a minimal list with a streak counter.",
    "Dentist appointment moved to next Thursday afternoon.",
    "Started reading a book on the history of cartography and old maps.",
    "Watered the tomato plants on the balcony and repotted the basil.",
    "Team retro: we decided to timebox standups to ten minutes.",
    "Booked flights to Lisbon for the October conference; still need to sort the hotel.",
    "The car needs an oil change and the left brake pad has started squealing.",
    "Finished the pottery class tonight — glazed two bowls and a chipped mug.",
    "Moving part of the emergency fund into a high-yield savings account.",
    "Debugging the payment webhook: Stripe retries were creating duplicate orders.",
    "Twelve-day streak learning Spanish on Duolingo, mostly verbs this week.",
    "Doctor flagged slightly high cholesterol and suggested less red meat.",
    "Repainted the bedroom a warm off-white over the weekend.",
]

# (query, expected substring in a correct result) — lowercased comparison.
EVAL_CASES = [
    ("what is happening with company finances and spending?", "budget"),
    ("my sourdough baking progress", "sourdough"),
    ("morning running and exercise", "kilometres"),
    ("funding for Project Falcon", "falcon"),
    ("the reading tracker app idea", "reading tracker"),
    ("what did I discuss with Dana Whitfield?", "dana whitfield"),
    ("travel plans for the conference", "lisbon"),
    ("what's wrong with my car?", "brake"),
    ("where should I move my emergency savings?", "high-yield"),
    ("the bug causing duplicate orders", "webhook"),
    ("how is my language learning going?", "spanish"),
    ("what did the doctor say about my health?", "cholesterol"),
]


@pytest.fixture(scope="module")
def eval_ws(tmp_path_factory):
    """A workspace with the corpus ingested once, shared across the eval tests."""
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("reval")
    mp.setattr(settings, "sqlite_db_path", str(base / "demo.db"))
    mp.setattr(settings, "chroma_persist_dir", str(base / "chroma"))
    mp.setattr(settings, "data_dir", str(base / "data"))
    mp.setattr(settings, "groq_api_key", "")
    manager.reset()

    ws = manager.get("reval_" + uuid4().hex[:10])
    for text in CORPUS:
        process_events(
            ws, TextAdapter().ingest_text(text, timestamp=utcnow(), source=SourceType.NOTE)
        )

    yield ws
    manager.reset()
    mp.undo()


def _first_hit_ranks(ws) -> list[tuple[str, str, int | None]]:
    """For each case, the 0-indexed rank of the first result containing the target."""
    ranks = []
    for query, expect in EVAL_CASES:
        results = ws.retriever.retrieve(query, top_k=5)
        rank = next((i for i, r in enumerate(results) if expect in r.content.lower()), None)
        ranks.append((query, expect, rank))
    return ranks


def _mrr(ranks) -> float:
    return sum(1.0 / (rank + 1) for _, _, rank in ranks if rank is not None) / len(ranks)


def test_hit_at_5_meets_baseline(eval_ws):
    ranks = _first_hit_ranks(eval_ws)
    hit_at_5 = sum(1 for _, _, r in ranks if r is not None) / len(ranks)
    misses = [f"{q!r}→{e!r}" for q, e, r in ranks if r is None]
    assert hit_at_5 >= 0.9, f"hit@5={hit_at_5:.2f}; missed: {misses}"


def test_mrr_meets_baseline(eval_ws):
    # Fused-only (no reranking). Baseline history on the small set: 0.6 → 0.7
    # (FTS OR) → 0.8 (Porter stemming). The larger, more diverse set is harder
    # and less overfit — fused MRR ≈ 0.79, so the floor is 0.75.
    mrr = _mrr(_first_hit_ranks(eval_ws))
    assert mrr >= 0.75, f"fused MRR={mrr:.2f}"


def test_reranking_improves_mrr(eval_ws):
    """The production config (cross-encoder reranking) must beat fused and clear 0.9."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401
    except Exception:
        pytest.skip("fastembed cross-encoder reranker unavailable")

    from contextweave.processing.reranker import CrossEncoderReranker

    fused = _mrr(_first_hit_ranks(eval_ws))
    eval_ws.retriever.reranker = CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    try:
        reranked = _mrr(_first_hit_ranks(eval_ws))
    finally:
        eval_ws.retriever.reranker = None

    assert reranked >= fused, (
        f"reranking should not hurt: fused={fused:.2f} reranked={reranked:.2f}"
    )
    assert reranked >= 0.9, f"reranked MRR={reranked:.2f}"
