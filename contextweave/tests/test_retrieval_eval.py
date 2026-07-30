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
]

# (query, expected substring in a correct result) — lowercased comparison.
EVAL_CASES = [
    ("what is happening with company finances and spending?", "budget"),
    ("my sourdough baking progress", "sourdough"),
    ("morning running and exercise", "kilometres"),
    ("funding for Project Falcon", "falcon"),
    ("the reading tracker app idea", "reading tracker"),
    ("what did I discuss with Dana Whitfield?", "dana whitfield"),
]


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """Ingest the corpus once, run every case, return per-case first-hit ranks."""
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

    ranks = []
    for query, expect in EVAL_CASES:
        results = ws.retriever.retrieve(query, top_k=5)
        rank = next(
            (i for i, r in enumerate(results) if expect in r.content.lower()),
            None,
        )
        ranks.append((query, expect, rank))

    yield ranks
    manager.reset()
    mp.undo()


def test_hit_at_5_meets_baseline(evaluated):
    hits = sum(1 for _, _, rank in evaluated if rank is not None)
    hit_at_5 = hits / len(evaluated)
    misses = [f"{q!r}→{e!r}" for q, e, rank in evaluated if rank is None]
    assert hit_at_5 >= 0.8, f"hit@5={hit_at_5:.2f}; missed: {misses}"


def test_mrr_meets_baseline(evaluated):
    reciprocal = sum(1.0 / (rank + 1) for _, _, rank in evaluated if rank is not None)
    mrr = reciprocal / len(evaluated)
    # Baseline raised from 0.6 to 0.7 after FTS OR-semantics lifted MRR to ~0.78.
    assert mrr >= 0.7, f"MRR={mrr:.2f}"
