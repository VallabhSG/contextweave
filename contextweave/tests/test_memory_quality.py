"""Memory-quality verification with REAL embeddings.

Unlike the API tests (which fake vectors for speed), this suite runs the
actual fastembed model end-to-end and asserts the memory system's core
promises: semantic retrieval, keyword recall, temporal decay, the
access-frequency boost, graph expansion, and filtering.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from contextweave.api.pipeline import process_events
from contextweave.config import settings
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.processing.importance_scorer import ImportanceScorer
from contextweave.schemas import SourceType
from contextweave.timeutils import utcnow
from contextweave.workspaces import manager


@pytest.fixture(scope="module")
def real_env(tmp_path_factory):
    """Module-scoped env so the embedding model loads once for the suite."""
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("memq")
    mp.setattr(settings, "sqlite_db_path", str(base / "demo.db"))
    mp.setattr(settings, "chroma_persist_dir", str(base / "chroma"))
    mp.setattr(settings, "data_dir", str(base / "data"))
    mp.setattr(settings, "groq_api_key", "")
    manager.reset()
    yield
    manager.reset()
    mp.undo()


@pytest.fixture
def ws(real_env):
    """Fresh isolated workspace per test, sharing the loaded model."""
    return manager.get("memq_" + uuid4().hex[:10])


def ingest(ws, content: str, days_ago: int = 0, source: SourceType = SourceType.NOTE):
    events = TextAdapter().ingest_text(
        content, timestamp=utcnow() - timedelta(days=days_ago), source=source
    )
    process_events(ws, events)


class TestSemanticRetrieval:
    def test_paraphrase_query_finds_right_memory(self, ws):
        ingest(
            ws,
            "Quarterly budget review with the finance team is scheduled for Tuesday. "
            "We agreed to cut cloud infrastructure spend by twenty percent.",
        )
        ingest(ws, "My sourdough starter finally doubled overnight after switching to rye flour.")
        ingest(ws, "Long run went well, 18 kilometers at a steady pace before sunrise.")

        results = ws.retriever.retrieve("what is happening with company finances and spending?")

        assert results, "retrieval returned nothing"
        assert "budget" in results[0].content.lower(), (
            f"expected the budget memory first, got: {results[0].content[:80]}"
        )

    def test_keyword_recall_via_fts(self, ws):
        ingest(ws, "My sourdough starter finally doubled overnight after switching to rye flour.")
        ingest(ws, "Reviewed the quarterly numbers with the finance team today.")

        results = ws.retriever.retrieve("sourdough")

        assert results
        assert "sourdough" in results[0].content.lower()


class TestTemporalDecay:
    def test_recent_memory_outranks_old_duplicate(self, ws):
        text = "Recurring thought: build a personal knowledge garden to collect what I learn."
        ingest(ws, text, days_ago=120)
        ingest(ws, text, days_ago=0)

        results = ws.retriever.retrieve("personal knowledge garden idea")

        assert len(results) >= 2, "both copies should be retrieved"
        newest_first = results[0].timestamp > results[1].timestamp
        assert newest_first, "temporal decay should rank the recent copy first"
        assert results[0].score > results[1].score

    def test_decay_math_half_life(self):
        scorer = ImportanceScorer(half_life_days=30.0)
        now = utcnow()
        fresh = scorer.score(0.6, now, now=now)
        month = scorer.score(0.6, now - timedelta(days=30), now=now)
        quarter = scorer.score(0.6, now - timedelta(days=90), now=now)

        assert fresh == pytest.approx(0.6, abs=0.01)
        assert month == pytest.approx(0.3, abs=0.02)
        assert quarter == pytest.approx(0.075, abs=0.02)


class TestIntentAwareDecay:
    def test_temporal_intent_relaxes_decay_for_old_memory(self, ws):
        # A 200-day-old reflection is heavily decayed under the default 30-day
        # half-life — but a temporal query ("how has this evolved?") wants exactly
        # this history surfaced, so intent-aware retrieval relaxes the decay.
        ingest(
            ws,
            "Recurring reflection: I want to build a personal knowledge garden.",
            days_ago=200,
        )

        q = "personal knowledge garden reflection"
        default = ws.retriever.retrieve(q)
        temporal = ws.retriever.retrieve(q, query_type="temporal")

        def score_of(results):
            return next((r.score for r in results if "knowledge garden" in r.content), 0.0)

        default_score = score_of(default)
        temporal_score = score_of(temporal)
        assert default_score > 0.0 and temporal_score > 0.0, "memory retrieved under both intents"
        assert temporal_score > default_score * 2, (
            "relaxing decay for a temporal query should substantially raise an old memory's score"
        )


class TestAccessBoost:
    def test_recalled_memories_rank_higher(self, ws):
        ingest(ws, "Sketch for the reading tracker app, option Alpha: minimal list.", days_ago=30)
        ingest(ws, "Sketch for the reading tracker app, option Beta: minimal list.", days_ago=30)

        def scores():
            results = ws.retriever.retrieve("reading tracker app sketch")
            return {
                ("Beta" if "Beta" in r.content else "Alpha"): (i, r.score)
                for i, r in enumerate(results)
                if "reading tracker" in r.content
            }

        before = scores()
        assert "Alpha" in before and "Beta" in before
        assert before["Alpha"][1] == pytest.approx(before["Beta"][1], abs=0.05), (
            "near-identical unaccessed memories should score similarly"
        )

        # Recall Beta five times
        beta_memory = next(m for m in ws.memory_store.list_recent(10) if "Beta" in m.summary)
        for _ in range(5):
            ws.memory_store.record_chunk_access(beta_memory.chunk_ids[0])

        after = scores()
        assert after["Beta"][1] > after["Alpha"][1], (
            "access-frequency boost should raise the recalled memory's score"
        )
        assert after["Beta"][0] < after["Alpha"][0], "Beta should now rank above Alpha"

    def test_access_boost_math(self):
        scorer = ImportanceScorer(half_life_days=30.0, access_boost_factor=1.2)
        now = utcnow()
        base = scorer.score(0.3, now - timedelta(days=30), access_count=0, now=now)
        boosted = scorer.score(0.3, now - timedelta(days=30), access_count=5, now=now)
        assert boosted > base * 2, "five recalls should more than double the score"


class TestGraphExpansion:
    def test_connected_chunk_surfaces_without_lexical_overlap(self, ws):
        ingest(ws, "Spoke with Dana Whitfield about the funding options for Project Falcon.")
        ingest(ws, "The prototype for Project Falcon needs a dedicated hardware fund.")
        ingest(ws, "Watered the tomato plants on the balcony this morning.")

        # Graph structure: Dana Whitfield ↔ Project Falcon (co-occurrence),
        # so 2-hop traversal from Dana must reach the prototype chunk
        connected = ws.knowledge_graph.get_connected_chunks("Dana Whitfield", hops=2)
        chunk_contents = [ws.memory_store.get_chunk(cid).content for cid in connected]
        assert any("prototype" in c for c in chunk_contents), (
            "2-hop traversal should reach the Falcon prototype chunk"
        )

        results = ws.retriever.retrieve("what did I discuss with Dana Whitfield?")
        order = [r.content for r in results]

        assert "Dana Whitfield" in order[0], "the direct mention should rank first"
        falcon_rank = next(i for i, c in enumerate(order) if "prototype" in c)
        tomato_rank = next((i for i, c in enumerate(order) if "tomato" in c), len(order))
        assert falcon_rank < tomato_rank, (
            "the graph-connected chunk should outrank the unrelated one"
        )

    def test_graph_expansion_survives_vector_outage(self, ws, monkeypatch):
        # Graph seeds must not come from vector hits alone: when embedding is
        # down, keyword (FTS) matches still drive graph expansion.
        ingest(ws, "Spoke with Dana Whitfield about the funding options for Project Falcon.")
        ingest(ws, "The prototype for Project Falcon needs a dedicated hardware fund.")

        def boom(*_args, **_kwargs):
            raise RuntimeError("embedding backend down")

        monkeypatch.setattr(ws.retriever.embedder, "embed_query", boom)

        results = ws.retriever.retrieve("Dana Whitfield funding")
        contents = [r.content for r in results]

        assert any("Dana Whitfield" in c for c in contents), "keyword match still retrieved"
        assert any("prototype" in c for c in contents), (
            "FTS-seeded graph expansion should still reach the connected Falcon chunk"
        )


class TestFilters:
    def test_date_and_source_filters(self, ws):
        ingest(
            ws,
            "Journal: feeling scattered, too many projects at once.",
            days_ago=100,
            source=SourceType.JOURNAL,
        )
        ingest(ws, "Note: the single-tasking experiment starts today.", days_ago=1)

        recent_only = ws.retriever.retrieve(
            "projects and focus", date_from=utcnow() - timedelta(days=50)
        )
        assert recent_only
        assert all(r.timestamp >= utcnow() - timedelta(days=50) for r in recent_only)

        journal_only = ws.retriever.retrieve("projects and focus", source_filter="journal")
        assert journal_only
        assert all(r.source == SourceType.JOURNAL for r in journal_only)
