"""Proactive digest endpoint and engine tests."""

from __future__ import annotations

from datetime import timedelta

from contextweave.reasoning.digest import DigestEngine
from contextweave.schemas import Memory, SourceType
from contextweave.timeutils import utcnow


class TestDigestEndpoint:
    def test_empty_memory_still_nudges(self, client):
        r = client.get("/api/digest")
        assert r.status_code == 200
        body = r.json()
        assert body["memory_count"] == 0
        assert body["headline"]

    def test_digest_after_ingest_detects_commitments(self, client):
        client.post(
            "/api/ingest/text",
            json={"content": "Action item: follow up with Priya Nair about the launch deadline."},
        )
        r = client.get("/api/digest")
        assert r.status_code == 200
        body = r.json()
        assert body["cached"] is False
        assert body["memory_count"] >= 1
        assert body["commitments"]

    def test_second_call_is_cached_and_force_regenerates(self, client):
        client.post("/api/ingest/text", json={"content": "Planning the quarter with Marcus Lee."})
        first = client.get("/api/digest").json()
        assert first["cached"] is False

        second = client.get("/api/digest").json()
        assert second["cached"] is True

        forced = client.get("/api/digest?force=true").json()
        assert forced["cached"] is False


class TestDigestEngineFallback:
    def _memory(self, summary, entities, days_ago=0):
        ts = utcnow() - timedelta(days=days_ago)
        return Memory(
            chunk_ids=["c1"],
            summary=summary,
            entities=entities,
            source=SourceType.NOTE,
            timestamp=ts,
        )

    def test_stale_threads_surface_as_gaps(self, client):
        engine = DigestEngine()
        memories = [
            self._memory("Fresh note about Project Iris", ["Project Iris"], days_ago=0),
            self._memory("Older note on Project Iris", ["Project Iris"], days_ago=1),
            self._memory("Discussed Novel Draft edits", ["Novel Draft"], days_ago=10),
            self._memory("More Novel Draft planning", ["Novel Draft"], days_ago=12),
        ]
        digest = engine.generate(memories)
        assert digest.llm_generated is False
        assert any("Novel Draft" in g for g in digest.gaps)
        assert "Project Iris" in digest.top_entities

    def test_commitment_signals_detected(self, client):
        engine = DigestEngine()
        memories = [self._memory("I'll send the deck by Friday, promised Alice.", ["Alice"])]
        digest = engine.generate(memories)
        assert digest.commitments
        assert "commitment" in digest.headline
