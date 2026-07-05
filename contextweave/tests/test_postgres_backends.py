"""Postgres/Supabase backend integration tests against a real pgvector DB.

Spins up a disposable pgvector container (or uses CW_TEST_DATABASE_URL if
set). Skipped automatically when neither Docker nor a test database is
available. Embeddings are faked — this suite verifies storage behavior,
tenancy isolation, auth, and the whole point of the migration: state
surviving a process restart.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from contextweave import workspaces
from contextweave.api.rate_limit import limiter
from contextweave.config import settings
from contextweave.processing.embedder import LocalEmbedder

CONTAINER = "cw-pgtest"
PG_PORT = 54329
FAKE_VECTOR = [0.1] * 384


def _docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    return probe.returncode == 0


@pytest.fixture(scope="module")
def pg_url():
    override = os.environ.get("CW_TEST_DATABASE_URL")
    if override:
        yield override
        return

    if not _docker_ready():
        pytest.skip("Docker unavailable and CW_TEST_DATABASE_URL not set")

    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            CONTAINER,
            "-p",
            f"{PG_PORT}:5432",
            "-e",
            "POSTGRES_PASSWORD=cw",
            "pgvector/pgvector:pg16",
        ],
        capture_output=True,
        timeout=300,
    )
    if started.returncode != 0:
        pytest.skip(f"could not start pgvector container: {started.stderr.decode()[:200]}")

    url = f"postgresql://postgres:cw@127.0.0.1:{PG_PORT}/postgres"

    import psycopg

    for _ in range(60):
        try:
            psycopg.connect(url, connect_timeout=2).close()
            break
        except Exception:
            time.sleep(1)
    else:
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        pytest.skip("pgvector container never became ready")

    yield url
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


@pytest.fixture
def pg_client(pg_url, tmp_path, monkeypatch):
    """App client in Postgres mode with a clean database."""
    monkeypatch.setattr(settings, "database_url", pg_url)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "supabase_jwt_secret", "")
    monkeypatch.setattr(LocalEmbedder, "embed_query", lambda self, q: FAKE_VECTOR)
    monkeypatch.setattr(
        LocalEmbedder,
        "embed_chunks",
        lambda self, chunks, batch_size=50: [
            c.model_copy(update={"embedding": FAKE_VECTOR}) for c in chunks
        ],
    )
    workspaces.manager.reset()
    limiter.reset()

    from contextweave.storage.postgres import get_pool

    with get_pool().connection() as conn:
        conn.execute(
            "TRUNCATE cw_events, cw_chunks, cw_memories, cw_digests, "
            "cw_entities, cw_entity_edges, cw_entity_chunks, cw_users"
        )

    from main import app

    with TestClient(app) as c:
        yield c

    workspaces.manager.reset()
    limiter.reset()


def _register(client):
    body = client.post("/api/auth/register", json={"name": "pg"}).json()
    return body["user_id"], body["api_key"]


class TestPostgresMode:
    def test_ingest_query_and_fts(self, pg_client):
        _, key = _register(pg_client)
        h = {"X-API-Key": key}
        r = pg_client.post(
            "/api/ingest/text",
            json={"content": "Migration plan: move ContextWeave storage to Supabase pgvector."},
            headers=h,
        )
        assert r.status_code == 200
        assert r.json()["vectors_stored"] == 1

        q = pg_client.post("/api/query", json={"query": "supabase pgvector"}, headers=h)
        assert q.status_code == 200
        assert q.json()["context_count"] >= 1

        health = pg_client.get("/api/health", headers=h).json()
        assert health["memories"] == 1
        assert health["vectors"] == 1

    def test_workspace_isolation(self, pg_client):
        _, key_a = _register(pg_client)
        _, key_b = _register(pg_client)
        pg_client.post(
            "/api/ingest/text",
            json={"content": "Private plan for tenant A only."},
            headers={"X-API-Key": key_a},
        )
        assert pg_client.get("/api/memories", headers={"X-API-Key": key_a}).json()["count"] == 1
        assert pg_client.get("/api/memories", headers={"X-API-Key": key_b}).json()["count"] == 0
        assert pg_client.get("/api/memories").json()["count"] == 0  # demo space

    def test_state_survives_process_restart(self, pg_client):
        """The reason this migration exists."""
        _, key = _register(pg_client)
        h = {"X-API-Key": key}
        pg_client.post(
            "/api/ingest/text",
            json={"content": "Remember me across restarts: Aurora Project kickoff notes."},
            headers=h,
        )
        first_digest = pg_client.get("/api/digest", headers=h).json()
        assert first_digest["cached"] is False

        # Simulate a full process restart: drop every in-memory object and pool
        workspaces.manager.reset()
        limiter.reset()

        from main import app

        with TestClient(app) as reborn:
            mems = reborn.get("/api/memories", headers=h).json()
            assert mems["count"] == 1, "memories must survive a restart"
            assert "Aurora Project" in mems["memories"][0]["summary"]

            # API key survived (cw_users in Postgres), digest cache survived
            assert reborn.get("/api/me", headers=h).json()["private"] is True
            assert reborn.get("/api/digest", headers=h).json()["cached"] is True

            # Graph reloaded from Postgres into NetworkX
            entities = reborn.get("/api/graph/entities", headers=h).json()
            assert entities["count"] >= 1

    def test_export_wipe_and_access_counts(self, pg_client):
        _, key = _register(pg_client)
        h = {"X-API-Key": key}
        pg_client.post(
            "/api/ingest/text",
            json={"content": "Met Priya Nair about the export feature deadline."},
            headers=h,
        )

        ws = workspaces.manager.get(pg_client.get("/api/me", headers=h).json()["user_id"])
        chunk_id = ws.memory_store.list_recent(1)[0].chunk_ids[0]
        ws.memory_store.record_chunk_access(chunk_id)
        ws.memory_store.record_chunk_access(chunk_id)
        assert ws.memory_store.access_counts_by_chunk() == {chunk_id: 2}

        data = pg_client.get("/api/export", headers=h).json()
        for section in ("events", "chunks", "memories", "graph"):
            assert section in data
        assert len(data["memories"]) == 1

        assert pg_client.delete("/api/memory").status_code == 403  # demo protected
        assert pg_client.delete("/api/memory", headers=h).status_code == 200
        health = pg_client.get("/api/health", headers=h).json()
        assert health["memories"] == 0
        assert health["vectors"] == 0
        assert health["entities"] == 0


class TestSupabaseAuth:
    def _token(self, secret, sub="11111111-2222-3333-4444-555555555555", **claims):
        import jwt as pyjwt

        payload = {
            "sub": sub,
            "aud": "authenticated",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            **claims,
        }
        return pyjwt.encode(payload, secret, algorithm="HS256")

    def test_supabase_jwt_maps_to_stable_private_workspace(self, pg_client, monkeypatch):
        monkeypatch.setattr(settings, "supabase_jwt_secret", "test-secret-123")
        token = self._token("test-secret-123")
        h = {"Authorization": f"Bearer {token}"}

        me = pg_client.get("/api/me", headers=h).json()
        assert me["private"] is True
        assert me["user_id"].startswith("sb_")

        pg_client.post("/api/ingest/text", json={"content": "Signed-in note."}, headers=h)

        # A second session token for the same auth user reaches the same memory
        again = {"Authorization": f"Bearer {self._token('test-secret-123')}"}
        assert pg_client.get("/api/memories", headers=again).json()["count"] == 1

    def test_wrong_signature_rejected(self, pg_client, monkeypatch):
        monkeypatch.setattr(settings, "supabase_jwt_secret", "real-secret")
        token = self._token("attacker-secret")
        r = pg_client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401

    def test_jwt_rejected_when_not_configured(self, pg_client):
        token = self._token("whatever")
        r = pg_client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401
