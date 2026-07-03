"""API validation, security-header, and rate-limit tests.

These use the real FastAPI app with temp storage. No LLM or embedding
calls happen: the Groq key is blanked (regex/fallback paths) and the
vector store stays empty so retrieval skips embedding entirely.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import contextweave.api.routes as routes
from contextweave.api.rate_limit import limiter
from contextweave.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sqlite_db_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "groq_api_key", "")
    routes._instances.clear()
    limiter.reset()

    from main import app

    with TestClient(app) as c:
        yield c

    routes._instances.clear()
    limiter.reset()


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        for key in ("status", "events", "chunks", "memories", "vectors", "entities", "edges"):
            assert key in body
        assert body["status"] == "ok"


class TestSecurityHeaders:
    def test_root_has_security_headers(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Content-Security-Policy" in r.headers
        assert "script-src 'self'" in r.headers["Content-Security-Policy"]
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "Referrer-Policy" in r.headers

    def test_docs_exempt_from_csp_only(self, client):
        r = client.get("/docs")
        assert r.status_code == 200
        assert "Content-Security-Policy" not in r.headers
        assert r.headers["X-Content-Type-Options"] == "nosniff"


class TestIngestValidation:
    def test_rejects_unknown_source(self, client):
        r = client.post("/api/ingest/text", json={"content": "hello", "source": "bogus"})
        assert r.status_code == 400

    def test_rejects_empty_content(self, client):
        r = client.post("/api/ingest/text", json={"content": ""})
        assert r.status_code == 422

    def test_rejects_oversized_content(self, client):
        r = client.post("/api/ingest/text", json={"content": "x" * 100_001})
        assert r.status_code == 422

    def test_rejects_unknown_file_type(self, client):
        r = client.post("/api/ingest", files={"file": ("evil.exe", b"MZ", "application/x-dosexec")})
        assert r.status_code == 400

    def test_rejects_oversized_file(self, client):
        blob = b"a" * (routes.MAX_UPLOAD_BYTES + 1)
        r = client.post("/api/ingest", files={"file": ("big.txt", blob, "text/plain")})
        assert r.status_code == 413

    def test_rejects_too_many_batch_files(self, client):
        files = [
            ("files", (f"f{i}.txt", b"hi", "text/plain")) for i in range(routes.MAX_BATCH_FILES + 1)
        ]
        r = client.post("/api/ingest/batch", files=files)
        assert r.status_code == 400


class TestQueryValidation:
    def test_rejects_bad_date(self, client):
        r = client.post("/api/query", json={"query": "test", "date_from": "not-a-date"})
        assert r.status_code == 400

    def test_rejects_out_of_range_top_k(self, client):
        assert client.post("/api/query", json={"query": "t", "top_k": 0}).status_code == 422
        assert client.post("/api/query", json={"query": "t", "top_k": 51}).status_code == 422

    def test_rejects_overlong_query(self, client):
        r = client.post("/api/query", json={"query": "q" * 2_001})
        assert r.status_code == 422

    def test_empty_store_returns_fallback_answer(self, client):
        r = client.post("/api/query", json={"query": "what do I know?"})
        assert r.status_code == 200
        assert r.json()["context_count"] == 0


class TestRateLimit:
    def test_query_rate_limited(self, client):
        codes = [client.post("/api/query", json={"query": "hi"}).status_code for _ in range(21)]
        assert codes[:20] == [200] * 20
        assert codes[20] == 429
