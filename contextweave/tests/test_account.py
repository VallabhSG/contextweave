"""Private workspace auth, isolation, and data-control tests."""

from __future__ import annotations


def _register(client, name=""):
    r = client.post("/api/auth/register", json={"name": name})
    assert r.status_code == 200
    body = r.json()
    return body["user_id"], body["api_key"]


class TestAuth:
    def test_register_returns_key_once(self, client):
        user_id, key = _register(client, "alice")
        assert key.startswith("cw_")
        r = client.get("/api/me", headers={"X-API-Key": key})
        assert r.status_code == 200
        assert r.json()["user_id"] == user_id
        assert r.json()["private"] is True

    def test_invalid_key_rejected(self, client):
        r = client.get("/api/me", headers={"X-API-Key": "cw_definitely_not_real"})
        assert r.status_code == 401

    def test_no_key_falls_back_to_demo(self, client):
        r = client.get("/api/me")
        assert r.status_code == 200
        assert r.json()["user_id"] == "demo"
        assert r.json()["private"] is False

    def test_bearer_header_also_works(self, client):
        _, key = _register(client)
        r = client.get("/api/me", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200
        assert r.json()["private"] is True


class TestIsolation:
    def test_workspaces_do_not_leak(self, client):
        _, key_a = _register(client)
        _, key_b = _register(client)
        ha = {"X-API-Key": key_a}
        hb = {"X-API-Key": key_b}

        r = client.post(
            "/api/ingest/text",
            json={"content": "Project Falcon budget approved by Dana Whitfield."},
            headers=ha,
        )
        assert r.status_code == 200

        assert client.get("/api/memories", headers=ha).json()["count"] == 1
        assert client.get("/api/memories", headers=hb).json()["count"] == 0
        assert client.get("/api/memories").json()["count"] == 0  # demo untouched

        q = client.post("/api/query", json={"query": "Falcon budget"}, headers=hb)
        assert q.status_code == 200
        assert q.json()["context_count"] == 0

        q = client.post("/api/query", json={"query": "Falcon budget"}, headers=ha)
        assert q.json()["context_count"] >= 1

    def test_demo_ingest_stays_in_demo(self, client):
        _, key = _register(client)
        client.post("/api/ingest/text", json={"content": "Public demo note"})
        assert client.get("/api/memories").json()["count"] == 1
        assert client.get("/api/memories", headers={"X-API-Key": key}).json()["count"] == 0


class TestDataControl:
    def test_demo_wipe_forbidden(self, client):
        assert client.delete("/api/memory").status_code == 403

    def test_wipe_erases_private_space(self, client):
        _, key = _register(client)
        h = {"X-API-Key": key}
        client.post("/api/ingest/text", json={"content": "Secret plan with Priya Nair"}, headers=h)
        assert client.get("/api/memories", headers=h).json()["count"] == 1

        r = client.delete("/api/memory", headers=h)
        assert r.status_code == 200

        assert client.get("/api/memories", headers=h).json()["count"] == 0
        health = client.get("/api/health", headers=h).json()
        assert health["memories"] == 0
        assert health["vectors"] == 0
        assert health["entities"] == 0

    def test_export_contains_all_sections(self, client):
        _, key = _register(client)
        h = {"X-API-Key": key}
        client.post("/api/ingest/text", json={"content": "Exportable note"}, headers=h)

        data = client.get("/api/export", headers=h).json()
        for section in ("events", "chunks", "memories", "graph"):
            assert section in data
        assert len(data["memories"]) == 1
        assert "entities" in data["graph"]


class TestAuthConfig:
    """GET /api/auth/config — frontend discovery of the Supabase project."""

    def test_disabled_when_not_configured(self, client):
        body = client.get("/api/auth/config").json()
        assert body["supabase"]["enabled"] is False
        assert "url" not in body["supabase"]
        assert "anon_key" not in body["supabase"]

    def test_exposes_public_config_when_fully_configured(self, client, monkeypatch):
        from contextweave.config import settings

        monkeypatch.setattr(settings, "supabase_url", "https://abc.supabase.co")
        monkeypatch.setattr(settings, "supabase_anon_key", "anon-public-key")
        monkeypatch.setattr(settings, "supabase_jwt_secret", "s3cret")
        body = client.get("/api/auth/config").json()
        assert body["supabase"] == {
            "enabled": True,
            "url": "https://abc.supabase.co",
            "anon_key": "anon-public-key",
        }

    def test_disabled_without_jwt_secret(self, client, monkeypatch):
        # Sign-in is pointless if the API cannot verify the session tokens
        from contextweave.config import settings

        monkeypatch.setattr(settings, "supabase_url", "https://abc.supabase.co")
        monkeypatch.setattr(settings, "supabase_anon_key", "anon-public-key")
        body = client.get("/api/auth/config").json()
        assert body["supabase"]["enabled"] is False
