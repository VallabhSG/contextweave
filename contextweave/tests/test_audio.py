"""Audio transcription endpoint tests (transcriber mocked, no network)."""

from __future__ import annotations

import contextweave.api.audio as audio_mod
from contextweave.config import settings
from contextweave.processing.entity_extractor import EntityExtractor


class TestAudioValidation:
    def test_rejects_unsupported_format(self, client):
        r = client.post("/api/ingest/audio", files={"file": ("note.txt", b"hello", "text/plain")})
        assert r.status_code == 400

    def test_rejects_oversized_audio(self, client):
        blob = b"a" * (audio_mod.MAX_AUDIO_BYTES + 1)
        r = client.post("/api/ingest/audio", files={"file": ("big.m4a", blob, "audio/m4a")})
        assert r.status_code == 413

    def test_no_groq_key_gives_503(self, client):
        r = client.post(
            "/api/ingest/audio", files={"file": ("seg.m4a", b"\x00\x01data", "audio/m4a")}
        )
        assert r.status_code == 503


class TestAudioIngestion:
    def _arm(self, monkeypatch, transcript):
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(audio_mod, "_transcribe", lambda filename, data: transcript)
        # Keep entity extraction offline even though a (fake) key is set
        monkeypatch.setattr(EntityExtractor, "extract_from_chunk", lambda self, chunk: [])

    def test_transcript_becomes_conversation_memory(self, client, monkeypatch):
        self._arm(monkeypatch, "We agreed to ship the beta on Friday with Sam Rivera.")

        r = client.post(
            "/api/ingest/audio", files={"file": ("seg.m4a", b"\x00\x01data", "audio/m4a")}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["transcript"].startswith("We agreed")
        assert body["chunks_created"] == 1

        mems = client.get("/api/memories?source=conversation").json()
        assert mems["count"] == 1
        assert "ship the beta" in mems["memories"][0]["summary"]

    def test_empty_transcript_stores_nothing(self, client, monkeypatch):
        self._arm(monkeypatch, "")

        r = client.post(
            "/api/ingest/audio", files={"file": ("seg.m4a", b"\x00\x01data", "audio/m4a")}
        )
        assert r.status_code == 200
        assert "No speech" in r.json()["message"]
        assert client.get("/api/memories").json()["count"] == 0

    def test_transcriber_failure_gives_502(self, client, monkeypatch):
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")

        def boom(filename, data):
            raise RuntimeError("whisper unavailable")

        monkeypatch.setattr(audio_mod, "_transcribe", boom)
        r = client.post(
            "/api/ingest/audio", files={"file": ("seg.m4a", b"\x00\x01data", "audio/m4a")}
        )
        assert r.status_code == 502
