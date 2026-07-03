"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from contextweave import workspaces
from contextweave.api.rate_limit import limiter
from contextweave.config import settings
from contextweave.processing.embedder import LocalEmbedder

# Deterministic fake vectors keep tests hermetic — no model download, no
# embedding compute — while still exercising Chroma storage and retrieval.
FAKE_VECTOR = [0.1] * 384


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App test client on temp storage, blank Groq key, fake embeddings."""
    monkeypatch.setattr(settings, "sqlite_db_path", str(tmp_path / "demo.db"))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(settings, "groq_api_key", "")
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

    from main import app

    with TestClient(app) as c:
        yield c

    workspaces.manager.reset()
    limiter.reset()
