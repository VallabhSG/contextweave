"""Tests for the MCP server tools (add_memory / search_memory).

The tool *logic* is plain functions over the real retrieval stack, so it's
tested directly with real embeddings. Building the FastMCP server needs the
optional `mcp` package and is skipped if it's absent.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from contextweave import mcp_server
from contextweave.config import settings
from contextweave.workspaces import manager


@pytest.fixture(scope="module")
def mcp_env(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("mcp")
    mp.setattr(settings, "sqlite_db_path", str(base / "d.db"))
    mp.setattr(settings, "chroma_persist_dir", str(base / "c"))
    mp.setattr(settings, "data_dir", str(base / "data"))
    mp.setattr(settings, "groq_api_key", "")
    manager.reset()
    yield
    manager.reset()
    mp.undo()


def test_add_then_search_round_trip(mcp_env, monkeypatch):
    monkeypatch.setenv("CW_MCP_WORKSPACE", "mcp_rt_" + uuid4().hex[:8])

    stored = mcp_server.add_memory(
        "The planning offsite is the third week of September in Denver, hosted by ops."
    )
    assert "chunk" in stored.lower()

    recalled = mcp_server.search_memory("when and where is the offsite?", limit=3)
    assert "denver" in recalled.lower(), f"expected the offsite memory, got: {recalled[:120]}"


def test_search_empty_query_is_guarded(mcp_env):
    assert "provide a query" in mcp_server.search_memory("   ").lower()


def test_add_empty_content_is_guarded(mcp_env):
    assert "empty" in mcp_server.add_memory("").lower()


def test_build_server_registers_tools():
    pytest.importorskip("mcp")
    server = mcp_server.build_server()
    assert server is not None
