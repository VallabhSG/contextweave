"""ContextWeave as an MCP server — private, local-first memory for any agent.

Exposes ContextWeave's ingestion + hybrid-retrieval stack as Model Context
Protocol tools, so an MCP client (Claude Desktop, an agent framework, …) can
store and recall memory *on the user's own machine* — nothing leaves the box
unless a cloud LLM is explicitly configured.

Run it:  python -m contextweave.mcp_server   (stdio transport)

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "contextweave": {
          "command": "python",
          "args": ["-m", "contextweave.mcp_server"]
        }
      }
    }

Requires the optional `mcp` package: pip install mcp
"""

from __future__ import annotations

import os

from contextweave.api.pipeline import process_events
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.schemas import SourceType
from contextweave.timeutils import utcnow
from contextweave.workspaces import manager


def _workspace():
    """The single local memory workspace (override with CW_MCP_WORKSPACE)."""
    return manager.get(os.environ.get("CW_MCP_WORKSPACE", "mcp"))


def add_memory(content: str, source: str = "note") -> str:
    """Store a piece of text in private memory. Returns a short confirmation."""
    if not content or not content.strip():
        return "Nothing to store: content was empty."
    try:
        src = SourceType(source)
    except ValueError:
        src = SourceType.NOTE
    events = TextAdapter().ingest_text(content, timestamp=utcnow(), source=src)
    resp = process_events(_workspace(), events)
    return f"Stored {resp.chunks_created} memory chunk(s) ({resp.entities_extracted} entities)."


def search_memory(query: str, limit: int = 5) -> str:
    """Recall the most relevant memories for a query, ranked and cited.

    Returns the raw relevant passages (source + date + text) for the agent to
    use as context — the full hybrid retrieval + reranking stack runs locally.
    """
    if not query or not query.strip():
        return "Provide a query to search memory."
    results = _workspace().retriever.retrieve(query, top_k=max(1, min(limit, 20)))
    if not results:
        return "No relevant memories found."
    blocks = []
    for i, r in enumerate(results, 1):
        ts = r.timestamp.strftime("%Y-%m-%d")
        blocks.append(f"{i}. [{r.source.value} · {ts}] {r.content}")
    return "\n\n".join(blocks)


def build_server():
    """Construct the FastMCP server with the tools registered."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("contextweave")
    server.tool()(add_memory)
    server.tool()(search_memory)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
