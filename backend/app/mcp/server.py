"""ToolRef MCP Server — FastMCP instance and tool registration.

Creates a :class:`FastMCP` application named ``"toolref"`` and registers
all public RAG tools from :mod:`app.mcp.tools`.

Usage (from *main.py*)::

    from app.mcp.server import mcp

    mcp.run(transport="stdio")          # Claude Desktop
    mcp.run(transport="sse", port=8080) # SSE clients
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from app.mcp.tools import toolref_query as _toolref_query

logger = logging.getLogger(__name__)

# ── FastMCP instance ──────────────────────────────────────────────────────────

mcp: FastMCP = FastMCP(
    "toolref",
    instructions=(
        "ToolRef is an Agentic RAG engine. "
        "Use the `toolref_query` tool to search the knowledge base and get "
        "grounded answers with source citations."
    ),
)

# ── Tool registration ─────────────────────────────────────────────────────────


@mcp.tool()
async def toolref_query(
    query: str,
    namespace: str = "default",
    top_k: int = 5,
) -> dict:  # type: ignore[type-arg]
    """Search the ToolRef knowledge base and return a grounded answer.

    Runs the full Agentic RAG pipeline:
    query analysis → hybrid retrieval → cross-encoder reranking →
    relevance grading → optional query rewrite → LLM generation.

    Args:
        query:     Natural-language question or search query (max 2000 chars).
        namespace: Document namespace to scope the search (default: ``"default"``).
                   Different namespaces allow multi-tenant knowledge isolation.
        top_k:     Number of source documents to retrieve (1–20, default 5).

    Returns:
        ``{"answer": str, "sources": list[dict], "confidence": float}``

        * **answer**     — Synthesised answer grounded in retrieved documents.
        * **sources**    — List of source chunks, each containing
                          ``doc_title``, ``chunk_text``, ``url``, ``score``.
        * **confidence** — Heuristic confidence in ``[0.0, 1.0]`` derived from
                          retrieval relevance scores and pipeline signals.
    """
    return await _toolref_query(query=query, namespace=namespace, top_k=top_k)


# ── Health endpoint (for Docker healthcheck) ─────────────────────────────────


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):  # noqa: ARG001
    """Lightweight health endpoint for Docker/K8s liveness probes."""
    from starlette.responses import JSONResponse  # noqa: PLC0415

    return JSONResponse({"status": "ok", "service": "toolref-mcp"})


logger.debug("MCP server 'toolref' initialised with %d tool(s)", len(mcp._tool_manager._tools))  # noqa: SLF001
