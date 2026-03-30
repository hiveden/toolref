# ToolRef

> The reference engine for AI Agents — hot-swappable domain expertise via RAG + MCP.

**ToolRef** is a production-grade RAG engine that turns professional documentation into on-demand domain knowledge for AI Agents and humans. Each namespace is an isolated knowledge domain. Agents query via MCP; humans query via Chat UI.

🚧 **Status: Pre-alpha — Under active development**

## The Problem

AI Agents are powerful generalist reasoners, but they can't keep up with domain-specific knowledge:

- **Knowledge staleness** — Models are trained months ago. Framework APIs change weekly.
- **No depth** — An LLM "knows" LangGraph exists, but can't tell you the exact `StateGraph` parameter signature in v0.3.
- **No boundaries** — When an Agent answers from general training data, you can't audit where the answer came from or control what it sees.

Web search helps with public, popular content. But professional documentation — framework references, design specifications, internal standards — needs structured, version-aware, traceable retrieval.

## The Thesis

> AI should not be omniscient generalists. The future is **generalist reasoning + on-demand specialized knowledge**.

This is the [Context Engineering](https://simonwillison.net/2025/Jun/27/context-engineering/) insight: **what matters isn't model size, but the quality of context you provide.** A focused 300-token context often outperforms an unfocused 100K-token dump.

ToolRef is the knowledge layer that makes this practical:

- **For Agents** — Query domain knowledge via MCP tools (`rag_query`, `document_add`, `namespace_list`). Like giving an Agent a hot-swappable expert reference library.
- **For Humans** — Ask questions in a Chat UI with traceable citations back to source documents.

## Why Not Just Use...

| Solution | What it does | What it doesn't do |
|---|---|---|
| **Longer context windows** | Fit more text in one prompt | Solve attention degradation, cost scaling, or knowledge staleness |
| **Web search / Perplexity** | Find public, popular content | Index professional docs, provide version-precise answers, or audit sources |
| **Mem0 / OpenMemory** | Remember what a *user* said | Know what a *domain* is about |
| **RAGFlow** | Enterprise document platform | Lightweight, MCP-native, developer-first |
| **MCP RAG demos** | Prove the pattern works | Production quality — no hybrid retrieval, reranking, evaluation, or namespace isolation |

ToolRef sits in the gap: **MCP-native, production-grade RAG with retrieval quality you can measure.**

## Demo

![Chat UI — Agentic RAG query with source citations](docs/screenshots/chat-ui-query.png)

*Chat UI showing a self-corrective RAG query with BGE-M3 retrieval, cross-encoder reranking, and traceable source citations.*

## Architecture

```
┌─────────────┐     ┌─────────────┐
│   Chat UI   │     │  AI Agent   │
│  (React)    │     │ (via MCP)   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               ▼
       ┌───────────────┐
       │   FastAPI      │
       │   Gateway      │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │  LangGraph     │
       │  RAG Engine    │
       │               │
       │  analyze →    │
       │  retrieve →   │
       │  rerank →     │
       │  grade →      │
       │  generate     │
       │  (or retry)   │
       └───────┬───────┘
               ▼
  ┌────────┬───────┬──────────┐
  │ Milvus │ Redis │ Postgres │
  │vectors │ cache │ metadata │
  └────────┴───────┴──────────┘
```

### Key Components

- **Ingestion Pipeline** — Parse (PDF/MD/TXT/HTML) → hierarchical chunking → BGE-M3 embedding → Milvus + PostgreSQL
- **Retrieval Engine** — LangGraph state machine: hybrid search (dense + sparse + RRF) → cross-encoder reranking → document grading → self-correction (max 2 retries)
- **Semantic Cache** — Redis-backed, cosine similarity threshold 0.92, 24h TTL
- **MCP Server** — Exposes `rag_query`, `document_add`, `namespace_list` as MCP tools
- **Namespace Isolation** — Each knowledge domain is a separate namespace with independent vector collections
- **Evaluation** — DSPy + Arize Phoenix for retrieval quality measurement

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Orchestration | LangGraph |
| Embeddings | BGE-M3 (1024-dim, dense + sparse) |
| Reranking | BGE-reranker-v2-m3 |
| Vector DB | Milvus (HNSW) |
| Metadata DB | PostgreSQL 16 |
| Cache | Redis 7 |
| Parsing | Unstructured.io |
| Frontend | React 18 + TypeScript + Shadcn/ui |
| Evaluation | DSPy + Arize Phoenix |
| LLM (dev) | Ollama + Qwen2.5-7B |
| LLM (prod) | DeepSeek-V3 / GPT-4o (swappable) |

## Design Principles

1. **Close to users and data, not to models.** Models are swappable. Knowledge management and precision retrieval are the core.
2. **Every component earns its place.** Hybrid retrieval, reranking, self-correction — each has comparative experiment data proving its value over the simpler alternative.
3. **MCP-native.** Not a web app with MCP bolted on. The MCP interface is a first-class citizen.
4. **Namespace isolation.** Knowledge domains don't leak into each other. An Agent querying LangGraph docs never gets React docs mixed in.

## Roadmap

- **MVP (Weeks 1-3):** Scaffolding, PDF/MD/TXT parsing, fixed-size chunking, dense retrieval, basic Chat UI
- **V1 (Weeks 4-9):** Hierarchical chunking, full agentic RAG flow, hybrid retrieval, reranking, self-correction, WebSocket streaming, semantic cache, MCP server, DSPy evaluation, auth, CI/CD
- **V2 (Weeks 10-12, optional):** GraphRAG, human-in-the-loop, incremental re-indexing

## Getting Started

> Coming soon — scaffolding in progress.

## License

MIT

---

*Built by [Toolman Lab](https://github.com/toolmanlab) — where tools get serious.*
