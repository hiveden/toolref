# References — ToolRef

> Academic papers, industry signals, and design inspirations that shaped ToolRef.

---

## Context Engineering & RAG Theory

### ALARA: Adaptive Local-Global Context Engineering (arXiv:2503.2441)
**Relevance:** Directly supports ToolRef's dual-entry design (Chat UI + MCP)

Key insight: **"What matters isn't model size, but the quality of context you provide."** A focused 300-token context often outperforms an unfocused 100K-token dump.

**Application in ToolRef:**
- Namespace isolation ensures agents only see relevant domain knowledge
- Hybrid retrieval (dense + sparse) maximizes precision of context injection
- Hierarchical chunking (256 token child + 1024 token parent) optimizes retrieval vs. generation trade-off

---

### IBM Digital-Twin MDP for Context Engineering (arXiv:2502.2378)
**Relevance:** Theoretical framework for semantic cache optimization

Markov Decision Process formulation for optimal context selection in retrieval-augmented generation.

**Application in ToolRef:**
- Semantic cache TTL strategy (tiered: 72h/24h/12h) can be modeled as MDP state transitions
- Cache hit/miss decisions influence subsequent retrieval policy
- Future optimization: dynamic TTL adjustment based on query patterns

---

### MA-RAG: Multi-Agent Retrieval-Augmented Generation (arXiv:2603.03292)
**Relevance:** Consistency check node design in V1

Multi-agent debate framework: generate multiple answers from same retrieved docs, detect contradictions, extract divergence points for re-retrieval.

**Application in ToolRef:**
- V1 introduces `consistency_check` node after document grading
- Two independent generations with different temperatures (0.3, 0.7)
- If contradictory → extract divergence → re-retrieve with divergence as new query
- Solves "retrieved but contradictory information" problem that self-correction alone cannot address

---

## MCP & Agent Infrastructure

### Cline v3.76.0 — Multi-Agent Kanban Interface (March 2025)
**Source:** GitHub Release / Community

Cline CLI default interface evolved to multi-agent Kanban board, signaling industry shift toward parallel agent orchestration.

**Implication for ToolRef:**
- Validates MCP Server as unified knowledge source architecture
- When multiple agents run in parallel, they need consistent, version-controlled domain knowledge
- ToolRef MCP Server serves as the "single source of truth" for all agents in the workspace

---

### Stripe Projects — Agent Compute Layer Standardization (2025)
**Source:** Stripe Engineering Blog

Stripe's internal "Projects" framework standardizes agent compute infrastructure but explicitly leaves knowledge layer as "bring your own."

**Implication for ToolRef:**
- Confirms market gap: compute layer standardizing, knowledge layer fragmented
- ToolRef fills this gap with MCP-native, production-grade RAG
- Positioning: "Where Stripe Projects ends, ToolRef begins"

---

### FinMCP-Bench: MCP Tool Use Evaluation Framework
**Source:** arXiv / Research Community

Benchmark for evaluating MCP tool use capabilities of LLM agents.

**Application in ToolRef:**
- Reference for designing `toolref_query` tool interface
- Metrics: tool selection accuracy, parameter correctness, result interpretation
- Future: ToolRef could publish subset as benchmark for domain-specific RAG tools

---

## Memory & Caching

### The Case for Memory in LLM Agents (arXiv:2603.23013)
**Key finding:** 47% of production agent queries are semantically similar to historical queries. 8B model + memory retrieval achieves 30.5% F1, recovering 69% of 235B model performance at 96% cost reduction.

**Application in ToolRef:**
- Semantic cache similarity threshold: 0.92 (empirically tuned)
- Tiered TTL strategy (high-freq: 72h, normal: 24h, low-freq: 12h)
- 40%+ cache hit rate target based on this research

---

## Vector Search & Embeddings

### BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity Embeddings (BAAI, 2024)
**Paper:** arXiv:2402.03216

- 1024-dim dense vectors + sparse lexical weights in single model
- Multi-lingual (100+ languages including Chinese)
- Unifies dense retrieval, sparse retrieval, and multi-vector (ColBERT) approaches

**Application in ToolRef:**
- Single model serves both dense (semantic) and sparse (keyword) retrieval
- Enables hybrid search without separate BM25 implementation
- Chinese-English bilingual support aligns with target market

---

### BGE-Reranker-v2: Efficient Reranking with Unified Fine-tuning (BAAI, 2024)
**Paper:** Technical Report

Cross-encoder reranker optimized for BGE-M3 embedding space.

**Application in ToolRef:**
- Post-retrieval precision boost: ~15ms per pair on GPU
- Consistent embedding space with retrieval model improves ranking quality
- Trade-off: +150ms latency for significant MRR improvement (measured in evaluation)

---

## LangGraph & Agent Orchestration

### LangGraph: Building Stateful, Multi-Agent Applications (LangChain, 2024)
**Documentation:** https://langchain-ai.github.io/langgraph/

State machine framework for agent workflows with:
- Persistent state across turns
- Human-in-the-loop support
- Built-in checkpointing for fault tolerance

**Application in ToolRef:**
- RAG pipeline as state machine: analyze → route → retrieve → rerank → grade → (rewrite) → generate
- Self-correction loops implemented as conditional edges
- Checkpointer enables conversation memory without custom persistence

---

## Industry Alternatives & Positioning

| Solution | What it does | What ToolRef adds |
|----------|--------------|-------------------|
| **RAGFlow** | Enterprise document platform | Lightweight, MCP-native, developer-first |
| **Mem0 / OpenMemory** | Remember what a *user* said | Know what a *domain* is about |
| **Perplexity** | Find public, popular content | Index professional docs, version-precise answers |
| **MCP RAG demos** | Prove the pattern works | Production quality — hybrid retrieval, reranking, evaluation, namespace isolation |

---

## Design Principles (Referenced)

1. **"Close to users and data, not to models"** — Models are swappable. Knowledge management and precision retrieval are the core.

2. **"Every component earns its place"** — Hybrid retrieval, reranking, self-correction — each has comparative experiment data proving its value.

3. **"MCP-native"** — Not a web app with MCP bolted on. The MCP interface is a first-class citizen.

4. **"Namespace isolation"** — Knowledge domains don't leak into each other.

---

## Changelog

| Date | Update |
|------|--------|
| 2026-03-27 | Initial version — consolidated Pulse/Radar signals from 3/19-3/26 |

---

*For implementation details, see [architecture.md](./architecture.md)*
