# MindVault Architecture

This document defines the system boundaries, security model, and module responsibilities for MindVault.

## 1. High-Level System Boundaries

MindVault is a single-user, offline-first desktop application composed of two local processes communicating strictly over loopback (`127.0.0.1`):

```text
+---------------------------------------------------------------+
|                    Electron Desktop Client                    |
|                                                               |
|  +-----------------------+       +-------------------------+  |
|  |   Renderer Process    |  IPC  |      Main Process       |  |
|  |  (Vanilla JS + D3.js) | <---> |   (main.js / Window)    |  |
|  |  - sandbox: true      |       |   - CSP & nav guards    |  |
|  |  - contextIsolation   |       |   - minimal IPC handlers|  |
|  |  - nodeIntegration: F |       +-------------------------+  |
|  +-----------+-----------+                                    |
|              | REST / SSE (http://127.0.0.1:<port>)           |
+--------------+------------------------------------------------+
               |
               v
+---------------------------------------------------------------+
|               Local FastAPI Backend (127.0.0.1)               |
|                                                               |
|  api/        -> Input validation, REST endpoints, SSE streams |
|  core/       -> Settings, loopback enforcement, health        |
|  ingestion/  -> Untrusted file parsing, sanitization, chunks  |
|  retrieval/  -> Hybrid vector + lexical retrieval             |
|  chat/       -> Bounded planner, prompts, SSE chat workflow   |
|  beliefs/    -> Claim extraction, belief evolution, conflicts |
|  graph/      -> Entity/belief graph construction for D3.js    |
|  scheduler/  -> Retention scoring & spaced review queues      |
|  models/     -> Local embedding & GGUF / Ollama adapters      |
|  db/         -> Parameterized SQLite access & migrations      |
+--------------+--------------------------------+---------------+
               |                                |
               v                                v
+------------------------------+  +-----------------------------+
|     SQLite Metadata Store    |  |    ChromaDB Vector Store    |
|  (local file in DATA_DIR)    |  |  (local dir in DATA_DIR)    |
+------------------------------+  +-----------------------------+
```

## 2. Process & Trust Boundaries

### 2.1 Renderer Process (Untrusted UI Boundary)
- Renders user content, extracted document text, and D3.js visualizations.
- Treats all document and model text as **untrusted data** (no unsafe HTML injection).
- Runs with `nodeIntegration: false`, `contextIsolation: true`, and `sandbox: true`.
- Enforces a strict Content Security Policy (CSP) allowing only local assets and loopback API calls (`http://127.0.0.1:*`).
- Cannot access Node.js APIs, the filesystem, or arbitrary IPC channels.

### 2.2 Preload Bridge (`frontend/preload.js`)
- Exposes a minimal, frozen `window.mindvault` API via `contextBridge.exposeInMainWorld`.
- Only permits explicitly whitelisted IPC operations: `getConfig`, `getBackendStatus`, `retryBackend`, `windowControl` (`minimize` | `maximize` | `close`), and `onBackendStatus` (with event stripping and unsubscribe).
- Never exposes raw `ipcRenderer`, `fs`, `child_process`, shell execution, or `require`.

### 2.3 Electron Main Process (`frontend/main.js` & `frontend/backendManager.js`)
- Displays a frameless, sandboxed loading splash (`renderer/splash.html`) during startup and transitions to the frameless `1280x800` main window (`renderer/index.html`) once ready (or gracefully on backend startup failure).
- Manages the local backend lifecycle via `BackendManager`: probes `http://127.0.0.1:8765/health` first, spawns `run.py` with `shell: false` only when no healthy backend is running, polls `/health` continuously, and terminates any managed child process cleanly on `before-quit`.
- Blocks external window creation (`setWindowOpenHandler` denies all) and blocks navigation away from local `file://` application files.

### 2.4 Local Backend API (`backend/app/api`)
- Bound **exclusively** to `127.0.0.1:8765` (non-loopback hosts are rejected by configuration validation).
- Managed via FastAPI `lifespan` for local directory creation, SQLite initialization, transactional migrations, structured JSON logging, and graceful shutdown.
- Validates all incoming payloads using Pydantic schemas and returns structured `422` validation errors.
- Exposes root `GET /health` and versioned `/api/v1` endpoints:
  - System: `GET /api/v1/health`, `GET /api/v1/system/status`
  - Ingestion & Documents: `POST /api/v1/ingest/file`, `POST /api/v1/ingest/batch`, `GET /api/v1/documents`, `GET /api/v1/documents/{id}`, `DELETE /api/v1/documents/{id}`
  - Retrieval & Indexing: `POST /api/v1/search`, `POST /api/v1/index/rebuild`, `GET /api/v1/index/status`
  - Chat & SSE Streaming: `POST /api/v1/chat`, `POST /api/v1/chat/stream`
  - Beliefs & Contradictions: `POST /api/v1/beliefs/extract/{chunk_id}`, `GET /api/v1/beliefs`, `GET /api/v1/beliefs/{id}`, `GET /api/v1/contradictions`, `POST /api/v1/contradictions/{id}/resolve`, `POST /api/v1/contradictions/{id}/dismiss`
- Local ingestion (`backend/app/ingestion`) validates file types and sizes, deduplicates by SHA-256 content hash, imports files into `MINDVAULT_VAULT_DIR`, extracts text via modular adapters (PyMuPDF for PDF, UTF-8 text/Markdown, optional Pillow+pytesseract OCR, optional offline Whisper with local model paths), normalizes untrusted text, splits into boundary-aware overlapping chunks persisted atomically in SQLite, and synchronizes chunk IDs into the retrieval indexes.
- Offline hybrid retrieval (`backend/app/retrieval` & `backend/app/models/embeddings.py`) combines persistent ChromaDB cosine similarity search (with local sentence-transformer-compatible embeddings loaded with `HF_HUB_OFFLINE=1` / `local_files_only=True`) and `rank_bm25` (`BM25L`) lexical retrieval using weighted Reciprocal Rank Fusion (RRF) and optional deterministic phrase/coverage reranking. Metadata filters (`epoch_min`, `epoch_max`, `document_ids`, `file_types`) are enforced across both engines, and hybrid search degrades gracefully to BM25 when no local embedding model is available.
- Local streaming chat (`backend/app/chat` & `backend/app/models/llm.py`) runs a strictly bounded 7-step workflow: (1) query analysis, (2) optional single-pass subquestion decomposition (capped at `MINDVAULT_CHAT_MAX_SUBQUESTIONS`, never recursive), (3) primary retrieval with epoch/metadata filters, (4) per-subquestion retrieval, (5) SQLite citation persistence in `citations`, (6) hardened `<vault_evidence>` prompt construction supporting `brief`, `summary`, `full`, and `socratic` modes, and (7) local token streaming via `LlamaCppProvider` (GGUF) or loopback-only `OllamaProvider` (`trust_env=False`, no `/api/pull`). `POST /api/v1/chat/stream` emits `status`, `subquestion`, `citation`, `token`, `answer_section`, `error`, and `done` SSE events and cancels local generation immediately on client disconnect.
- Belief tracking & contradiction detection (`backend/app/beliefs` & `backend/app/models/nli.py`) uses `BeliefExtractor` (local LLM only, 1-5 beliefs per chunk, hard 20-word cap, embedding cosine deduplication at `0.90` default) and `ContradictionDetector` (semantic pre-filtering capped at `MINDVAULT_BELIEF_CANDIDATE_TOP_K`, scoring pairs with `LocalSequenceClassificationNLIAdapter` offline or `HeuristicLexicalNLIAdapter` fallback). Contradictions support manual review (`accept_newer`, `accept_older`, `keep_both`, `context_dependent`, `false_positive`) while retaining full epoch history and superseding lineage via `superseded_by_id`.

### 2.5 Persistence & Model Adapters
- **SQLite (`backend/app/db`):** Managed by `DatabaseManager` (`WAL` mode, `foreign_keys = ON`, parameterized queries only) and `MigrationRunner` (versioned SQL migrations tracked with SHA-256 checksums in `schema_migrations`).
  - Core tables (`001_initial_schema.sql` & `002_belief_engine.sql`): `documents`, `chunks`, `beliefs` (with `superseded_by_id`, `context_note`, `embedding_json`, `updated_at`), `contradictions` (with `entailment_score`, `neutral_score`, `similarity_score`, `resolution_note`), `belief_relations`, `graph_edges`, `retention_scores`, `citations`, `morning_briefs`.
- **ChromaDB (`backend/app/retrieval/vector_store.py`):** Stores local embeddings on disk under `MINDVAULT_CHROMA_DIR` with `anonymized_telemetry=False`, keyed by SQLite `chunks.id` for idempotent upserts and cascade-consistent deletions.
- **Local Models (`backend/app/models`):** Wraps offline sentence-transformer-compatible embedding models (`SentenceTransformerEmbeddingAdapter`), local sequence-classification NLI models (`LocalSequenceClassificationNLIAdapter` + `HeuristicLexicalNLIAdapter`), and local LLM backends (`LlamaCppProvider` for `.gguf` files via `llama-cpp-python`, and loopback-only `OllamaProvider`). Unavailable or missing models surface clear `503` / SSE `error` responses or degrade cleanly with zero remote inference fallback.

## 3. Offline-First Guarantee

1. No telemetry, analytics, or external network requests during normal application operation.
2. Backend host configuration strictly enforces loopback binding (`127.0.0.1` / `localhost`).
3. Electron renderer CSP and navigation guards prevent loading remote scripts, styles, fonts, or pages.
4. Model files and user databases are stored in configurable local directories excluded from version control.

## 4. Module Ownership

| Package | Responsibility |
| --- | --- |
| `backend/app/api` | HTTP/SSE route definitions and request/response schemas |
| `backend/app/core` | Application settings, offline/loopback guards, shared errors |
| `backend/app/db` | SQLite connection management, schema initialization, repositories |
| `backend/app/ingestion` | Safe local file ingestion, text extraction, chunking |
| `backend/app/retrieval` | Embedding indexing, vector search, lexical search, ranking |
| `backend/app/chat` | Bounded query planning, grounded prompts, local SSE streaming chat |
| `backend/app/beliefs` | Belief statements, evidence links, temporal evolution, contradictions |
| `backend/app/graph` | Knowledge graph nodes/edges computation and filtering |
| `backend/app/scheduler` | Memory decay modeling and spaced-review scheduling |
| `backend/app/models` | Pluggable local embedding, NLI, and LLM runtime adapters |
