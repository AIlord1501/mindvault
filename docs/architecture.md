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
- Only permits explicitly whitelisted request/response IPC channels (e.g., application metadata and backend loopback URL).
- Never exposes raw `ipcRenderer`, `fs`, `child_process`, or `require`.

### 2.3 Electron Main Process (`frontend/main.js`)
- Creates and manages the application window.
- Blocks external window creation (`setWindowOpenHandler` denies all) and blocks navigation away from local application files.
- Handles minimal IPC queries from the preload bridge.

### 2.4 Local Backend API (`backend/app/api`)
- Bound **exclusively** to `127.0.0.1` (non-loopback hosts are rejected by configuration validation).
- Validates all incoming payloads using Pydantic schemas.
- Exposes REST endpoints for CRUD/queries and Server-Sent Events (SSE) for streaming long-running local jobs and LLM responses.

### 2.5 Persistence & Model Adapters
- **SQLite (`backend/app/db`):** Stores documents, chunks, beliefs, contradictions, retention schedules, and graph metadata using parameterized SQL queries only.
- **ChromaDB (`backend/app/retrieval`):** Stores local embeddings on disk under the configured `MINDVAULT_CHROMA_DIR`.
- **Local Models (`backend/app/models`):** Wraps local sentence-transformer-compatible embedding models and local LLM backends (`llama.cpp` GGUF or optional local Ollama). Unavailable or missing models must surface clear, structured status rather than crashing the application.

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
| `backend/app/beliefs` | Belief statements, evidence links, temporal evolution, contradictions |
| `backend/app/graph` | Knowledge graph nodes/edges computation and filtering |
| `backend/app/scheduler` | Memory decay modeling and spaced-review scheduling |
| `backend/app/models` | Pluggable local embedding and LLM runtime adapters |
