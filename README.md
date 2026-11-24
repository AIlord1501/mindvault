# MindVault

MindVault is an offline-first personal knowledge desktop application for local document ingestion, knowledge retrieval, belief evolution, contradiction detection, retention tracking, and knowledge graph visualization.

## Core Principles

- **Offline-first:** No external network calls during normal operation. All storage, embeddings, and LLM inference run locally.
- **Loopback-only backend:** The FastAPI backend binds exclusively to `127.0.0.1`.
- **Hardened Electron shell:** `nodeIntegration` is disabled in the renderer, `contextIsolation` and `sandbox` are enabled, and only a minimal IPC surface is exposed through `preload.js`.
- **Untrusted input posture:** All API inputs are validated, SQL queries are parameterized, and ingested documents are treated as untrusted data.
- **Configurable storage & models:** Data paths, SQLite/ChromaDB locations, embedding models, and local LLM backends are configurable via environment variables and degrade gracefully when optional models are unavailable.

## Project Structure

```text
MindVault/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers (REST + SSE)
│   │   ├── core/         # Configuration, security, lifecycle
│   │   ├── db/           # SQLite metadata persistence
│   │   ├── ingestion/    # Document parsing and chunking
│   │   ├── retrieval/    # Vector + lexical retrieval
│   │   ├── chat/         # Bounded planning, prompts, streaming chat
│   │   ├── beliefs/      # Belief evolution & contradiction tracking
│   │   ├── graph/        # Knowledge graph projection
│   │   ├── scheduler/    # Retention & spaced-review scheduling
│   │   └── models/       # Local embedding & LLM adapters
│   ├── tests/            # Backend pytest suite
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── main.js           # Electron main process
│   ├── preload.js        # Minimal contextBridge IPC surface
│   ├── package.json
│   ├── tests/            # Frontend smoke & security tests
│   └── renderer/
│       ├── index.html
│       ├── css/
│       └── js/
├── scripts/              # Local developer helper scripts
├── docs/
│   └── architecture.md   # System boundaries and design overview
├── .gitignore
├── README.md
└── run.py                # Local backend launcher
```

See [docs/architecture.md](docs/architecture.md) for detailed system boundaries and security architecture.

## Installation

### 1. Python Backend (Python 3.9+)

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Electron Frontend (Node.js 18+)

```bash
cd frontend
npm install
```

## Configuration

All runtime paths and ports can be configured with environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MINDVAULT_HOST` | `127.0.0.1` | Backend bind address (loopback only) |
| `MINDVAULT_PORT` | `8765` | Backend HTTP port |
| `MINDVAULT_DATA_DIR` | `./data` | Root directory for local user data |
| `MINDVAULT_SQLITE_PATH` | `<DATA_DIR>/mindvault.db` | SQLite metadata database path |
| `MINDVAULT_CHROMA_DIR` | `<DATA_DIR>/chroma` | ChromaDB persistence directory |
| `MINDVAULT_CHROMA_COLLECTION` | `mindvault_chunks` | ChromaDB collection name for chunk vectors |
| `MINDVAULT_MODELS_DIR` | `<DATA_DIR>/models` | Local model directory (GGUF / embeddings) |
| `MINDVAULT_VAULT_DIR` | `<DATA_DIR>/vault` | Managed local storage for ingested documents |
| `MINDVAULT_LOGS_DIR` | `<DATA_DIR>/logs` | Local structured log directory |
| `MINDVAULT_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `MINDVAULT_LOG_TO_FILE` | `false` | Write rotating JSON logs to `<LOGS_DIR>/mindvault.log` |
| `MINDVAULT_SQLITE_BUSY_TIMEOUT_MS` | `5000` | SQLite busy timeout in milliseconds |
| `MINDVAULT_MAX_UPLOAD_SIZE_BYTES` | `52428800` | Maximum allowed file size per upload (50 MiB) |
| `MINDVAULT_CHUNK_SIZE` | `1000` | Default target chunk size in characters |
| `MINDVAULT_CHUNK_OVERLAP` | `150` | Default chunk overlap in characters |
| `MINDVAULT_EMBEDDING_MODEL` | `""` | Optional local embedding model path/name (`local_files_only`) |
| `MINDVAULT_EMBEDDING_BATCH_SIZE` | `32` | Batch size for local embedding generation and Chroma upserts |
| `MINDVAULT_RETRIEVAL_TOP_K` | `10` | Default number of hybrid search results returned |
| `MINDVAULT_RETRIEVAL_CANDIDATE_K` | `30` | Candidate pool size fetched from ChromaDB and BM25 before RRF |
| `MINDVAULT_SEMANTIC_WEIGHT` | `0.6` | Weight assigned to semantic vector ranks in RRF fusion |
| `MINDVAULT_BM25_WEIGHT` | `0.4` | Weight assigned to BM25 lexical ranks in RRF fusion |
| `MINDVAULT_RRF_K` | `60` | Reciprocal Rank Fusion smoothing constant |
| `MINDVAULT_RERANK_ENABLED` | `true` | Enable deterministic local phrase/coverage reranking |
| `MINDVAULT_LLM_PROVIDER` | `llamacpp` | Local LLM backend (`llamacpp` or `ollama`) |
| `MINDVAULT_LLM_MODEL_PATH` | `""` | Optional local GGUF model file path |
| `MINDVAULT_LLM_CONTEXT_SIZE` | `4096` | Context window size for local llama.cpp inference |
| `MINDVAULT_LLM_MAX_TOKENS` | `1024` | Upper token limit per generated answer |
| `MINDVAULT_LLM_TEMPERATURE` | `0.2` | Sampling temperature (`0.0` - `2.0`) |
| `MINDVAULT_LLM_N_GPU_LAYERS` | `0` | GPU offload layers for llama.cpp (`0` = CPU only) |
| `MINDVAULT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Loopback-only local Ollama endpoint |
| `MINDVAULT_OLLAMA_MODEL` | `""` | Local Ollama model tag (must exist locally; no remote pulls) |
| `MINDVAULT_OLLAMA_TIMEOUT_SECONDS` | `30.0` | Timeout in seconds for local Ollama requests |
| `MINDVAULT_CHAT_MAX_SUBQUESTIONS` | `3` | Maximum subquestions per decomposed complex query (1-5) |
| `MINDVAULT_CHAT_MAX_CONTEXT_CHUNKS` | `12` | Maximum grounded chunks injected into a chat prompt |
| `MINDVAULT_BELIEF_DEDUP_THRESHOLD` | `0.90` | Cosine similarity threshold for deduplicating extracted beliefs |
| `MINDVAULT_BELIEF_MIN_CONFIDENCE` | `0.30` | Minimum extraction confidence required to retain a belief |
| `MINDVAULT_BELIEF_CANDIDATE_TOP_K` | `10` | Max active candidate beliefs checked per new belief with NLI |
| `MINDVAULT_BELIEF_CANDIDATE_MIN_SIMILARITY` | `0.35` | Minimum semantic similarity before running NLI comparison |
| `MINDVAULT_CONTRADICTION_THRESHOLD` | `0.70` | Minimum NLI contradiction probability to flag an open conflict |
| `MINDVAULT_ENTAILMENT_THRESHOLD` | `0.70` | Minimum NLI entailment probability to record an entailment edge |
| `MINDVAULT_NLI_MODEL_PATH` | `""` | Optional local NLI classifier path (`local_files_only`) |
| `MINDVAULT_GRAPH_SIMILARITY_THRESHOLD` | `0.65` | Minimum cosine similarity for Concept Graph `similar_chunk` edges |
| `MINDVAULT_GRAPH_MAX_NEIGHBORS_PER_NODE` | `8` | Maximum ANN candidate neighbors linked per chunk node |
| `MINDVAULT_GRAPH_MAX_NODES` | `500` | Default maximum nodes returned per D3 graph query |
| `MINDVAULT_GRAPH_MAX_EDGES` | `2000` | Default maximum edges returned per D3 graph query |
| `MINDVAULT_RETENTION_INITIAL_STABILITY_DAYS` | `1.0` | Initial memory stability $S_0$ (days) for newly indexed chunks |
| `MINDVAULT_RETENTION_MIN_STABILITY_DAYS` | `0.25` | Minimum memory stability floor (days) on repeated lapses |
| `MINDVAULT_RETENTION_MAX_STABILITY_DAYS` | `3650.0` | Maximum memory stability cap (days) |
| `MINDVAULT_RETENTION_THRESHOLD` | `0.75` | Retention cutoff below which a chunk becomes due for review |
| `MINDVAULT_RETENTION_FACTOR_FORGOT` | `0.50` | Stability multiplier on `forgot` (rating `1`) |
| `MINDVAULT_RETENTION_FACTOR_HARD` | `1.20` | Stability multiplier on `hard` (rating `2`) |
| `MINDVAULT_RETENTION_FACTOR_GOOD` | `2.20` | Base stability multiplier on `good` (rating `3`) |
| `MINDVAULT_RETENTION_FACTOR_EASY` | `3.20` | Base stability multiplier on `easy` (rating `4`) |
| `MINDVAULT_WHISPER_MODEL_PATH` | `""` | Optional local Whisper model path (offline only) |
| `MINDVAULT_TESSERACT_CMD` | `""` | Optional path to local Tesseract OCR binary |
| `MINDVAULT_OFFLINE_ONLY` | `true` | Enforce offline-only runtime mode |

## Startup

### Run the local backend

```bash
python run.py
```

Verify the backend health and system status endpoints:

```text
GET http://127.0.0.1:8765/health
GET http://127.0.0.1:8765/api/v1/health
GET http://127.0.0.1:8765/api/v1/system/status
```

Validate launcher configuration without starting the server:

```bash
python run.py --check
```

### Run the Electron desktop client

```bash
cd frontend
npm start
```

The desktop shell opens a frameless `1280x800` window with a loading splash, sidebar navigation (`Chat`, `Graph`, `Dashboard`, `Research`, `Settings`), bottom status bar, light/dark themes, and a `Ctrl+K` command palette. If the local backend is not already running on `127.0.0.1:8765`, the Electron main process spawns `run.py` automatically (without a shell) and shuts it down cleanly on exit. Set `MINDVAULT_SKIP_BACKEND_SPAWN=1` to disable auto-spawning.

## Running Tests

### Backend tests

```bash
pytest backend/tests
```

### Frontend smoke tests and syntax checks

```bash
cd frontend
npm run check
npm test
```
