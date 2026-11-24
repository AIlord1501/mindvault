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
| `MINDVAULT_MODELS_DIR` | `<DATA_DIR>/models` | Local model directory (GGUF / embeddings) |
| `MINDVAULT_EMBEDDING_MODEL` | `""` | Optional local embedding model path/name |
| `MINDVAULT_LLM_MODEL_PATH` | `""` | Optional local GGUF model file path |
| `MINDVAULT_OFFLINE_ONLY` | `true` | Enforce offline-only runtime mode |

## Startup

### Run the local backend

```bash
python run.py
```

Verify the backend health endpoint:

```text
GET http://127.0.0.1:8765/health
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
