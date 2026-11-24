# MindVault Troubleshooting & Recovery Guide

## 1. Backend Fails to Start or Reports Non-Loopback Error

- **Symptom:** `ValueError: Invalid host ... MindVault backend must bind exclusively to 127.0.0.1.`
- **Fix:** Ensure `MINDVAULT_HOST` is unset or explicitly `127.0.0.1`. Remote hosts (`0.0.0.0`, LAN IPs) are rejected by design.

## 2. Corrupted SQLite Database Recovery

- **Behavior:** On startup, `DatabaseManager.ensure_healthy_or_recover()` runs `PRAGMA quick_check;`.
- **Recovery:** If corruption is detected, MindVault automatically closes handles, quarantines the broken file (and `-wal`/`-shm` sidecars) to `<sqlite_path>.corrupt.<timestamp>.bak`, creates a fresh SQLite DB, applies all migrations, and reports `"recovery.database.recovered": true` in `GET /api/v1/system/status`.

## 3. Corrupted Vector Index Recovery (ChromaDB)

- **Behavior:** On startup or query, `ChromaVectorStore.ensure_healthy_or_recover()` verifies the Chroma collection.
- **Recovery:** If Chroma raises an error due to corrupt persistence files, MindVault quarantines `<chroma_dir>` to `<chroma_dir>_corrupt_<timestamp>`, recreates the collection, and automatically runs `rebuild_index()` when SQLite chunks are present and a local embedding model is configured. You can also trigger `POST /api/v1/index/rebuild` from Settings.

## 4. Missing Local LLM or Embedding Model

- **Chat (`503` or SSE `error`):** Configure `MINDVAULT_LLM_MODEL_PATH` (GGUF) or start local loopback Ollama and set `MINDVAULT_OLLAMA_MODEL`. MindVault will never fall back to cloud inference.
- **Search degraded to BM25:** Hybrid retrieval automatically falls back to BM25 lexical search with a clear warning if no local sentence-transformer model path is set.

## 5. Desktop Shell Shows "Local backend offline"

- Click **Retry Local Backend** in the red banner or Settings panel.
- Verify port `8765` is not occupied by another service, or launch with `python run.py` directly to view structured JSON startup logs.
- Check Settings -> **Model Status** and **Index Status** to confirm synchronization.
