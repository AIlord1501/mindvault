# Local Model Installation & Configuration

MindVault enforces **offline-only operation**: models are **never** downloaded automatically at runtime and no prompts or vault documents ever leave `127.0.0.1`. Place or download models explicitly beforehand into `MINDVAULT_MODELS_DIR` (default: `./data/models`).

## 1. Local Embedding Model (Sentence-Transformers)

Used for dense semantic search (ChromaDB), Concept Graph edges, and belief deduplication. When absent, MindVault degrades cleanly to BM25 lexical search and warns cleanly on semantic-only endpoints.

```bash
# Example: pre-download once on a connected setup machine into data/models/bge-small
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5').save('data/models/bge-small')"
```

Configure in environment:

```powershell
$env:MINDVAULT_EMBEDDING_MODEL = ".\data\models\bge-small"
```

## 2. Local LLM (GGUF via `llama-cpp-python` or Loopback Ollama)

Used for streaming Chat (`/api/v1/chat/stream`), Socratic Mode, and Belief Extraction.

### Option A: `llama-cpp-python` + GGUF file

1. Install `llama-cpp-python` in your local Python environment.
2. Place a local quantized GGUF model (e.g. `qwen2.5-7b-instruct-q4_k_m.gguf`) in `./data/models/`.
3. Configure:

```powershell
$env:MINDVAULT_LLM_PROVIDER = "llamacpp"
$env:MINDVAULT_LLM_MODEL_PATH = ".\data\models\qwen2.5-7b-instruct-q4_k_m.gguf"
$env:MINDVAULT_LLM_CONTEXT_SIZE = "4096"
$env:MINDVAULT_LLM_N_GPU_LAYERS = "0"
```

### Option B: Loopback Ollama

MindVault strictly restricts `MINDVAULT_OLLAMA_BASE_URL` to `http://127.0.0.1:<port>` or `http://localhost:<port>` and never calls `/api/pull`.

```powershell
$env:MINDVAULT_LLM_PROVIDER = "ollama"
$env:MINDVAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:MINDVAULT_OLLAMA_MODEL = "qwen2.5:7b"
```

## 3. Local NLI Cross-Encoder (Optional)

Used for contradiction vs. entailment scoring between beliefs (`ContradictionDetector`). Falls back deterministically to `HeuristicLexicalNLIAdapter` when unconfigured.

```powershell
$env:MINDVAULT_NLI_MODEL_PATH = ".\data\models\deberta-v3-small-nli"
```

## 4. Optional Offline OCR & Whisper

- **OCR (`pytesseract` + Tesseract):** Set `MINDVAULT_TESSERACT_CMD` if `tesseract` is not on `PATH`.
- **Audio (`faster-whisper`):** Set `MINDVAULT_WHISPER_MODEL_PATH` to a local CTranslate2 Whisper folder.
