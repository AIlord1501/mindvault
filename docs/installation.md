# MindVault Installation Guide

MindVault runs strictly locally on your machine with zero cloud dependencies during normal operation.

## Prerequisites

- **Python:** 3.9 or higher (3.9+ validated in automated tests)
- **Node.js:** 18 or higher (includes `npm`)
- **Optional OCR binary:** Tesseract OCR (only needed for PNG/JPG OCR)
- **Optional local AI runtimes:**
  - `llama-cpp-python` (for local `.gguf` chat & belief extraction)
  - Local Ollama daemon (`http://127.0.0.1:11434`, optional alternative provider)
  - Local Sentence-Transformer & NLI model directories

## 1. Python Backend Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Verify backend configuration:

```bash
python run.py --check
```

## 2. Electron Desktop Frontend Setup

```bash
cd frontend
npm install
```

Run frontend syntax checks and unit tests:

```bash
npm run check
npm test
```

## 3. Running MindVault Locally

### Option A: PowerShell One-Command Launcher (Windows)

```powershell
.\start.ps1
```

Other launcher modes:
- `.\start.ps1 -Check` — Validate configuration and run all backend/frontend tests
- `.\start.ps1 -BackendOnly` — Start only the loopback FastAPI server (`127.0.0.1:8765`)
- `.\start.ps1 -FrontendOnly` — Start only the Electron shell

### Option B: Manual Two-Process Startup

```bash
# Terminal 1 (optional; Electron auto-spawns run.py if not already running)
python run.py

# Terminal 2
cd frontend
npm start
```

## 4. Packaging the Desktop App

Model weights (`.gguf`, `.bin`, `.onnx`, `.safetensors`, `models/`, and user vault `data/`) are excluded from packaging by default in `frontend/package.json`. Install `electron-builder` when producing binaries:

```bash
cd frontend
npm install --save-dev electron-builder
npm run pack
```
