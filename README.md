# Images extractor AI - Backend

Backend for extracting business-trip expenses from receipts and invoices (PDF, PNG, JPG, WEBP).
It uses **FastAPI**, **PaddleOCR 3.x** and **Ollama** (`qwen3-vl:8b-instruct`), and streams progress to the client via Server-Sent Events.

---

## 🏗️ Architecture

```
upload → PDF/image → page images ─┬─ preprocess (CPU) → PaddleOCR (GPU, 1 instance) ─┐
                                  │                                                  ├→ Ollama VLM (OCR text + image) → JSON → guardrails
                                  └──────────── downscaled image (1280px) ───────────┘
```

- **Endpoints:** `POST /api/report-stream` (mobile: first expense) and `POST /api/web-report-stream` (web: aggregated trip). Both return `text/event-stream`.
- **Pipelining:** pages of a request run concurrently. OCR is serialized on a single PaddleOCR instance; LLM calls are capped by `LLM_CONCURRENCY` across *all* requests. While one page is in the LLM, the next is in OCR.
- **Warm-up:** on startup the OCR model and the Ollama model are loaded, so the first request isn't slow.
- **Structured output:** the LLM is constrained to the JSON schema of `LLMExtraction` (`core/state.py`).
- **Stateless:** no database, no files written to disk. No authentication (run it on a private network or behind a gateway).

---

## 🖥️ Production server (Ubuntu 24.04, RTX 5060 Ti 16 GB)

Ollama, OCR and this backend all run on the same server. GPU memory budget: ~9-10 GB Ollama + ~1-2 GB PaddleOCR.

### 1. Ollama
Needs a recent Ollama build with Blackwell (CUDA 12.8+) support. `sudo systemctl edit ollama`:
```ini
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=2"
```
Then `sudo systemctl restart ollama && ollama pull qwen3-vl:8b-instruct`.

### 2. Backend
```bash
uv sync                      # on Linux installs paddlepaddle-gpu 3.3.1 (CUDA 12.9, bundled in the wheel)
OCR_DEVICE=gpu uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Run a single uvicorn worker: each worker loads its own OCR model into the GPU.

The first start downloads the PaddleOCR models into `~/.paddlex/official_models` (set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip the connectivity check on later starts).

### 3. Check
- `nvidia-smi` shows both `ollama` and `python` using GPU memory.
- Logs show `LLM page N: prompt_tokens=...` — it must stay well below `OLLAMA_NUM_CTX`.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_MODEL` | `qwen3-vl:8b-instruct` | Vision model used for extraction. |
| `OLLAMA_NUM_CTX` | `8192` | Context window. Ollama's own default (4096) silently truncates prompts. |
| `OLLAMA_KEEP_ALIVE` | `-1` | How long Ollama keeps the model loaded (`-1` = forever). |
| `OLLAMA_TIMEOUT` | `120` | Seconds per LLM call before the page fails. |
| `LLM_CONCURRENCY` | `2` | Max parallel LLM calls; keep equal to `OLLAMA_NUM_PARALLEL`. |
| `LLM_IMAGE_MAX_SIDE` | `1280` | Longest side (px) of the image sent to the LLM. |
| `OCR_DEVICE` | `cpu` | `gpu` on the server. |
| `OCR_LANG` | `ro` | PaddleOCR language (Latin model, keeps diacritics). |
| `MAX_FILE_SIZE_MB` | `10` | Max size per uploaded file. |
| `MAX_PAGES` | `20` | Max pages per request. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. |
| `LOG_LEVEL` | `INFO` | `DEBUG` also logs the full OCR text (contains personal data). |

---

## 🚀 How to Run Locally (macOS)

```bash
uv sync
uv run python -m uvicorn main:app --port 8000
```
On macOS OCR runs on CPU (slow, ~20-30 s per page). Point `OLLAMA_BASE_URL` at the server, or use a local model, e.g. `OLLAMA_MODEL=qwen2.5vl:7b`.

Swagger UI: `http://localhost:8000/docs`

---

## 🔍 Health Checks
- **`GET /api/health`** (Returns `{"status": "ok", ...}`)
