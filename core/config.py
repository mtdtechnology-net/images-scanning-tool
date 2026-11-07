"""Runtime configuration, read from environment variables."""
import os


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


# ─── Ollama (LLM) ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b-instruct")
# Ollama defaults to a 4096-token context on GPUs < 24 GB, which silently truncates
# image + OCR text + prompt. 8192 fits comfortably in 16 GB VRAM.
OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 8192)
# "-1" keeps the model loaded forever; Ollama wants a number or a duration like "30m".
_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "-1")
OLLAMA_KEEP_ALIVE = int(_keep_alive) if _keep_alive.lstrip("-").isdigit() else _keep_alive
OLLAMA_TIMEOUT = _int("OLLAMA_TIMEOUT", 120)  # seconds per LLM call
# Max concurrent LLM calls across all requests; keep equal to OLLAMA_NUM_PARALLEL on the server.
LLM_CONCURRENCY = _int("LLM_CONCURRENCY", 2)
# Longest side of the image sent to the vision model. The exact text comes from OCR,
# the image is only for layout/logos, so a smaller image means far fewer prompt tokens.
LLM_IMAGE_MAX_SIDE = _int("LLM_IMAGE_MAX_SIDE", 1280)

# ─── Receipt verifier (POST /api/verify-receipt) ─────────────────────────────
# Background re-read of submitted receipts, image only (no OCR). Accuracy over speed.
# Primary model: reads A (full page), B (amount lines), C (zoom on each field).
# qwen3-vl:8b is the thinking variant (Q4_K_M); the "-thinking-q8_0" tags need Ollama > 0.19.
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "qwen3-vl:8b")
# Second model, other Qwen generation: read D (full page). Empty string disables it.
# Off by default: on Ollama 0.19 qwen2.5vl:7b takes 14 GB loaded (Q4 file is 6 GB), so it
# only runs 74% on GPU next to PaddleOCR (4.9 GB). Set a Qwen VL model that fits to enable it.
VERIFY_SECOND_MODEL = os.getenv("VERIFY_SECOND_MODEL", "")
# Max verifications at once, separate from LLM_CONCURRENCY (live scans).
VERIFY_CONCURRENCY = _int("VERIFY_CONCURRENCY", 1)
# Longest side (px) of the full-page image sent to the verifier models.
VERIFY_IMAGE_MAX_SIDE = _int("VERIFY_IMAGE_MAX_SIDE", 1792)
# Qwen2.5-VL in Ollama resizes images above this pixel count; we resize first
# (to multiples of 28) so its absolute box coordinates map 1:1 to our image.
VERIFY_QWEN25_MAX_PIXELS = _int("VERIFY_QWEN25_MAX_PIXELS", 28 * 28 * 1280)
# PDF render scale for the verifier (the live scan uses 1.5). Pages are still capped at 2048 px.
VERIFY_PDF_SCALE = float(os.getenv("VERIFY_PDF_SCALE", "2.5"))
# Context window for verifier calls: image tokens + prompt + thinking + JSON.
VERIFY_NUM_CTX = _int("VERIFY_NUM_CTX", 8192)
# Max generated tokens per call (includes the thinking of a thinking model).
VERIFY_NUM_PREDICT = _int("VERIFY_NUM_PREDICT", 4096)
_verify_keep_alive = os.getenv("VERIFY_KEEP_ALIVE", "2m")
VERIFY_KEEP_ALIVE = int(_verify_keep_alive) if _verify_keep_alive.lstrip("-").isdigit() else _verify_keep_alive
# "swap": unload the scanner model while verifying, reload it afterwards (fits big models).
# "resident": leave the scanner loaded (needs OLLAMA_MAX_LOADED_MODELS=2 and enough VRAM).
VERIFY_GPU_MODE = os.getenv("VERIFY_GPU_MODE", "swap").strip().lower()
VERIFY_TIMEOUT = _int("VERIFY_TIMEOUT", 300)              # seconds per model call
VERIFY_TOTAL_TIMEOUT = _int("VERIFY_TOTAL_TIMEOUT", 1200)  # seconds per receipt
# How long to wait for live scans to finish before answering status "error" (retry later).
VERIFY_WAIT_IDLE_TIMEOUT = _int("VERIFY_WAIT_IDLE_TIMEOUT", 60)
# A verification starts only after no live scan has run for this many seconds.
VERIFY_IDLE_GRACE_S = float(os.getenv("VERIFY_IDLE_GRACE_S", "10"))

# ─── OCR ─────────────────────────────────────────────────────────────────────
OCR_DEVICE = os.getenv("OCR_DEVICE", "cpu")  # "gpu" on the server
OCR_LANG = os.getenv("OCR_LANG", "ro")  # Latin model, keeps Romanian diacritics

# ─── Uploads ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = _int("MAX_FILE_SIZE_MB", 10) * 1024 * 1024
MAX_PAGES = _int("MAX_PAGES", 20)  # per request, after PDFs are split into pages

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
