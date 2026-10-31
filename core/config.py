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

# ─── OCR ─────────────────────────────────────────────────────────────────────
OCR_DEVICE = os.getenv("OCR_DEVICE", "cpu")  # "gpu" on the server
OCR_LANG = os.getenv("OCR_LANG", "ro")  # Latin model, keeps Romanian diacritics

# ─── Uploads ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = _int("MAX_FILE_SIZE_MB", 10) * 1024 * 1024
MAX_PAGES = _int("MAX_PAGES", 20)  # per request, after PDFs are split into pages

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
