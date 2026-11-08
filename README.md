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
- **Verifier:** `POST /api/verify-receipt` re-reads a submitted receipt in the background (vision model reading the image blind, OCR as an independent witness) and returns plain JSON. See [Receipt verifier](#-receipt-verifier-post-apiverify-receipt).
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

The verifier's `VERIFY_*` variables are listed in [Receipt verifier](#-receipt-verifier-post-apiverify-receipt).

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

---

## 🔎 Receipt verifier (`POST /api/verify-receipt`)

Step ④ of the Spendle flow. After the employee submits a report, the Spendle backend queues
each receipt and calls this endpoint in the background. A vision model reads the receipt **from the
image only**: it never gets OCR text and never the values the employee declared. PaddleOCR runs
separately as an **independent witness** for the total and the currency. It reports what is
printed and how sure it is. The comparison with the declared values (steps ⑤–⑥) happens in
the Spendle backend with plain rules. The verifier never approves or judges anything.

### Contract
Request: `multipart/form-data` with one file in field `file` (PDF, PNG, JPEG, WEBP). Same limits
as the scan endpoints (`MAX_FILE_SIZE_MB`, `MAX_PAGES`).

```json
{
  "status": "ok",
  "model": "qwen3-vl:8b",
  "models": ["qwen3-vl:8b"],
  "duration_ms": 18400,
  "pages": [{ "page": 1, "width": 1240, "height": 1754 }],
  "fields": {
    "total_amount":    { "value": 120.0, "raw": "120,00", "confidence": "high", "page": 1, "bbox": [412, 1880, 590, 1932] },
    "currency":        { "value": "RON", "raw": "LEI", "confidence": "high", "page": 1, "bbox": [600, 1880, 660, 1932] },
    "receipt_date":    { "value": "2026-11-03", "raw": "03.11.2026", "confidence": "medium", "page": 1, "bbox": null },
    "document_number": { "value": "F-0042", "raw": "Nr. F-0042", "confidence": "low", "page": 1, "bbox": [100, 360, 300, 400] }
  },
  "notes": "document_number: reads disagree (A=F-0042, B=F-0047, C=F-0042, D=F-0047)"
}
```

| `status` | Meaning | HTTP |
|---|---|---|
| `ok` | Read; check each field's `confidence`. | 200 |
| `unreadable` | The reads disagree on the total, or no total was found. "I can't tell" is a valid answer. | 200 |
| `not_a_receipt` | Both models say this is not a receipt or invoice. | 200 |
| `multiple_receipts` | More than one receipt in the file (by count, or different document numbers on different pages). | 200 |
| `error` | Temporary: GPU busy, preempted by a live scan, model missing, timeout. **Retry later.** | 503 + `Retry-After` |

An unsupported or corrupt file is a client error: HTTP 400 like the other endpoints. Don't retry it.

- `value` is normalized:
  - amounts are numbers (`1.234,56` and `1,234.56` become `1234.56`);
  - currencies are ISO 4217 codes (`LEI`/`lei`/`RON` become `RON`, `€` becomes `EUR`);
  - dates are `YYYY-MM-DD`, read day-first as on Romanian receipts.
- `raw` is the text exactly as read.
- A field that is not on the document is `{"value": null, "raw": null, "confidence": "none", "page": null, "bbox": null}`.
- `total_amount` is the final amount paid, VAT included. It is never the subtotal, the VAT line, the cash handed over or the change.
- `bbox` is `[x0, y0, x1, y1]` in pixels of the page image described in `pages`, or `null`. It is never `[]`. To highlight it on a page rendered at another size, scale by `rendered_width / width`.
- `notes` is a short reason when the status isn't `ok` or a field isn't `high`; otherwise `null`.
- Expect minutes per receipt, not seconds. Set the caller's HTTP timeout to at least `VERIFY_WAIT_IDLE_TIMEOUT + VERIFY_TOTAL_TIMEOUT` (about 21 minutes with the defaults).

### How it reads

| Read | Model | Input | Asks for |
|---|---|---|---|
| A | `VERIFY_MODEL` | every page, longest side `VERIFY_IMAGE_MAX_SIDE` | all four fields, each with a box; document type; receipt count |
| B | `VERIFY_MODEL` | the total's page at 0.75× that size, other prompt | every amount line with its label (subtotal, VAT, total, cash, change); code picks the final total |
| C | `VERIFY_MODEL` | a full-width strip around the total, currency and date A located, enlarged up to 2× | that one value again, at a higher resolution (not done for the document number, which is informative only) |
| D | `VERIFY_SECOND_MODEL` (another Qwen generation) | the pages where A found fields | the same as A |

Confidence is measured from agreement, never self-reported:

| Confidence | When |
|---|---|
| `high` | every read that ran agrees after normalization, and **two reads put a box on the same spot** |
| `medium` | they agree but there is no usable box, a read did not find the value, or they differ only in formatting (`F-0042` vs `F42`) |
| `low` | they disagree, or only one read found the value. The majority value is returned and `notes` lists every read. |

**OCR witness (total and currency).** `VERIFY_OCR_WITNESS=true` (default). PaddleOCR reads the
total's page once, as it is (no crop, so its boxes are page pixels), on the instance the live scans
use (~1–2 s on GPU). The model never sees this text. Plain code finds the **TOTAL row** (not
SUBTOTAL / TOTAL TVA / TAXE; `TVA` OCR'd as `TUA` included), the **payment rows** (CARD, NUMERAR,
CASH, CARTE CREDIT) minus **REST**, and the **currency on the TOTAL row** ("TOTAL LEI"). Only amounts
with two decimals count, so card numbers and codes never match.

| Model total vs OCR | Result |
|---|---|
| OCR reads the same amount on the TOTAL row | `high`, with OCR's exact box. A payment line that disagrees only lowers it to `medium` with a note (payment lines get misread too). |
| A minority model read equals OCR's TOTAL row **and** the payment | that value, `medium`, with a note |
| OCR reads a different amount on the TOTAL row | `low` → status `unreadable` |
| OCR has no TOTAL row, payment matches / differs | payment decides: `medium` / `low` |
| OCR finds neither TOTAL nor payment (faded receipt) | fallback: the model's own read of the payment lines (read B); a different *payment − change* makes the total `low` |

Currency: confirmed by the currency on the TOTAL row → `high` with its box; the model found none but
the TOTAL row prints one → filled in, `medium`; different currencies → `low`. A currency found
anywhere else on the page is ignored. On the 10 benchmark receipts, real PaddleOCR found the right
TOTAL and payment on 9 (the 10th is the faded one) and "LEI" on every Rompetrol receipt.

**Boxes need two reads.** A box is returned only when two reads (e.g. A and B) put it on the same
spot (≥30% overlap). On the first benchmark about a third of the single-read boxes were on the wrong
line or off the receipt.

**Document number is informative.** It is read (A, B) and returned, but it never changes the
status. Receipts carry several numbers (fiscal receipt, card slip `RECEIPT NO`, transaction), so the
Spendle backend should not raise "needs a look" on it.

**Currency is special.** Many receipts (e.g. Romanian fuel receipts) print no currency at all:
- No read finding a currency gives `"confidence": "none"`, and the status stays `ok`.
- One read finding it while the others don't gives `medium`, not `low`.
- Currency is `low` only when the reads found **different** currencies (e.g. `RON` vs `EUR`).
- The prompts tell the models never to deduce the currency from the country or the language.

Recommended Spendle rule: currency `none` means nothing to compare, so **no "needs a look"**. Only
flag it when the verifier reads a different currency than the declared one with `high`/`medium`.

Boxes come from the models' grounding:
- Qwen2.5-VL answers in pixels of the image it received. The verifier sends sizes in multiples of 28 within `VERIFY_QWEN25_MAX_PIXELS`, so Ollama doesn't resize again.
- Qwen3-VL answers on a 0–1000 grid.
- Both are scaled to the page size. A box outside the image, with zero size, or covering most of the page becomes `null`.

### Live scans keep priority
- Each verification has its own limit, `VERIFY_CONCURRENCY` (default 1), separate from `LLM_CONCURRENCY`.
- It starts only after no scan has run for `VERIFY_IDLE_GRACE_S`. If scans keep the GPU busy for longer than `VERIFY_WAIT_IDLE_TIMEOUT`, it answers `error` ("GPU busy") straight away.
- **Swap mode** (`VERIFY_GPU_MODE=swap`, the default):
  - the live-scan model is unloaded;
  - the verifier models are loaded one after the other, and each is checked in `/api/ps` to be **100% on GPU** (if not, the answer is `error` with the reason);
  - afterwards the verifier models are unloaded and the live-scan model is reloaded with its own `num_ctx`.
  Swapped models stay in the RAM page cache, so a reload takes seconds.
- **Preemption:** if a scan starts mid-verification, the verifier's Ollama call is cancelled at once, the live-scan model is reloaded and the answer is `error` ("preempted"). A scan waits at most for that reload, never for a verification.
- CPU work on this path: one PDF render (at `VERIFY_PDF_SCALE`), a few image resizes, and the host side of one OCR pass (no OpenCV preprocessing). The OCR pass shares the live scans' PaddleOCR lock; a scan arriving meanwhile waits for it (~1–2 s).
- Single-process assumption: the scan tracking lives in memory, so keep one uvicorn worker (already required for OCR).

### Ollama models to pull
```bash
ollama pull qwen3-vl:8b          # VERIFY_MODEL: thinking variant, Q4_K_M
# VERIFY_SECOND_MODEL is off: qwen2.5vl:7b needs 14 GB loaded on Ollama 0.19 and doesn't fit.
```
On the server (Ollama 0.19.0) the `qwen3-vl:8b-thinking-*` and `-q8_0` tags fail with
"requires a newer version of Ollama". Use them only after an Ollama upgrade (a change to
production: test live scans afterwards).
Check the exact tags in the Ollama library before pulling. Qwen VL models above 8B (`qwen3-vl:30b`, `qwen3-vl:32b`, `qwen2.5vl:32b`) need about 19 GB or more even at Q4. They can't run 100% on a 16 GB GPU, so they are excluded.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `VERIFY_MODEL` | `qwen3-vl:8b` | Primary model (reads A, B, C). On Ollama this tag is the **thinking** variant (Q4_K_M); thinking is turned on automatically when the model supports it. |
| `VERIFY_SECOND_MODEL` | *(empty: off)* | Read D. Off by default: on Ollama 0.19 `qwen2.5vl:7b` takes **14 GB** once loaded (6 GB file) and runs only 74% on GPU next to PaddleOCR; `qwen2.5vl:7b-q8_0` 60%. Set a Qwen VL model that fits 100% on GPU to enable it. |
| `VERIFY_CONCURRENCY` | `1` | Verifications at once. |
| `VERIFY_IMAGE_MAX_SIDE` | `1792` | Longest side of the full-page image (benchmark it: more pixels means more image tokens and VRAM). |
| `VERIFY_QWEN25_MAX_PIXELS` | `1003520` | Qwen2.5-VL image pixel budget (28·28·1280). |
| `VERIFY_PDF_SCALE` | `2.5` | PDF render scale for the verifier (pages are still capped at 2048 px). |
| `VERIFY_NUM_CTX` | `12288` | Context per verifier call: image tokens (~2,300 at 1792 px) + prompt + thinking + JSON. |
| `VERIFY_NUM_PREDICT` | `8192` | Max generated tokens per call, thinking included. With 4096, a third of the reads ran out of tokens while thinking. |
| `VERIFY_THINK_TEMPERATURE` / `_TOP_P` / `_TOP_K` | `0.6` / `0.95` / `20` | Sampling for thinking models (Qwen's recommendation). Greedy decoding (temperature 0) makes them loop until the token limit. Non-thinking calls use temperature 0. |
| `VERIFY_READ_RETRIES` | `1` | Extra attempts for a read that returns no valid JSON. |
| `VERIFY_KEEP_ALIVE` | `2m` | Keep-alive for verifier calls (swap mode unloads them explicitly anyway). |
| `VERIFY_GPU_MODE` | `swap` | `swap`, or `resident` (only if the measurements show the scanner and the verifier fit together). |
| `VERIFY_TIMEOUT` | `300` | Seconds per model call. |
| `VERIFY_TOTAL_TIMEOUT` | `1200` | Seconds per receipt. |
| `VERIFY_WAIT_IDLE_TIMEOUT` | `60` | Seconds to wait for live scans to end before answering `error`. |
| `VERIFY_IDLE_GRACE_S` | `10` | Quiet seconds after the last scan before a verification starts. |
| `VERIFY_OCR_WITNESS` | `true` | Run PaddleOCR once on the total's page as an independent witness for the total and the currency (exact boxes). |

### VRAM budget (16 GB RTX 5060 Ti)

Measured on 2026-11-07, idle, with the service warm (`nvidia-smi`, `ollama ps`):

| Resident | Measured | Note |
|---|---|---|
| PaddleOCR (service process) | **4,944 MiB** | `auto_growth` keeps the peak it reached; far above the 1–2 GB first assumed |
| Live scan: `qwen3-vl:8b-instruct` (Q4), context 8192, `OLLAMA_NUM_PARALLEL=2` | **7,012 MiB** (`ollama ps`: 7.5 GB, 100% GPU) | |
| Total used / available | **11,970 / 16,311 MiB** | ~4.3 GB free |
| Verifier, swap mode (scanner unloaded, PaddleOCR stays) | Ollama reports **10.2 GiB available** | `qwen3-vl:8b` (thinking, Q4): **7.0 GiB**, 100% GPU. `qwen2.5vl:7b` (Q4): **14 GB** loaded, 74% GPU, rejected. `qwen2.5vl:7b-q8_0`: 60% GPU, rejected. |

Conclusions:
- **Resident mode is out:** the scanner, PaddleOCR and any ~7B verifier don't fit together (7.0 + 4.9 + 7+ GB > 16 GB). Swap mode it is.
- **Q8 verifier models may not reach 100% GPU** next to PaddleOCR's 4.9 GB. The verifier checks `/api/ps` and refuses with a clear `error` if not. Fallback: the Q4 tags (`qwen3-vl:8b-thinking`, `qwen2.5vl:7b`), or a lower `VERIFY_NUM_CTX`.
- Still to measure: the peak during a live scan, and the verifier peak (the benchmark reports it).

### Recommended Ollama settings
Current `ollama.service` environment: `OLLAMA_HOST=0.0.0.0:11434 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=2 OLLAMA_CONTEXT_LENGTH=8192`.

- **`OLLAMA_HOST=0.0.0.0` exposes Ollama to the whole network without authentication.** Anyone on the LAN can run, pull or delete models. The scanning service talks to it via `localhost`. Unless another machine needs Ollama directly, bind it to `127.0.0.1:11434`.
- `OLLAMA_KEEP_ALIVE` is not set on the server. That's fine: the service sends `keep_alive=-1` per request (`ollama ps` shows `Forever`), and the verifier passes its own value and unloads its models explicitly.
- `OLLAMA_MAX_LOADED_MODELS`: the default is fine in swap mode, because only one model is loaded at a time.
- `OLLAMA_NUM_PARALLEL=2` applies to every model, so the verifier's KV cache is also allocated ×2. If VRAM is tight, lower `VERIFY_NUM_CTX` rather than `OLLAMA_NUM_PARALLEL`, because live scans need the parallelism.

### CPU: only 1 CPU visible

Measured on 2026-11-07:

| Check | Result |
|---|---|
| `nproc`, `lscpu` | 1 CPU on-line (`On-line CPU(s) list: 0`), model AMD Ryzen 9 7900 12-Core |
| VM? | No: `systemd-detect-virt` = none, no hypervisor flag. Bare metal. |
| cgroup / systemd | No limit: `cpu.max` = `max`, `AllowedCPUs=` and `CPUAffinity=` empty for `scanning-tool` and `ollama` |
| Kernel command line | **`acpi=off`** |

**Cause: the kernel boot option `acpi=off`.** On modern x86 machines, the extra CPU cores are
discovered through ACPI tables. With ACPI disabled, Linux brings up only the boot CPU, so 11 of
the 12 cores (23 of 24 threads) are unused. It isn't a cgroup, systemd or VM limit. `acpi=off`
also disables ACPI power management and thermal reporting.

Recommendation (a person decides; nothing is changed by this repo):
1. Find out why `acpi=off` was added. It is usually a workaround for a boot or installation problem.
2. Test once without it: at the GRUB menu press `e`, remove `acpi=off` from the `linux` line, boot with `Ctrl+X`. Nothing is saved, so the next reboot is back to the current settings. Have console/IPMI access in case it doesn't boot.
3. If it boots fine and `nproc` shows 24, remove it for good from `GRUB_CMDLINE_LINUX` / `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `update-grub` and reboot.
4. If the original problem returns, try narrower options instead (for example `pci=noacpi` or `acpi=noirq`), which keep CPU discovery working.

With one CPU, PDF rendering, OCR preprocessing (OpenCV), uvicorn, PaddleOCR's host side and
Ollama's image preprocessing all queue on the same core. Every live scan and verification pays for it.

### Benchmark
On the GPU server, with the service running:
```bash
# truth.csv: file,total_amount,currency,receipt_date,document_number   (empty = not printed)
uv run python scripts/benchmark_verify.py --receipts data/receipts --truth data/truth.csv \
  --models "qwen3-vl:8b+qwen2.5vl:7b,qwen3-vl:8b" \
  --out bench.csv [--bbox-check] [--max-side 1792]

# impact on live scans (uses the service's configured VERIFY_MODEL)
uv run python scripts/benchmark_verify.py --receipts data/receipts --truth data/truth.csv --mode live \
  --live-scans 10 --scan-interval 45
```
Per field and model config, the summary reports:
- first-pass vs verifier accuracy;
- mistakes caught (first pass wrong, verifier right);
- mistakes flagged;
- false alarms;
- same-way wrong;
- accuracy per confidence level;
- bbox hit rate (`--bbox-check` runs OCR on each box offline, on the CPU by default, so VRAM stays free).

Per config it also reports:
- average and p95 duration;
- scanner reload time;
- peak VRAM;
- the share of each model on the GPU;
- wall time vs CPU time.

The CSV has one row per receipt and config.

### Tests
```bash
uv run pytest
```
They cover normalization, confidence rules, bbox scaling, status rules, the live-scan gate and
the endpoint contract, all with mocked model responses.
