#!/usr/bin/env python
"""Benchmark the receipt verifier against ground truth, before choosing its models.

Run on the GPU server, from the repo root, with the service running:

  uv run python scripts/benchmark_verify.py \\
      --receipts data/receipts --truth data/truth.csv \\
      --models "qwen3-vl:8b+qwen2.5vl:7b-q8_0,qwen3-vl:8b+qwen2.5vl:7b" \\
      --out bench.csv

truth.csv columns: file,total_amount,currency,receipt_date,document_number
(an empty cell means "not printed on the document").

Modes (--mode):
  accuracy  (default)
    1. first pass: every receipt through the service's POST /api/web-report-stream;
    2. verifier: every receipt through graph.verifier in this process, once per model config.
       A config is "primary+second" ("+second" optional = no read D). Each receipt follows the
       production flow: unload the scanner model, verify, reload the scanner model.
    3. optional (--bbox-check): OCR each returned box to see whether it contains the value.
  live
    Live scans (POST /api/web-report-stream) alone, then again while verifications run
    through the service's POST /api/verify-receipt (real gate + preemption). Compares scan
    latency and counts how verifications ended. Uses the service's configured VERIFY_MODEL.
  both
    accuracy, then live.

Per field and model config the summary reports: accuracy (first pass / verifier), mistakes
caught (first pass wrong, verifier right), mistakes flagged, false alarms, same-way wrong,
accuracy per confidence level, bbox hit rate. Per config: average / p95 duration, model swap
time, peak VRAM, share of each model on the GPU, wall time vs CPU time.

"Flagged" assumes the Spendle rule: a field needs a look when the verifier status is not ok,
its confidence is low, or its high/medium value differs from the declared one (here: the
first-pass value, i.e. the employee did not correct it).
"""
import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from core.config import (  # noqa: E402
    OCR_LANG, OLLAMA_BASE_URL, VERIFY_IMAGE_MAX_SIDE, VERIFY_PDF_SCALE, VERIFY_TOTAL_TIMEOUT,
)
from graph.normalize import (  # noqa: E402
    FIELDS, _NUMBER_RE, document_number_key, normalize_amount, normalize_currency, normalize_date,
    normalize_document_number,
)
from graph.verifier import decode_page, verify_pages  # noqa: E402
from graph.verifier_ollama import OllamaBackend, VerifierError  # noqa: E402
from tools.parser import parse_document_to_images  # noqa: E402

CONTENT_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
                 ".jpeg": "image/jpeg", ".webp": "image/webp"}
TRUTH_NORMALIZERS = {"total_amount": normalize_amount, "currency": normalize_currency,
                     "receipt_date": normalize_date, "document_number": normalize_document_number}


# ─── Comparisons ─────────────────────────────────────────────────────────────

def same(fld: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if fld == "total_amount":
        return abs(float(a) - float(b)) < 0.005
    if fld == "document_number":
        return document_number_key(a) == document_number_key(b)
    return a == b


def fp_same(fld: str, fp_value: Any, value: Any) -> bool:
    """First-pass value vs another value. The first pass has no separate document number:
    it is inside `invoice_number_date` ("F-0042 / 03.11.2026"), so look for it there."""
    if fld != "document_number":
        return same(fld, fp_value, value)
    if value is None or not fp_value:
        return value is None and not fp_value
    key = document_number_key(value)
    return bool(key) and key in (document_number_key(fp_value) or "")


def spendle_flag(status: str, ver_field: dict, fld: str, declared: Any) -> bool:
    if status != "ok" or ver_field["confidence"] == "low":
        return True
    if ver_field["confidence"] in ("high", "medium"):
        return not fp_same(fld, declared, ver_field["value"])
    return False  # "none": nothing printed, nothing to compare


def p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]


# ─── System probes (Linux; return None elsewhere) ────────────────────────────

_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def cpu_snapshot() -> Optional[tuple[float, float]]:
    """(system-wide busy CPU seconds, CPU seconds of all ollama processes)."""
    try:
        parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        vals = [int(v) for v in parts]
        busy = sum(vals) - vals[3] - (vals[4] if len(vals) > 4 else 0)
        ollama = 0
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                stat = (d / "stat").read_text()
            except OSError:
                continue
            comm = stat[stat.find("(") + 1: stat.rfind(")")]
            if comm.startswith("ollama"):
                rest = stat[stat.rfind(")") + 2:].split()
                ollama += int(rest[11]) + int(rest[12])  # utime, stime
        return busy / _CLK, ollama / _CLK
    except (OSError, ValueError, IndexError):
        return None


class GpuSampler(threading.Thread):
    """Polls nvidia-smi (VRAM used) and Ollama /api/ps (share of each model on the GPU)."""

    def __init__(self, ollama_url: str, interval: float = 0.5):
        super().__init__(daemon=True)
        self.ollama_url, self.interval = ollama_url, interval
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.peak_vram_mib: Optional[int] = None
            self.gpu_share: dict[str, float] = {}

    def snapshot(self) -> tuple[Optional[int], dict[str, float]]:
        with self._lock:
            return self.peak_vram_mib, dict(self.gpu_share)

    def stop(self):
        self._halt.set()

    def run(self):
        with httpx.Client(timeout=2) as client:
            while not self._halt.is_set():
                vram = None
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=2).stdout
                    vram = sum(int(v) for v in out.split())
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                shares = {}
                try:
                    for m in client.get(f"{self.ollama_url}/api/ps").json().get("models", []):
                        if m.get("size"):
                            shares[m.get("name") or m.get("model")] = m.get("size_vram", 0) / m["size"]
                except (httpx.HTTPError, ValueError):
                    pass
                with self._lock:
                    if vram is not None:
                        self.peak_vram_mib = max(self.peak_vram_mib or 0, vram)
                    for name, share in shares.items():
                        self.gpu_share[name] = min(share, self.gpu_share.get(name, 1.0))
                self._halt.wait(self.interval)


# ─── Inputs ──────────────────────────────────────────────────────────────────

@dataclass
class Receipt:
    file: str
    path: Path
    content_type: str
    truth: dict[str, Any]
    fp: dict[str, Any] = field(default_factory=dict)        # first-pass values
    fp_ms: Optional[int] = None
    fp_error: Optional[str] = None
    pages: Optional[list[str]] = None                       # verifier page images (b64)


def load_receipts(folder: Path, truth_csv: Path) -> list[Receipt]:
    receipts = []
    with truth_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = folder / row["file"]
            ctype = CONTENT_TYPES.get(path.suffix.lower())
            if not path.exists() or not ctype:
                print(f"skip {row['file']}: {'missing' if not path.exists() else 'unsupported type'}")
                continue
            truth = {fld: TRUTH_NORMALIZERS[fld]((row.get(fld) or "").strip() or None) for fld in FIELDS}
            receipts.append(Receipt(row["file"], path, ctype, truth))
    return receipts


# ─── Service calls ───────────────────────────────────────────────────────────

async def web_scan(client: httpx.AsyncClient, service: str, rec: Receipt) -> tuple[float, Optional[dict], Optional[str]]:
    """POST /api/web-report-stream. Returns (seconds to the final event, done data, error)."""
    start = time.monotonic()
    files = {"files": (rec.path.name, rec.path.read_bytes(), rec.content_type)}
    async with client.stream("POST", f"{service}/api/web-report-stream", files=files) as resp:
        if resp.status_code != 200:
            return time.monotonic() - start, None, f"HTTP {resp.status_code}"
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["step"] == "done":
                return time.monotonic() - start, event.get("data", {}), None
            if event["step"] == "error":
                return time.monotonic() - start, None, event.get("message")
    return time.monotonic() - start, None, "stream ended without done"


def first_pass_values(done: Optional[dict]) -> dict[str, Any]:
    items = (done or {}).get("items") or []
    if not items:
        return {fld: None for fld in FIELDS}
    it = items[0]
    amount = it.get("expense_amount")
    return {
        "total_amount": round(float(amount), 2) if isinstance(amount, (int, float)) else normalize_amount(str(amount)),
        "currency": normalize_currency(it.get("currency")),
        "receipt_date": normalize_date(it.get("receipt_date")),
        "document_number": (it.get("invoice_number_date") or "").strip() or None,
    }


async def run_first_pass(service: str, receipts: list[Receipt]) -> None:
    print(f"\n== First pass ({len(receipts)} receipts via {service}/api/web-report-stream)")
    async with httpx.AsyncClient(timeout=httpx.Timeout(600)) as client:
        for rec in receipts:
            try:
                secs, done, err = await web_scan(client, service, rec)
            except httpx.HTTPError as e:
                secs, done, err = 0.0, None, f"{type(e).__name__}: {e}"
            rec.fp, rec.fp_ms, rec.fp_error = first_pass_values(done), int(secs * 1000), err
            print(f"  {rec.file}: {secs:.1f}s" + (f" ERROR {err}" if err else ""))


# ─── Verifier runs ───────────────────────────────────────────────────────────

@dataclass
class ConfigRun:
    config: str
    rows: list[dict] = field(default_factory=list)
    peak_vram_mib: Optional[int] = None
    gpu_share: dict[str, float] = field(default_factory=dict)


async def run_verifier_config(config: str, receipts: list[Receipt], sampler: Optional[GpuSampler],
                              max_side: int) -> ConfigRun:
    primary, _, second = config.partition("+")
    print(f"\n== Verifier {config}")
    run = ConfigRun(config)
    if sampler:
        sampler.reset()
    for rec in receipts:
        if rec.pages is None:
            rec.pages = await asyncio.to_thread(
                parse_document_to_images, rec.path.read_bytes(), rec.content_type, VERIFY_PDF_SCALE)
        backend = OllamaBackend()
        cpu0, t0 = cpu_snapshot(), time.monotonic()
        status, body, err = "error", None, None
        try:
            await backend.begin()
            outcome = await asyncio.wait_for(
                verify_pages(rec.pages, backend, primary, second or None, max_side), VERIFY_TOTAL_TIMEOUT)
            body, status = outcome.response, outcome.response["status"]
        except (VerifierError, asyncio.TimeoutError) as e:
            err = str(e) or type(e).__name__
        t1, cpu1 = time.monotonic(), cpu_snapshot()
        await backend.end()
        swap_ms = int((time.monotonic() - t1) * 1000)

        row: dict[str, Any] = {
            "file": rec.file, "config": config, "status": status,
            "duration_ms": int((t1 - t0) * 1000), "swap_ms": swap_ms, "wall_s": round(t1 - t0, 2),
            "cpu_sys_s": round(cpu1[0] - cpu0[0], 2) if cpu0 and cpu1 else None,
            "cpu_ollama_s": round(cpu1[1] - cpu0[1], 2) if cpu0 and cpu1 else None,
            "fp_ms": rec.fp_ms, "fp_error": rec.fp_error,
            "notes": (body or {}).get("notes") or err,
        }
        for fld in FIELDS:
            vf = (body or {}).get("fields", {}).get(fld) or {"value": None, "confidence": "none", "bbox": None, "page": None}
            truth, fp = rec.truth[fld], rec.fp.get(fld)
            row.update({
                f"truth_{fld}": truth, f"fp_{fld}": fp,
                f"ver_{fld}": vf["value"], f"conf_{fld}": vf["confidence"],
                f"bbox_{fld}": json.dumps(vf["bbox"]) if vf["bbox"] else "", f"page_{fld}": vf["page"],
                f"fp_ok_{fld}": fp_same(fld, fp, truth) if rec.fp_error is None else False,
                f"ver_ok_{fld}": same(fld, vf["value"], truth) if body else None,
                f"flag_{fld}": spendle_flag(status, vf, fld, fp) if body else None,
                f"bbox_hit_{fld}": None,
            })
        run.rows.append(row)
        print(f"  {rec.file}: {status} {row['duration_ms'] / 1000:.1f}s (swap back {swap_ms / 1000:.1f}s)"
              + (f" — {err}" if err else ""))
    if sampler:
        run.peak_vram_mib, run.gpu_share = sampler.snapshot()
    return run


# ─── Optional bbox check (offline OCR of each returned box) ──────────────────

def value_in_text(fld: str, text: str, value: Any) -> bool:
    if fld == "total_amount":
        return any(normalize_amount(m) is not None and abs(normalize_amount(m) - value) < 0.005
                   for m in _NUMBER_RE.findall(text))
    if fld == "currency":
        return normalize_currency(text) == value
    if fld == "receipt_date":
        return normalize_date(text) == value
    key = document_number_key(value)
    return bool(key) and key in (document_number_key(normalize_document_number(text)) or "")


def run_bbox_check(runs: list[ConfigRun], receipts: list[Receipt], device: str) -> None:
    import numpy as np
    from paddleocr import PaddleOCR
    print(f"\n== Bbox check (PaddleOCR on each box, device={device})")
    ocr = PaddleOCR(lang=OCR_LANG, device=device, use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    by_file = {r.file: r for r in receipts}
    images: dict[tuple[str, int], Any] = {}
    for run in runs:
        for row in run.rows:
            for fld in FIELDS:
                if not row[f"bbox_{fld}"] or row[f"ver_{fld}"] is None:
                    continue
                rec, page = by_file[row["file"]], row[f"page_{fld}"] - 1
                if (rec.file, page) not in images:
                    images[(rec.file, page)] = decode_page(rec.pages[page])
                img = images[(rec.file, page)]
                x0, y0, x1, y1 = json.loads(row[f"bbox_{fld}"])
                crop = img.crop((max(0, x0 - 4), max(0, y0 - 4), min(img.width, x1 + 4), min(img.height, y1 + 4)))
                if crop.height < 48:
                    s = 48 / crop.height
                    crop = crop.resize((max(1, int(crop.width * s)), 48))
                res = ocr.predict(np.array(crop)[:, :, ::-1])
                text = " ".join(res[0]["rec_texts"]) if res else ""
                row[f"bbox_hit_{fld}"] = value_in_text(fld, text, row[f"ver_{fld}"])


# ─── Report ──────────────────────────────────────────────────────────────────

def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:5.1f}% ({num}/{den})" if den else "  n/a"


def summarize(runs: list[ConfigRun]) -> None:
    for run in runs:
        ok_rows = [r for r in run.rows if r["status"] != "error"]
        print(f"\n### {run.config}")
        statuses: dict[str, int] = {}
        for r in run.rows:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        durations = [r["duration_ms"] / 1000 for r in ok_rows]
        swaps = [r["swap_ms"] / 1000 for r in run.rows]
        print(f"status: {statuses}")
        if durations:
            print(f"duration: avg {statistics.mean(durations):.1f}s, p95 {p95(durations):.1f}s; "
                  f"scanner reload after each: avg {statistics.mean(swaps):.1f}s")
        print(f"peak VRAM: {run.peak_vram_mib} MiB; share on GPU (min seen): "
              + (", ".join(f"{m} {s:.0%}" for m, s in run.gpu_share.items()) or "n/a"))
        cpu = [(r["wall_s"], r["cpu_sys_s"], r["cpu_ollama_s"]) for r in ok_rows if r["cpu_sys_s"] is not None]
        if cpu:
            wall, sys_s, oll = (sum(c[i] for c in cpu) for i in range(3))
            print(f"wall {wall:.0f}s vs CPU {sys_s:.0f}s system-wide ({sys_s / wall:.2f} cores busy on average), "
                  f"{oll:.0f}s in ollama processes")

        header = f"{'field':16} {'first pass':>16} {'verifier':>16} {'caught':>14} {'flagged':>16} " \
                 f"{'false alarm':>16} {'same wrong':>14} {'bbox hit':>16}  accuracy by confidence"
        print(header)
        for fld in FIELDS:
            rows = [r for r in ok_rows if r[f"ver_ok_{fld}"] is not None]
            fp_wrong = [r for r in rows if not r[f"fp_ok_{fld}"]]
            fp_right = [r for r in rows if r[f"fp_ok_{fld}"]]
            caught = sum(1 for r in fp_wrong if r[f"ver_ok_{fld}"])
            flagged = sum(1 for r in fp_wrong if r[f"flag_{fld}"])
            false_alarm = sum(1 for r in fp_right if r[f"flag_{fld}"])
            same_wrong = sum(1 for r in fp_wrong if not r[f"ver_ok_{fld}"]
                             and r[f"ver_{fld}"] is not None and fp_same(fld, r[f"fp_{fld}"], r[f"ver_{fld}"]))
            checked = [r for r in rows if r[f"bbox_hit_{fld}"] is not None]
            by_conf = []
            for conf in ("high", "medium", "low", "none"):
                c_rows = [r for r in rows if r[f"conf_{fld}"] == conf]
                if c_rows:
                    by_conf.append(f"{conf} {sum(r[f'ver_ok_{fld}'] for r in c_rows)}/{len(c_rows)}")
            print(f"{fld:16} {_pct(sum(r[f'fp_ok_{fld}'] for r in rows), len(rows)):>16} "
                  f"{_pct(sum(r[f'ver_ok_{fld}'] for r in rows), len(rows)):>16} "
                  f"{_pct(caught, len(fp_wrong)):>14} {_pct(flagged, len(fp_wrong)):>16} "
                  f"{_pct(false_alarm, len(fp_right)):>16} {_pct(same_wrong, len(fp_wrong)):>14} "
                  f"{_pct(sum(r[f'bbox_hit_{fld}'] for r in checked), len(checked)):>16}  {', '.join(by_conf)}")


def write_csv(path: Path, runs: list[ConfigRun]) -> None:
    rows = [r for run in runs for r in run.rows]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-receipt results: {path}")


# ─── Live mode ───────────────────────────────────────────────────────────────

async def run_live(service: str, receipts: list[Receipt], n_scans: int, interval: float) -> None:
    timeout = httpx.Timeout(VERIFY_TOTAL_TIMEOUT + 300)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def scans(label: str) -> list[float]:
            latencies = []
            for i in range(n_scans):
                try:
                    secs, _, err = await web_scan(client, service, receipts[i % len(receipts)])
                except httpx.HTTPError as e:
                    secs, err = float("nan"), f"{type(e).__name__}: {e}"
                latencies.append(secs)
                print(f"  {label} scan {i + 1}/{n_scans}: {secs:.1f}s" + (f" ERROR {err}" if err else ""))
                if i < n_scans - 1:
                    await asyncio.sleep(interval)
            return latencies

        print(f"\n== Live mode: {n_scans} scans, {interval:.0f}s apart. Baseline (no verifications)")
        baseline = await scans("baseline")

        outcomes: list[tuple[str, str, float]] = []
        stop = asyncio.Event()

        async def verify_loop():
            i = 0
            while not stop.is_set():
                rec = receipts[i % len(receipts)]
                i += 1
                t0 = time.monotonic()
                files = {"file": (rec.path.name, rec.path.read_bytes(), rec.content_type)}
                try:
                    resp = await client.post(f"{service}/api/verify-receipt", files=files)
                    body = resp.json()
                    outcomes.append((body.get("status", f"HTTP {resp.status_code}"), body.get("notes") or "",
                                     time.monotonic() - t0))
                except (httpx.HTTPError, ValueError) as e:
                    outcomes.append(("client error", str(e), time.monotonic() - t0))
                    await asyncio.sleep(5)

        print("\n== Same scans while verifications run through /api/verify-receipt")
        verifier = asyncio.create_task(verify_loop())
        await asyncio.sleep(interval)  # let the first verification get going
        loaded = await scans("with verifier")
        stop.set()
        verifier.cancel()

    def line(name, xs):
        xs = [x for x in xs if x == x] or [float("nan")]  # drop failed scans (NaN)
        return f"{name:15} p50 {statistics.median(xs):6.1f}s  p95 {p95(xs):6.1f}s  max {max(xs):6.1f}s"
    print("\n### Live scan latency (POST /api/web-report-stream, time to final event)")
    print(line("baseline", baseline))
    print(line("with verifier", loaded))
    counts: dict[str, int] = {}
    for status, notes, _ in outcomes:
        key = status + (" (preempted)" if "Preempted" in notes else " (gpu busy)" if "busy" in notes else "")
        counts[key] = counts.get(key, 0) + 1
    print(f"verifications: {counts}")
    done = [secs for status, _, secs in outcomes if status != "error"]
    if done:
        print(f"completed verifications: avg {statistics.mean(done):.0f}s")


# ─── Main ────────────────────────────────────────────────────────────────────

def visible_cpus() -> str:
    try:
        return f"{len(os.sched_getaffinity(0))} visible to this process, {os.cpu_count()} in the system"
    except AttributeError:
        return f"{os.cpu_count()} in the system"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipts", type=Path, required=True, help="folder with the receipt files")
    ap.add_argument("--truth", type=Path, required=True, help="CSV: file,total_amount,currency,receipt_date,document_number")
    ap.add_argument("--models", default="qwen3-vl:8b+qwen2.5vl:7b",
                    help='comma-separated configs, each "primary[+second]"')
    ap.add_argument("--mode", choices=["accuracy", "live", "both"], default="accuracy")
    ap.add_argument("--service", default="http://localhost:8588", help="running scanning service")
    ap.add_argument("--max-side", type=int, default=VERIFY_IMAGE_MAX_SIDE, help="VERIFY_IMAGE_MAX_SIDE to test")
    ap.add_argument("--out", type=Path, default=Path("benchmark_verify.csv"))
    ap.add_argument("--bbox-check", action="store_true", help="OCR each returned box (offline, after the runs)")
    ap.add_argument("--bbox-ocr-device", default="cpu", help="cpu keeps VRAM free; gpu is faster")
    ap.add_argument("--live-scans", type=int, default=10, help="live mode: scans per phase")
    ap.add_argument("--scan-interval", type=float, default=45, help="live mode: seconds between scans")
    args = ap.parse_args()

    receipts = load_receipts(args.receipts, args.truth)
    if not receipts:
        sys.exit("No receipts to benchmark.")
    print(f"{len(receipts)} receipts. CPUs: {visible_cpus()}. Ollama: {OLLAMA_BASE_URL}")

    if args.mode in ("accuracy", "both"):
        await run_first_pass(args.service, receipts)
        sampler = GpuSampler(OLLAMA_BASE_URL)
        sampler.start()
        runs = []
        try:
            for config in [c.strip() for c in args.models.split(",") if c.strip()]:
                runs.append(await run_verifier_config(config, receipts, sampler, args.max_side))
        finally:
            sampler.stop()
        if args.bbox_check:
            run_bbox_check(runs, receipts, args.bbox_ocr_device)
        summarize(runs)
        write_csv(args.out, runs)

    if args.mode in ("live", "both"):
        await run_live(args.service, receipts, args.live_scans, args.scan_interval)


if __name__ == "__main__":
    asyncio.run(main())
