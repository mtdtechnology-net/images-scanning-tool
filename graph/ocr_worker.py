"""OCR worker module — runs in separate processes via ProcessPoolExecutor.

Each worker process has its own PaddleOCR instance in completely isolated
memory. This solves the thread-safety issue where PaddlePaddle's internal
C++ buffers get corrupted when multiple threads call ocr.ocr() concurrently.

Usage from nodes.py:
    from graph.ocr_worker import run_ocr_parallel
    results = run_ocr_parallel([(b64, idx, total), ...])
"""
import base64
import io
import re
import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np


_NUM_WORKERS = min(4, os.cpu_count() or 4)

_DATE_LINE_RE = re.compile(r"^\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}")

_ocr = None


def _init_worker():
    """Called once when a worker process starts. Loads PaddleOCR into this process."""
    global _ocr
    from paddleocr import PaddleOCR
    _ocr = PaddleOCR(lang="en")


def _merge_continuation_lines(lines_text: list[str]) -> list[str]:
    """Merge OCR rows that don't start with a date into the previous row."""
    merged: list[str] = []
    for line in lines_text:
        if _DATE_LINE_RE.match(line) or not merged:
            merged.append(line)
        else:
            merged[-1] = f"{merged[-1]}\n{line}"
    return merged


def _process_single_image(args: tuple) -> tuple[int, str]:
    """Process a single image in an isolated worker process.
    
    Args:
        args: (img_b64, doc_index, total_docs)
    
    Returns:
        (doc_index, extracted_text) — doc_index is passed through to preserve ordering.
    """
    global _ocr
    img_b64, doc_index, total_docs = args


    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image)
    img_array = img_array[:, :, ::-1]

    result = _ocr.ocr(img_array)
    
    print(f"[DEBUG-OCR-WORKER pid={os.getpid()}] Raw result length: {len(result) if result else 'None'}")
    if result and result[0]:
        print(f"[DEBUG-OCR-WORKER pid={os.getpid()}] First page items count: {len(result[0])}")

    items = []

    if result and result[0]:
        res = result[0]

        if isinstance(res, dict):
            texts = res.get("rec_texts", [])
            polys = res.get("rec_polys") or res.get("dt_polys") or []
            for text, poly in zip(texts, polys):
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                items.append((sum(ys) / len(ys), sum(xs) / len(xs), text))
        else:
            for box, (text, conf) in res:
                y_center = sum(pt[1] for pt in box) / 4.0
                x_center = sum(pt[0] for pt in box) / 4.0
                items.append((y_center, x_center, text))

    items.sort(key=lambda x: x[0])
    rows = []
    current_row = []
    current_y = None

    for y, x, text in items:
        if current_y is None:
            current_y = y
            current_row.append((x, text))
        elif abs(y - current_y) < 15:
            current_row.append((x, text))
            current_y = (current_y * len(current_row) + y) / (len(current_row) + 1)
        else:
            rows.append(current_row)
            current_row = [(x, text)]
            current_y = y

    if current_row:
        rows.append(current_row)

    lines_text = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        lines_text.append(" | ".join(item[1] for item in row))

    lines_text = _merge_continuation_lines(lines_text)

    page_text = "\n".join(lines_text)
    doc_label = f"--- Document {doc_index + 1} / {total_docs} ---"
    full_text = f"{doc_label}\n{page_text}"

    print(f"[OCR-WORKER pid={os.getpid()}] Processed document {doc_index + 1}/{total_docs}"
          f" — {len(lines_text)} rows reconstructed")

    return (doc_index, full_text)


_pool = None


def _get_pool() -> ProcessPoolExecutor:
    """Get or create the process pool (lazy init)."""
    global _pool
    if _pool is None:
        # Pre-initialize in main process to ensure models are downloaded synchronously
        # without race conditions from multiple workers.
        from paddleocr import PaddleOCR
        _ = PaddleOCR(lang="en")
        
        _pool = ProcessPoolExecutor(
            max_workers=_NUM_WORKERS,
            initializer=_init_worker,
        )
    return _pool


def run_ocr_single(img_b64: str, doc_index: int, total_docs: int) -> str:
    """Run OCR on a single image using an isolated worker process.

    Designed to be called from concurrent LangGraph Send() threads — each thread
    submits its own task to the shared pool and blocks until the worker process
    returns the result. Because the pool has _NUM_WORKERS processes, up to
    _NUM_WORKERS documents are OCR-d simultaneously in truly isolated memory.

    Returns:
        extracted text for this document
    """
    pool = _get_pool()
    future = pool.submit(_process_single_image, (img_b64, doc_index, total_docs))
    _, text = future.result()
    return text


def run_ocr_parallel(tasks: list[tuple[str, int, int]]) -> list[str]:
    """Run OCR on multiple images in parallel using separate processes.

    Submits all tasks to the pool at once and collects results ordered by
    doc_index. Use this when you have all documents available upfront.

    Args:
        tasks: list of (img_b64, doc_index, total_docs) tuples

    Returns:
        list of extracted texts, ordered by doc_index
    """
    pool = _get_pool()
    futures = [pool.submit(_process_single_image, task) for task in tasks]

    results = {}
    for future in futures:
        doc_index, text = future.result()
        results[doc_index] = text

    return [results[i] for i in sorted(results.keys())]

