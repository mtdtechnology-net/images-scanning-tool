"""OCR worker module — thread-safe with per-thread PaddleOCR instances.

Each thread gets its own PaddleOCR instance via threading.local() to avoid
C++ buffer corruption when LangGraph runs multiple document processing
nodes concurrently.

Includes:
- Image preprocessing pipeline (auto-crop, enhance, etc.)
- OCR text reconstruction with row merging

Usage from nodes.py:
    from graph.ocr_worker import run_ocr_single
    text = run_ocr_single(b64, idx, total)
"""
import base64
import io
import re
import os
from PIL import Image
import numpy as np
from graph.preprocess import preprocess_for_ocr


_DATE_LINE_RE = re.compile(r"^\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}")

import threading

_thread_local = threading.local()
_ocr_init_lock = threading.Lock()

def _get_ocr():
    """Get or create a PaddleOCR instance for the current thread.
    
    PaddleOCR's C++ internals crash if called from a different thread
    than the one that created the instance. We use threading.local()
    so each LangGraph worker thread gets its own isolated instance.
    """
    if not hasattr(_thread_local, "ocr_instance"):
        with _ocr_init_lock:
            from paddleocr import PaddleOCR
            _thread_local.ocr_instance = PaddleOCR(lang="en", use_angle_cls=False, use_gpu=False)
    return _thread_local.ocr_instance

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
    img_b64, doc_index, total_docs = args

    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # --- NEW: Preprocessing pipeline (auto-crop, enhance, cap size) ---
    image = preprocess_for_ocr(image)

    img_array = np.array(image)
    img_array = img_array[:, :, ::-1]

    ocr_instance = _get_ocr()
    result = ocr_instance.ocr(img_array)
    
    print(f"[DEBUG-OCR] Raw result length: {len(result) if result else 'None'}")
    if result and result[0]:
        print(f"[DEBUG-OCR] First page items count: {len(result[0])}")

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

    print(f"[OCR] Processed document {doc_index + 1}/{total_docs}"
          f" — {len(lines_text)} rows reconstructed")

    return (doc_index, full_text)


def run_ocr_single(img_b64: str, doc_index: int, total_docs: int) -> str:
    """Run OCR on a single image (thread-safe via threading.local)."""
    _, text = _process_single_image((img_b64, doc_index, total_docs))
    return text


def run_ocr_parallel(tasks: list[tuple[str, int, int]]) -> list[str]:
    """Run OCR on multiple images sequentially (each in its own thread-local context)."""
    results = {}
    for task in tasks:
        doc_index, text = _process_single_image(task)
        results[doc_index] = text

    return [results[i] for i in sorted(results.keys())]
