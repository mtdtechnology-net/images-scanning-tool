"""OCR worker module — one PaddleOCR instance on one dedicated thread.

PaddleOCR's C++ internals crash if an instance is used from a different thread
than the one that created it, and a second instance would only duplicate the
model in (GPU) memory. So all OCR calls go through a single-thread executor,
serialized by an asyncio lock. Image preprocessing (CPU) runs outside that
thread so it overlaps with OCR of other pages.

Includes:
- Image preprocessing pipeline (auto-crop, enhance, etc.)
- OCR text reconstruction with row merging

Usage:
    from graph.ocr_worker import run_ocr
    text = await run_ocr(b64, idx, total)
"""
import asyncio
import base64
import io
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from core.config import OCR_DEVICE, OCR_LANG
from graph.preprocess import preprocess_for_ocr

logger = logging.getLogger(__name__)

_DATE_LINE_RE = re.compile(r"^\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}")

_ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
_ocr_lock = asyncio.Lock()
_ocr_instance = None


def _get_ocr():
    """Create the PaddleOCR instance lazily; only ever called on the OCR thread."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(
            lang=OCR_LANG,
            device=OCR_DEVICE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # oneDNN on CPU crashes with the Linux paddlepaddle-gpu build
            # ("ConvertPirAttribute2RuntimeAttribute not support"); it is only used on CPU.
            enable_mkldnn=not (OCR_DEVICE == "cpu" and sys.platform == "linux"),
        )
        logger.info("PaddleOCR loaded (lang=%s, device=%s)", OCR_LANG, OCR_DEVICE)
    return _ocr_instance


def _merge_continuation_lines(lines_text: list[str]) -> list[str]:
    """Merge OCR rows that don't start with a date into the previous row."""
    merged: list[str] = []
    for line in lines_text:
        if _DATE_LINE_RE.match(line) or not merged:
            merged.append(line)
        else:
            merged[-1] = f"{merged[-1]}\n{line}"
    return merged


def _prepare_image(img_b64: str) -> np.ndarray:
    """Decode + preprocess an image into the BGR array PaddleOCR expects."""
    image = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = preprocess_for_ocr(image)
    return np.array(image)[:, :, ::-1]


def _ocr_items(img_array: np.ndarray) -> list[tuple[float, float, str]]:
    """Run OCR and return (y_center, x_center, text) for every detected line."""
    result = _get_ocr().predict(img_array)
    if not result:
        return []

    res = result[0]
    items = []
    for text, poly in zip(res["rec_texts"], res["rec_polys"]):
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        items.append((sum(ys) / len(ys), sum(xs) / len(xs), text))
    return items


def _items_to_text(items: list[tuple[float, float, str]]) -> list[str]:
    """Group OCR items into visual rows (by y) and join each row left to right."""
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

    return _merge_continuation_lines(lines_text)


async def run_ocr(img_b64: str, doc_index: int, total_docs: int, on_start=None) -> str:
    """Preprocess + OCR one page. `on_start` is awaited once the OCR slot is acquired."""
    loop = asyncio.get_running_loop()
    img_array = await asyncio.to_thread(_prepare_image, img_b64)

    async with _ocr_lock:
        if on_start:
            await on_start()
        items = await loop.run_in_executor(_ocr_executor, _ocr_items, img_array)

    lines_text = _items_to_text(items)
    logger.info("OCR page %d/%d — %d rows reconstructed", doc_index + 1, total_docs, len(lines_text))

    page_text = "\n".join(lines_text)
    return f"--- Document {doc_index + 1} / {total_docs} ---\n{page_text}"


def _ocr_lines(img_array: np.ndarray) -> list[tuple[str, list[int]]]:
    """Run OCR and return (text, [x0, y0, x1, y1]) for every detected line."""
    result = _get_ocr().predict(img_array)
    if not result:
        return []
    res = result[0]
    lines = []
    for text, poly in zip(res["rec_texts"], res["rec_polys"]):
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        lines.append((text, [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]))
    return lines


async def ocr_page_lines(image: Image.Image) -> list[tuple[str, list[int]]]:
    """OCR one page image as it is (no crop, warp or resize), so the line boxes are in the
    page's own pixel coordinates. Used by the receipt verifier as an independent witness."""
    loop = asyncio.get_running_loop()
    img_array = await asyncio.to_thread(lambda: np.array(image.convert("RGB"))[:, :, ::-1].copy())
    async with _ocr_lock:
        return await loop.run_in_executor(_ocr_executor, _ocr_lines, img_array)


async def warmup_ocr() -> None:
    """Load the OCR model (and GPU kernels) at startup instead of on the first request."""
    loop = asyncio.get_running_loop()
    blank = np.full((64, 256, 3), 255, dtype=np.uint8)
    await loop.run_in_executor(_ocr_executor, _ocr_items, blank)
