"""Receipt verifier: re-read a receipt from the page image only and measure how sure we are.

Independence from the live scan: the models never see OCR text or the values the employee
declared. Confidence comes from agreement between independent reads, not from the model:

  A  primary model, full page, all four fields + boxes        (every page)
  B  primary model, other prompt and scale: labelled amount lines; code picks the total
  C  primary model, zoomed strip around each field A located
  D  second model (other Qwen generation), full page + boxes   (pages where A found fields)

  high    every read that ran agrees after normalization, and two reads put a box on the same spot
  medium  reads agree but there is no reliable box, a read did not find the value,
          or they differ only in formatting (document number "F-0042" vs "F42")
  low     reads disagree; the majority value is returned and `notes` lists every read
  none    no read found the field

The backend (graph/verifier_ollama.py) is passed in, so tests can use canned responses.
"""
import base64
import io
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

from PIL import Image

from core.config import VERIFY_IMAGE_MAX_SIDE, VERIFY_QWEN25_MAX_PIXELS, VERIFY_READ_RETRIES
from graph.normalize import FIELDS, NORMALIZERS, loose_key, normalize_amount
from graph.ocr_witness import OcrLine, apply_currency_witness, apply_total_witness, build_evidence
from graph.verifier_ollama import VerifierError
from prompts.verify import (
    AMOUNT_LINES_PROMPT, AMOUNT_LINES_SCHEMA, FULL_PAGE_PROMPT, FULL_PAGE_SCHEMA,
    ZOOM_PROMPT, ZOOM_SCHEMA, ZOOM_TARGETS,
)
from tools.parser import image_to_base64

logger = logging.getLogger(__name__)

JPEG_QUALITY = 92        # digits must survive the re-encoding
READ_B_SCALE = 0.75      # read B sees the page at a different scale than read A
VALUE_PRIORITY = {"C": 0, "A": 1, "D": 2, "B": 3}  # tie-break when reads disagree
BBOX_PRIORITY = ("D", "A", "B")                    # Qwen2.5-VL boxes are the most precise


class Backend(Protocol):
    async def load(self, model: str) -> None: ...
    async def unload(self, model: str) -> None: ...
    async def chat(self, model: str, prompt: str, image_b64: str, schema: dict) -> str: ...


# ─── Images and boxes ────────────────────────────────────────────────────────

def bbox_mode(model: str) -> str:
    """Qwen2.5-VL answers boxes in pixels of the image it received; Qwen3-VL (and later)
    on a relative 0-1000 grid."""
    return "absolute" if re.search(r"qwen2\.?5[-_ ]?vl", model.lower()) else "relative"


@dataclass
class ModelImage:
    b64: str
    width: int
    height: int


def decode_page(img_b64: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    return image.convert("RGB") if image.mode != "RGB" else image


def prepare_image(image: Image.Image, model: str, max_side: int) -> ModelImage:
    """Resize for `model`. For Qwen2.5-VL, sides become multiples of 28 within its pixel
    budget, so Ollama doesn't resize again and box pixels map 1:1 to this image."""
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if bbox_mode(model) == "absolute":
        scale = min(scale, math.sqrt(VERIFY_QWEN25_MAX_PIXELS / (w * h)))
        nw = max(28, int(w * scale) // 28 * 28)
        nh = max(28, int(h * scale) // 28 * 28)
    else:
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image if (nw, nh) == (w, h) else image.resize((nw, nh), Image.LANCZOS)
    return ModelImage(image_to_base64(resized, quality=JPEG_QUALITY), nw, nh)


def scale_bbox(box: Any, mode: str, sent_w: int, sent_h: int, page_w: int, page_h: int) -> Optional[list[int]]:
    """Map a model box onto page pixels. Returns None for anything that isn't a usable box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    ref_w, ref_h = (1000.0, 1000.0) if mode == "relative" else (float(sent_w), float(sent_h))
    tol = 0.02
    if x0 < -tol * ref_w or y0 < -tol * ref_h or x1 > ref_w * (1 + tol) or y1 > ref_h * (1 + tol):
        return None  # outside the image: wrong coordinate system or hallucinated
    x0, x1 = max(0.0, x0), min(ref_w, x1)
    y0, y1 = max(0.0, y0), min(ref_h, y1)
    sx, sy = page_w / ref_w, page_h / ref_h
    out = [round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)]
    bw, bh = out[2] - out[0], out[3] - out[1]
    if bw < 2 or bh < 2:
        return None
    if bw * bh > 0.6 * page_w * page_h:
        return None  # a box around half the page grounds nothing
    return out


def zoom_strip(image: Image.Image, bbox: list[int], max_side: int) -> Image.Image:
    """Full-width strip around a box (so the line's label stays in view), enlarged up to 2x."""
    w, h = image.size
    pad = max((bbox[3] - bbox[1]) * 1.5, h * 0.02)
    top, bottom = max(0, int(bbox[1] - pad)), min(h, int(bbox[3] + pad))
    strip = image.crop((0, top, w, bottom))
    scale = min(2.0, max_side / max(strip.size))
    if abs(scale - 1.0) > 0.01:
        strip = strip.resize((max(1, round(strip.width * scale)), max(1, round(strip.height * scale))),
                             Image.LANCZOS)
    return strip


# ─── Model output ────────────────────────────────────────────────────────────

def parse_json(content: str) -> Optional[dict]:
    """Model output to dict; tolerates thinking tags and code fences around the JSON."""
    if not content:
        return None
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


@dataclass
class Reading:
    source: str              # "A" | "B" | "C" | "D"
    raw: Optional[str]       # exactly what the model read
    value: Any               # normalized, None = not found
    bbox: Optional[list[int]]
    page: int                # 0-based


def make_reading(source: str, fld: str, raw: Optional[str], page: int, bbox: Optional[list[int]] = None) -> Reading:
    raw = _text(raw)
    value = NORMALIZERS[fld](raw) if raw else None
    return Reading(source, raw, value, bbox if value is not None else None, page)


@dataclass
class PageRead:
    document_type: str
    receipt_count: int
    fields: dict[str, Reading]


def pick_total_line(lines: list) -> Optional[dict]:
    """The final total among labelled amount lines. Several distinct "total" values (net vs
    gross, or the VAT total mislabelled) -> the largest, since the amount paid includes VAT."""
    totals = []
    for line in lines or []:
        if isinstance(line, dict) and line.get("kind") == "total":
            amount = normalize_amount(_text(line.get("amount_text")))
            if amount is not None:
                totals.append((amount, line))
    if not totals:
        return None
    return max(totals, key=lambda t: t[0])[1]


# ─── Reads ───────────────────────────────────────────────────────────────────

async def _ask(backend: Backend, model: str, prompt: str, img: ModelImage, schema: dict) -> dict:
    """One read; retried when the model returns nothing usable (a thinking model that ran
    out of tokens, broken JSON). Transport errors are not retried: they come from Ollama."""
    error: Optional[VerifierError] = None
    for attempt in range(1 + max(0, VERIFY_READ_RETRIES)):
        try:
            data = parse_json(await backend.chat(model, prompt, img.b64, schema))
        except VerifierError as e:
            if "tokens" not in str(e):  # only the token-limit error is worth retrying
                raise
            error, data = e, None
        if data is not None:
            return data
        error = error or VerifierError(f"'{model}' returned no valid JSON")
        logger.info("Verifier %s: unusable answer (attempt %d): %s", model, attempt + 1, error)
    raise error


async def read_full_page(backend: Backend, model: str, source: str, image: Image.Image,
                         page: int, max_side: int) -> PageRead:
    img = prepare_image(image, model, max_side)
    data = await _ask(backend, model, FULL_PAGE_PROMPT, img, FULL_PAGE_SCHEMA)
    mode = bbox_mode(model)
    fields = {}
    for fld in FIELDS:
        item = data.get(fld) if isinstance(data.get(fld), dict) else {}
        bbox = scale_bbox(item.get("bbox_2d"), mode, img.width, img.height, *image.size)
        fields[fld] = make_reading(source, fld, item.get("text"), page, bbox)
    doc_type = str(data.get("document_type") or "").strip().lower()
    try:
        count = int(data.get("receipt_count") or 0)
    except (TypeError, ValueError):
        count = 0
    return PageRead(doc_type, count, fields)


def paid_amount(lines: list) -> Optional[float]:
    """What the receipt says was paid: payment lines minus change, or None without payment lines.
    Identical payment amounts are counted once (the same card payment is often printed twice)."""
    payments, change = set(), 0.0
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        amount = normalize_amount(_text(line.get("amount_text")))
        if amount is None:
            continue
        if line.get("kind") == "payment":
            payments.add(amount)
        elif line.get("kind") == "change":
            change += amount
    return round(sum(payments) - change, 2) if payments else None


async def read_amount_lines(backend: Backend, model: str, image: Image.Image, page: int,
                            max_side: int) -> tuple[dict[str, Reading], Optional[float]]:
    """Read B: labelled amount lines (code picks the total) + the other fields again.
    Also returns the paid amount (payments − change) printed on the receipt, if any."""
    img = prepare_image(image, model, max(256, int(max_side * READ_B_SCALE)))
    data = await _ask(backend, model, AMOUNT_LINES_PROMPT, img, AMOUNT_LINES_SCHEMA)
    lines = data.get("amount_lines") if isinstance(data.get("amount_lines"), list) else []
    total = pick_total_line(lines)
    box = lambda b: scale_bbox(b, bbox_mode(model), img.width, img.height, *image.size)  # noqa: E731
    readings = {
        "total_amount": make_reading("B", "total_amount", total.get("amount_text") if total else None,
                                     page, box(total.get("bbox_2d")) if total else None),
        "currency": make_reading("B", "currency", data.get("currency_text"), page, box(data.get("currency_bbox_2d"))),
        "receipt_date": make_reading("B", "receipt_date", data.get("date_text"), page, box(data.get("date_bbox_2d"))),
        "document_number": make_reading("B", "document_number", data.get("document_number_text"), page),
    }
    return readings, paid_amount(lines)


async def read_zoom(backend: Backend, model: str, fld: str, image: Image.Image, bbox: list[int],
                    page: int, max_side: int) -> Reading:
    strip = zoom_strip(image, bbox, max_side)
    img = ModelImage(image_to_base64(strip, quality=JPEG_QUALITY), *strip.size)
    data = await _ask(backend, model, ZOOM_PROMPT.format(target=ZOOM_TARGETS[fld]), img, ZOOM_SCHEMA)
    return make_reading("C", fld, data.get("text"), page)


# ─── Combining reads ─────────────────────────────────────────────────────────

NONE_FIELD = {"value": None, "raw": None, "confidence": "none", "page": None, "bbox": None}


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def combine_field(fld: str, readings: list[Reading]) -> tuple[dict, Optional[str]]:
    """Merge the reads of one field into the API field object (+ a note when not sure)."""
    present = [r for r in readings if r.value is not None]
    if not present:
        return dict(NONE_FIELD), None

    groups: dict[Any, list[Reading]] = {}
    for r in present:
        groups.setdefault(r.value, []).append(r)
    chosen = max(groups, key=lambda v: (len(groups[v]), -min(VALUE_PRIORITY[r.source] for r in groups[v])))
    support = sorted(groups[chosen], key=lambda r: VALUE_PRIORITY[r.source])
    bbox_reading = confirmed_box(support)
    missing = [r.source for r in readings if r.value is None]
    loose = {loose_key(fld, r.value) for r in present}

    note = None
    listing = ", ".join(f"{r.source}={_fmt(r.value)}" for r in sorted(readings, key=lambda r: r.source))
    if len(present) < 2 and fld == "currency":
        # Many receipts don't print a currency, so a read not finding one is normal, not a
        # doubt. Currency is "low" only when reads found different currencies (below).
        confidence = "medium"
        note = f"{fld}: found by one read only ({listing})"
    elif len(present) < 2:
        confidence = "low"
        note = f"{fld}: found by one read only ({listing})"
    elif len(groups) == 1:
        if missing:
            confidence = "medium"
            note = f"{fld}: not found by read {', '.join(sorted(missing))}"
        else:
            confidence = "high" if bbox_reading else "medium"
    elif len(loose) == 1:
        confidence = "medium"
        note = f"{fld}: reads differ only in formatting ({listing})"
    else:
        confidence = "low"
        note = f"{fld}: reads disagree ({listing})"

    return {
        "value": chosen,
        "raw": support[0].raw,
        "confidence": confidence,
        "page": (bbox_reading or support[0]).page + 1,
        "bbox": bbox_reading.bbox if bbox_reading else None,
    }, note


def boxes_agree(a: list[int], b: list[int]) -> bool:
    """Two boxes point at the same spot: they overlap by at least 30% of the smaller one."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return False
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h >= 0.3 * smaller


def confirmed_box(support: list[Reading]) -> Optional[Reading]:
    """The reading whose box is confirmed by another read's box on the same spot. A single
    box is not trusted: models sometimes place it on the wrong line or off the receipt."""
    boxed = sorted((r for r in support if r.bbox), key=lambda r: BBOX_PRIORITY.index(r.source)
                   if r.source in BBOX_PRIORITY else len(BBOX_PRIORITY))
    for i, r in enumerate(boxed):
        if any(o.page == r.page and boxes_agree(r.bbox, o.bbox) for o in boxed[i + 1:]):
            return r
    return None


# ─── Orchestration ───────────────────────────────────────────────────────────

DOC_TYPES = ("receipt", "invoice")
# Read C (zoom) only where it matters: the document number is informative, not compared.
ZOOM_FIELDS = ("total_amount", "currency", "receipt_date")


@dataclass
class Outcome:
    response: dict
    readings: dict[str, list[Reading]] = field(default_factory=dict)


def _field_pages(a_reads: list[PageRead], doc_pages: list[int]) -> dict[str, int]:
    """Page to check each field on: the total on the last page that shows one (final totals
    close the document), the other fields on the first page that shows them."""
    pages: dict[str, int] = {}
    total_pages = [i for i in doc_pages if a_reads[i].fields["total_amount"].value is not None]
    default = doc_pages[-1] if doc_pages else 0
    pages["total_amount"] = total_pages[-1] if total_pages else default
    for fld in FIELDS[1:]:
        found = [i for i in doc_pages if a_reads[i].fields[fld].value is not None]
        pages[fld] = found[0] if found else pages["total_amount"]
    return pages


OcrFn = Callable[[Image.Image], Awaitable[list[tuple[str, list[int]]]]]


async def verify_pages(images_b64: list[str], backend: Backend, primary: str,
                       second: Optional[str] = None, max_side: int = VERIFY_IMAGE_MAX_SIDE,
                       ocr: Optional[OcrFn] = None) -> Outcome:
    """Run every read on the page images and build the API response (without duration_ms).
    `ocr(page_image)` (optional) returns (text, box) lines: the independent OCR witness."""
    pages = [decode_page(b64) for b64 in images_b64]
    notes: list[str] = []
    paid: Optional[float] = None
    readings: dict[str, list[Reading]] = {fld: [] for fld in FIELDS}

    # ── Primary model: A on every page ──
    await backend.load(primary)
    a_reads = [await read_full_page(backend, primary, "A", img, i, max_side) for i, img in enumerate(pages)]
    doc_pages = [i for i, r in enumerate(a_reads) if r.document_type in DOC_TYPES]
    field_page = _field_pages(a_reads, doc_pages)
    for fld in FIELDS:
        readings[fld].append(a_reads[field_page[fld]].fields[fld])

    # ── Primary model: B and C on the pages that matter ──
    if doc_pages:
        total_page = field_page["total_amount"]
        try:
            b, paid = await read_amount_lines(backend, primary, pages[total_page], total_page, max_side)
            for fld in FIELDS:
                if field_page[fld] == total_page:
                    readings[fld].append(b[fld])
        except VerifierError as e:
            notes.append(f"read B failed ({e})")
        for fld in ZOOM_FIELDS:
            a = a_reads[field_page[fld]].fields[fld]
            if a.bbox is None:
                continue
            try:
                readings[fld].append(await read_zoom(backend, primary, fld, pages[a.page], a.bbox, a.page, max_side))
            except VerifierError as e:
                notes.append(f"read C on {fld} failed ({e})")

    # ── Second model: D ──
    d_reads: dict[int, PageRead] = {}
    if second:
        await backend.unload(primary)
        await backend.load(second)
        for i in sorted(set(field_page.values())):
            try:
                d_reads[i] = await read_full_page(backend, second, "D", pages[i], i, max_side)
            except VerifierError as e:
                notes.append(f"read D on page {i + 1} failed ({e})")
        for fld in FIELDS:
            if field_page[fld] in d_reads:
                readings[fld].append(d_reads[field_page[fld]].fields[fld])

    # ── Fields ──
    fields = {}
    for fld in FIELDS:
        fields[fld], note = combine_field(fld, readings[fld])
        if note:
            notes.append(note)

    # ── OCR witness on the total's page: confirms or contradicts the total and the currency ──
    total = fields["total_amount"]
    witnessed = False
    if ocr is not None and doc_pages:
        page = field_page["total_amount"]
        try:
            lines = [OcrLine(text, box) for text, box in await ocr(pages[page])]
            ev = build_evidence(lines, page)
            read_values = [r.value for r in readings["total_amount"] if r.value is not None]
            if ev.has_amounts:
                note = apply_total_witness(total, read_values, ev)
                if note:
                    notes.append(note)
                witnessed = True  # else (TOTAL and payment unreadable for OCR) keep the check below
            note = apply_currency_witness(fields["currency"], ev)
            if note:
                notes.append(note)
        except Exception as e:  # noqa: BLE001 - OCR trouble must not lose the model's reads
            logger.warning("OCR witness failed: %s", e)
            notes.append("OCR witness unavailable")

    # Without OCR, the payment read by the model is the only cross-check of the total.
    if not witnessed and paid is not None and total["value"] is not None and abs(total["value"] - paid) > 0.005:
        total["confidence"] = "low"
        notes.append(f"total_amount: the total read ({total['value']:.2f}) differs from the payment "
                     f"printed on the receipt minus change ({paid:.2f})")

    # ── Status ──
    status = "ok"
    a_types = [r.document_type for r in a_reads]
    d_types = [r.document_type for r in d_reads.values()]
    if not doc_pages:
        if all(t == "other" for t in a_types) and all(t == "other" for t in d_types):
            status = "not_a_receipt"
            notes.insert(0, "Not a receipt or invoice.")
        else:
            status = "unreadable"
            notes.insert(0, f"No page could be read as a receipt (read A: {', '.join(a_types)}"
                            + (f"; read D: {', '.join(d_types)}" if d_types else "") + ").")
    else:
        if any(t not in DOC_TYPES for t in d_types):
            notes.append(f"read D classified the document as: {', '.join(d_types)}")
        doc_numbers = {loose_key("document_number", a_reads[i].fields["document_number"].value)
                       for i in doc_pages} - {None}
        counts = [a_reads[i].receipt_count for i in doc_pages] + [r.receipt_count for r in d_reads.values()]
        if any(c > 1 for c in counts) or len(doc_numbers) > 1:
            status = "multiple_receipts"
            notes.insert(0, "More than one receipt/invoice in this file.")
        elif fields["total_amount"]["confidence"] == "none":
            status = "unreadable"
            notes.insert(0, "No total amount found on the document.")
        elif fields["total_amount"]["confidence"] == "low":
            status = "unreadable"
            notes.insert(0, "The total amount is uncertain (see total_amount below).")
        totals = {a_reads[i].fields["total_amount"].value for i in doc_pages} - {None}
        if len(totals) > 1 and status == "ok":
            notes.append(f"pages show different totals; using page {field_page['total_amount'] + 1}")

    response = {
        "status": status,
        "model": primary,
        "models": [m for m in (primary, second) if m],
        "duration_ms": 0,  # set by the caller
        "pages": [{"page": i + 1, "width": img.width, "height": img.height} for i, img in enumerate(pages)],
        "fields": fields,
        "notes": "; ".join(notes) or None,
    }
    return Outcome(response, readings)


def error_response(notes: str, primary: str, second: Optional[str], pages: list[dict] | None = None) -> dict:
    return {
        "status": "error",
        "model": primary,
        "models": [m for m in (primary, second) if m],
        "duration_ms": 0,
        "pages": pages or [],
        "fields": {fld: dict(NONE_FIELD) for fld in FIELDS},
        "notes": notes,
    }
