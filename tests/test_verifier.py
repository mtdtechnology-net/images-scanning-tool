"""Verifier logic with mocked model responses: bbox scaling, confidence rules, status."""
import asyncio
import json
from typing import Optional

import pytest
from PIL import Image

from core.config import VERIFY_QWEN25_MAX_PIXELS
from graph.verifier import (
    Reading, bbox_mode, combine_field, parse_json, pick_total_line, prepare_image,
    scale_bbox, verify_pages,
)
from graph.verifier_ollama import VerifierError
from prompts.verify import AMOUNT_LINES_SCHEMA, FULL_PAGE_SCHEMA, ZOOM_SCHEMA, ZOOM_TARGETS
from tools.parser import image_to_base64

PRIMARY = "qwen3-vl:8b-thinking-q8_0"   # relative 0-1000 boxes
SECOND = "qwen2.5vl:7b-q8_0"            # absolute pixel boxes
PAGE_W, PAGE_H = 1000, 2000


# ─── Boxes ───────────────────────────────────────────────────────────────────

def test_bbox_mode():
    assert bbox_mode("qwen2.5vl:7b") == "absolute"
    assert bbox_mode("qwen2.5vl:7b-q8_0") == "absolute"
    assert bbox_mode("qwen3-vl:8b-thinking-q8_0") == "relative"


def test_scale_bbox_relative():
    assert scale_bbox([100, 200, 300, 250], "relative", 0, 0, PAGE_W, PAGE_H) == [100, 400, 300, 500]


def test_scale_bbox_absolute():
    assert scale_bbox([10, 20, 30, 40], "absolute", 560, 784, 1120, 1568) == [20, 40, 60, 80]


def test_scale_bbox_sorts_corners():
    assert scale_bbox([300, 250, 100, 200], "relative", 0, 0, PAGE_W, PAGE_H) == [100, 400, 300, 500]


@pytest.mark.parametrize("box", [
    None, [], [1, 2, 3], ["a", 1, 2, 3], [0, 0, 1200, 10],   # outside the 0-1000 grid
    [100, 100, 100.5, 300],                                   # degenerate
    [0, 0, 1000, 900],                                        # most of the page
])
def test_scale_bbox_rejects_unusable(box):
    assert scale_bbox(box, "relative", 0, 0, PAGE_W, PAGE_H) is None


def test_scale_bbox_absolute_rejects_box_outside_sent_image():
    assert scale_bbox([0, 0, 900, 50], "absolute", 560, 784, 1120, 1568) is None


def test_prepare_image_qwen25_multiples_of_28_within_budget():
    img = prepare_image(Image.new("RGB", (1448, 2048), "white"), SECOND, 1792)
    assert img.width % 28 == 0 and img.height % 28 == 0
    assert img.width * img.height <= VERIFY_QWEN25_MAX_PIXELS


def test_prepare_image_relative_caps_longest_side():
    img = prepare_image(Image.new("RGB", (1448, 2048), "white"), PRIMARY, 1792)
    assert max(img.width, img.height) == 1792


def test_absolute_box_round_trip():
    """A box the model gives on the image it was sent lands on the same spot of the page."""
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    img = prepare_image(page, SECOND, 1792)
    sx, sy = img.width / PAGE_W, img.height / PAGE_H
    sent = [412 * sx, 1880 * sy, 590 * sx, 1932 * sy]
    out = scale_bbox(sent, "absolute", img.width, img.height, PAGE_W, PAGE_H)
    assert out == [412, 1880, 590, 1932]


# ─── Model output ────────────────────────────────────────────────────────────

def test_parse_json_tolerates_wrappers():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('<think>hmm {x}</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("no json") is None
    assert parse_json("") is None


def test_pick_total_line_prefers_final_total():
    lines = [
        {"label": "SUBTOTAL", "amount_text": "100,84", "kind": "subtotal"},
        {"label": "TOTAL FARA TVA", "amount_text": "100,84", "kind": "total"},
        {"label": "TOTAL", "amount_text": "120,00", "kind": "total"},
        {"label": "NUMERAR", "amount_text": "150,00", "kind": "payment"},
        {"label": "REST", "amount_text": "30,00", "kind": "change"},
    ]
    assert pick_total_line(lines)["amount_text"] == "120,00"
    assert pick_total_line([{"label": "REST", "amount_text": "30", "kind": "change"}]) is None
    assert pick_total_line([]) is None


# ─── Confidence rules ────────────────────────────────────────────────────────

def r(source: str, value, bbox: Optional[list] = None, raw: Optional[str] = None, page: int = 0) -> Reading:
    return Reading(source, raw if raw is not None else (None if value is None else str(value)), value, bbox, page)


BOX_A, BOX_D = [400, 1870, 600, 1940], [412, 1880, 590, 1932]


def test_all_agree_with_box_is_high_and_prefers_qwen25_box():
    field, note = combine_field("total_amount", [
        r("A", 120.0, BOX_A, "120,00"), r("B", 120.0), r("C", 120.0, raw="120,00"), r("D", 120.0, BOX_D),
    ])
    assert field == {"value": 120.0, "raw": "120,00", "confidence": "high", "page": 1, "bbox": BOX_D}
    assert note is None


def test_agree_without_box_is_medium():
    field, _ = combine_field("currency", [r("A", "RON", raw="LEI"), r("B", "RON"), r("D", "RON")])
    assert field["confidence"] == "medium" and field["value"] == "RON" and field["bbox"] is None


def test_agree_but_one_read_missed_is_medium():
    field, note = combine_field("receipt_date", [r("A", "2026-11-03", BOX_A), r("B", None), r("D", "2026-11-03")])
    assert field["confidence"] == "medium"
    assert "not found by read B" in note


def test_disagreement_is_low_with_majority_value_and_note():
    field, note = combine_field("total_amount", [
        r("A", 100.84, BOX_A), r("B", 120.0), r("C", 120.0), r("D", 120.0, BOX_D),
    ])
    assert field["confidence"] == "low"
    assert field["value"] == 120.0 and field["bbox"] == BOX_D
    assert "reads disagree" in note and "A=100.84" in note and "D=120.00" in note


def test_tie_prefers_zoom_read():
    field, _ = combine_field("total_amount", [r("A", 128.5), r("C", 123.5)])
    assert field["value"] == 123.5 and field["confidence"] == "low"


def test_document_number_formatting_difference_is_medium():
    field, note = combine_field("document_number", [r("A", "F-0042", BOX_A), r("D", "F42", BOX_D)])
    assert field["confidence"] == "medium"
    assert "formatting" in note


def test_single_read_is_low():
    field, note = combine_field("document_number", [r("A", "F-0042", BOX_A), r("B", None), r("D", None)])
    assert field["confidence"] == "low" and "one read only" in note


def test_nothing_found_is_none():
    field, note = combine_field("document_number", [r("A", None), r("B", None), r("D", None)])
    assert field == {"value": None, "raw": None, "confidence": "none", "page": None, "bbox": None}
    assert note is None


# ─── End to end with a fake backend ──────────────────────────────────────────

def _page_b64() -> str:
    return image_to_base64(Image.new("RGB", (PAGE_W, PAGE_H), "white"))


def _field(text: str, box: list) -> dict:
    return {"text": text, "bbox_2d": box}


def full_page(total="120,00", currency="LEI", date="03.11.2026", number="Nr. F-0042",
              doc_type="receipt", count=1, boxes=True, absolute=False) -> dict:
    """A canned full-page answer. Boxes: relative for Qwen3-VL, absolute (sent image px) for Qwen2.5-VL."""
    def box(page_box):
        if not boxes:
            return []
        if absolute:
            img = prepare_image(Image.new("RGB", (PAGE_W, PAGE_H)), SECOND, 1792)
            return [page_box[0] * img.width / PAGE_W, page_box[1] * img.height / PAGE_H,
                    page_box[2] * img.width / PAGE_W, page_box[3] * img.height / PAGE_H]
        return [page_box[0] * 1000 / PAGE_W, page_box[1] * 1000 / PAGE_H,
                page_box[2] * 1000 / PAGE_W, page_box[3] * 1000 / PAGE_H]
    return {
        "document_type": doc_type, "receipt_count": count,
        "total_amount": _field(total, box([412, 1880, 590, 1932])),
        "currency": _field(currency, box([600, 1880, 660, 1932])),
        "receipt_date": _field(date, box([100, 300, 300, 340])),
        "document_number": _field(number, box([100, 360, 300, 400])),
    }


def amount_lines(total="120,00", box=True) -> dict:
    return {
        "amount_lines": [
            {"label": "SUBTOTAL", "amount_text": "100,84", "kind": "subtotal", "bbox_2d": []},
            {"label": "TVA A 19%", "amount_text": "19,16", "kind": "vat", "bbox_2d": []},
            {"label": "TOTAL", "amount_text": total, "kind": "total", "bbox_2d": [412, 940, 590, 966] if box else []},
            {"label": "REST", "amount_text": "30,00", "kind": "change", "bbox_2d": []},
        ],
        "currency_text": "LEI", "date_text": "03.11.2026", "document_number_text": "Nr. F-0042",
    }


class FakeBackend:
    def __init__(self, primary_page, second_page, lines=None, zoom=None, fail=()):
        self.answers = {(PRIMARY, "full"): primary_page, (SECOND, "full"): second_page,
                        (PRIMARY, "lines"): lines or amount_lines()}
        self.zoom = zoom or {"total_amount": "120,00", "currency": "LEI",
                             "receipt_date": "03.11.2026", "document_number": "F-0042"}
        self.fail = set(fail)
        self.calls: list[tuple[str, str]] = []
        self.prompts: list[str] = []
        self.loaded: list[str] = []

    async def load(self, model):
        self.loaded.append(model)

    async def unload(self, model):
        self.loaded.remove(model)

    async def chat(self, model, prompt, image_b64, schema):
        kind = {id(FULL_PAGE_SCHEMA): "full", id(AMOUNT_LINES_SCHEMA): "lines", id(ZOOM_SCHEMA): "zoom"}[id(schema)]
        self.calls.append((model, kind))
        self.prompts.append(prompt)
        if (model, kind) in self.fail:
            raise VerifierError("boom")
        if kind == "zoom":
            fld = next(f for f, target in ZOOM_TARGETS.items() if target in prompt)
            return json.dumps({"text": self.zoom[fld]})
        return json.dumps(self.answers[(model, kind)])


def run(backend, pages=1):
    return asyncio.run(verify_pages([_page_b64()] * pages, backend, PRIMARY, SECOND, max_side=1792)).response


def test_clean_receipt_is_ok_and_high():
    backend = FakeBackend(full_page(), full_page(absolute=True))
    resp = run(backend)
    assert resp["status"] == "ok"
    assert resp["pages"] == [{"page": 1, "width": PAGE_W, "height": PAGE_H}]
    total = resp["fields"]["total_amount"]
    assert total["value"] == 120.0 and total["raw"] == "120,00" and total["confidence"] == "high"
    assert total["page"] == 1
    assert all(abs(a - b) <= 2 for a, b in zip(total["bbox"], [412, 1880, 590, 1932]))
    assert resp["fields"]["currency"]["value"] == "RON"
    assert resp["fields"]["receipt_date"]["value"] == "2026-11-03"
    assert resp["fields"]["document_number"]["value"] == "F-0042"
    assert resp["notes"] is None
    # A, B, then one zoom per located field on the primary model, then D on the second.
    assert backend.calls == [(PRIMARY, "full"), (PRIMARY, "lines")] + [(PRIMARY, "zoom")] * 4 + [(SECOND, "full")]


def test_subtotal_misread_by_one_read_makes_total_unreadable():
    backend = FakeBackend(full_page(total="100,84"), full_page(absolute=True))
    resp = run(backend)
    assert resp["status"] == "unreadable"
    assert resp["fields"]["total_amount"]["confidence"] == "low"
    assert resp["fields"]["total_amount"]["value"] == 120.0
    assert "disagree" in resp["notes"]


def test_without_boxes_confidence_is_medium():
    backend = FakeBackend(full_page(boxes=False), full_page(boxes=False, absolute=True),
                          lines=amount_lines(box=False))
    resp = run(backend)
    assert resp["status"] == "ok"
    assert resp["fields"]["total_amount"]["confidence"] == "medium"
    assert resp["fields"]["total_amount"]["bbox"] is None
    assert (PRIMARY, "zoom") not in backend.calls  # nothing located, nothing to zoom on


def test_not_a_receipt_when_both_models_agree():
    empty = dict(total="", currency="", date="", number="", boxes=False)
    backend = FakeBackend(full_page(doc_type="other", **empty), full_page(doc_type="other", **empty))
    resp = run(backend)
    assert resp["status"] == "not_a_receipt"
    assert all(f["confidence"] == "none" for f in resp["fields"].values())


def test_not_a_receipt_disputed_by_second_model_is_unreadable():
    backend = FakeBackend(full_page(doc_type="other"), full_page(absolute=True))
    assert run(backend)["status"] == "unreadable"


def test_multiple_receipts():
    backend = FakeBackend(full_page(count=2), full_page(absolute=True))
    assert run(backend)["status"] == "multiple_receipts"


def test_failed_read_is_noted_not_fatal():
    backend = FakeBackend(full_page(), full_page(absolute=True), fail={(PRIMARY, "lines")})
    resp = run(backend)
    assert resp["status"] == "ok"
    assert "read B failed" in resp["notes"]


def test_failed_first_read_is_an_error():
    backend = FakeBackend(full_page(), full_page(absolute=True), fail={(PRIMARY, "full")})
    with pytest.raises(VerifierError):
        run(backend)


def test_models_are_swapped_in_order():
    backend = FakeBackend(full_page(), full_page(absolute=True))
    run(backend)
    assert backend.loaded == [SECOND]  # primary unloaded before the second model was loaded


def test_prompts_never_carry_declared_or_ocr_values():
    backend = FakeBackend(full_page(), full_page(absolute=True))
    run(backend)
    for prompt in backend.prompts:
        assert "120" not in prompt and "F-0042" not in prompt and "declared" not in prompt.lower()


# ─── Currency not printed must not raise doubts ──────────────────────────────

def test_currency_found_by_one_read_only_is_medium_not_low():
    field, _ = combine_field("currency", [r("A", "RON", raw="LEI"), r("B", None), r("D", None)])
    assert field["confidence"] == "medium" and field["value"] == "RON"


def test_currency_not_printed_is_none_and_receipt_stays_ok():
    backend = FakeBackend(full_page(currency="", boxes=True), full_page(currency="", absolute=True),
                          lines={**amount_lines(), "currency_text": ""},
                          zoom={"total_amount": "120,00", "currency": "", "receipt_date": "03.11.2026",
                                "document_number": "F-0042"})
    resp = run(backend)
    assert resp["status"] == "ok"
    assert resp["fields"]["currency"] == {"value": None, "raw": None, "confidence": "none", "page": None, "bbox": None}


def test_conflicting_currencies_are_still_low():
    field, note = combine_field("currency", [r("A", "RON"), r("B", "EUR"), r("D", None)])
    assert field["confidence"] == "low" and "disagree" in note


# ─── Unusable answers are retried ────────────────────────────────────────────

class FlakyBackend(FakeBackend):
    """The first full-page read fails (no JSON, or ran out of tokens while thinking)."""
    def __init__(self, failure):
        super().__init__(full_page(), full_page(absolute=True))
        self.failure, self.failed = failure, False

    async def chat(self, model, prompt, image_b64, schema):
        if not self.failed and schema is FULL_PAGE_SCHEMA and model == PRIMARY:
            self.failed = True
            if isinstance(self.failure, Exception):
                raise self.failure
            return self.failure
        return await super().chat(model, prompt, image_b64, schema)


@pytest.mark.parametrize("failure", [
    "", "<think>long reasoning…</think>", VerifierError("'m' used all 8192 tokens (VERIFY_NUM_PREDICT) before answering"),
])
def test_unusable_read_is_retried(failure):
    resp = run(FlakyBackend(failure))
    assert resp["status"] == "ok"


def test_ollama_errors_are_not_retried():
    with pytest.raises(VerifierError, match="Ollama error"):
        run(FlakyBackend(VerifierError("Ollama error from 'm': boom")))
