"""OCR witness: confirms/contradicts the model's total and currency with OCR lines.
Lines mimic the benchmark receipts (Rompetrol, MOL, OMV) at page-pixel positions."""
import asyncio

import pytest

from graph.ocr_witness import (
    OcrLine, apply_currency_witness, apply_total_witness, build_evidence, group_rows, row_kind,
)
from tests.test_verifier import FakeBackend, full_page, run as run_verifier  # noqa: F401


def L(text, x0, y0, x1=None, y1=None):
    return OcrLine(text, [x0, y0, x1 if x1 is not None else x0 + 20 * len(text), y1 if y1 is not None else y0 + 30])


ROMPETROL = [  # "bon motorina": TOTAL LEI 219.10, CARD 219.10 (thermal 0 looks like an 8)
    L("SUBTOTAL", 340, 560), L("219.10", 1000, 560),
    L("TOTAL LEI", 340, 640), L("219.10", 980, 640, 1110, 690),
    L("CARD", 340, 730), L("219.10", 1000, 730),
    L("REST", 340, 770), L("0.00", 1040, 770),
    L("TOTAL TVA A - 21%", 340, 830), L("38.03", 1020, 830),
    L("PAN:", 340, 990), L("XXXX-XXXX-XXXX-0673", 800, 990),
    L("DATA: 13-02-2026 ORA: 18:54:34", 400, 1680),
]
MOL = [  # no currency printed
    L("PLATA: CARD BANCAR", 230, 900), L("5372 11** **** 0673", 230, 940),
    L("SUBTOTAL:", 230, 1050), L("220,37", 1000, 1050),
    L("TOTAL:", 230, 1150, 360, 1210), L("220,37", 1000, 1150, 1120, 1210),
    L("CARD", 230, 1215), L("220,37", 1000, 1215),
    L("REST", 230, 1255), L("0,00", 1040, 1255),
    L("TOTAL TVA", 230, 1390), L("38,25", 1020, 1390),
]


def field(value, confidence="high", raw=None):
    return {"value": value, "raw": raw or (None if value is None else str(value)), "confidence": confidence,
            "page": 1 if value is not None else None, "bbox": None}


def test_rows_and_kinds():
    rows = group_rows(ROMPETROL)
    assert [" ".join(l.text for l in r) for r in rows][1] == "TOTAL LEI 219.10"
    assert row_kind("TOTAL LEI 219.10") == "total"
    assert row_kind("TOTAL DE PLATĂ 120,00") == "total"
    for not_total in ("SUBTOTAL 219.10", "TOTAL TVA A - 21% 38.03", "TOTAL TAXE: 38,04", "TOTAL TVA BON 38.03"):
        assert row_kind(not_total) != "total", not_total
    assert row_kind("CARTE CREDIT 219,20") == "payment"
    assert row_kind("REST 0,00") == "change"


def test_evidence_rompetrol():
    ev = build_evidence(ROMPETROL, page=0)
    assert [f.value for f in ev.totals] == [219.1]
    assert ev.totals[0].bbox == [980, 640, 1110, 690]
    assert ev.paid == 219.1
    assert [c.value for c in ev.currencies] == ["RON"]


def test_card_number_is_not_money():
    ev = build_evidence(MOL, page=0)
    assert ev.paid == 220.37  # "5372 11** **** 0673" ignored
    assert ev.currencies == []


def test_model_misread_payment_no_longer_raises_an_alarm():
    """bon motorina: model reads 219.10 everywhere but the CARD line as 219.18."""
    fld = field(219.1, confidence="low")  # lowered by the model's own payment read
    note = apply_total_witness(fld, [219.1, 219.1, 219.1], build_evidence(ROMPETROL, 0))
    assert fld["confidence"] == "medium" and fld["value"] == 219.1
    assert fld["bbox"] == [980, 640, 1110, 690]
    assert "confirmed by OCR" in note


def test_agreement_with_ocr_is_high_with_exact_box():
    fld = field(220.37, confidence="medium")
    assert apply_total_witness(fld, [220.37, 220.37], build_evidence(MOL, 0)) is None
    assert fld["confidence"] == "high" and fld["bbox"] == [1000, 1150, 1120, 1210]


def test_faded_total_caught_by_payment():
    """pag1: TOTAL LEI "245.1" (last digit faded, no 2 decimals for OCR), CARD 245.18."""
    lines = [L("TOTAL LE", 340, 640), L("245.1", 1000, 640), L("CARD", 340, 730), L("245.18", 1000, 730)]
    fld = field(245.1, confidence="high")
    note = apply_total_witness(fld, [245.1, 245.1, 245.1], build_evidence(lines, 0))
    assert fld["confidence"] == "low" and "245.18" in note


def test_minority_read_confirmed_by_ocr_and_payment_wins():
    lines = [L("TOTAL LEI", 340, 640), L("245.18", 1000, 640), L("CARD", 340, 730), L("245.18", 1000, 730)]
    fld = field(245.1, confidence="low")
    note = apply_total_witness(fld, [245.1, 245.18, 245.1], build_evidence(lines, 0))
    assert fld["value"] == 245.18 and fld["confidence"] == "medium"
    assert "confirmed by OCR" in note


def test_ocr_contradicts_model():
    lines = [L("TOTAL:", 230, 1150), L("120,00", 1000, 1150)]
    fld = field(102.0, confidence="high")
    note = apply_total_witness(fld, [102.0, 102.0], build_evidence(lines, 0))
    assert fld["confidence"] == "low" and "120.00" in note


def test_no_ocr_evidence_changes_nothing():
    fld = field(120.0, confidence="medium")
    assert apply_total_witness(fld, [120.0, 120.0], build_evidence([L("MULTUMIM", 300, 2000)], 0)) is None
    assert fld["confidence"] == "medium"


def test_currency_confirmed_on_total_row():
    fld = field("RON", confidence="medium", raw="LEI")
    assert apply_currency_witness(fld, build_evidence(ROMPETROL, 0)) is None
    assert fld["confidence"] == "high" and fld["bbox"] == ROMPETROL[2].bbox


def test_currency_missed_by_model_found_on_total_row():
    fld = {"value": None, "raw": None, "confidence": "none", "page": None, "bbox": None}
    note = apply_currency_witness(fld, build_evidence(ROMPETROL, 0))
    assert fld["value"] == "RON" and fld["confidence"] == "medium" and "OCR" in note


def test_currency_not_printed_stays_none():
    fld = {"value": None, "raw": None, "confidence": "none", "page": None, "bbox": None}
    assert apply_currency_witness(fld, build_evidence(MOL, 0)) is None
    assert fld["confidence"] == "none"


def test_currency_conflict_is_low():
    fld = field("EUR", confidence="high")
    apply_currency_witness(fld, build_evidence(ROMPETROL, 0))
    assert fld["confidence"] == "low"


# ─── End to end: the witness inside verify_pages ─────────────────────────────

def test_verify_pages_with_ocr_witness():
    from graph.verifier import verify_pages
    from tests.test_verifier import PRIMARY, SECOND, _page_b64
    page_lines = [(l.text, l.bbox) for l in [L("TOTAL:", 230, 1850), L("120,00", 412, 1880, 590, 1932),
                                             L("CARD", 230, 1960), L("120,00", 412, 1960)]]

    async def fake_ocr(image):
        return page_lines

    backend = FakeBackend(full_page(), full_page(absolute=True))
    resp = asyncio.run(verify_pages([_page_b64()], backend, PRIMARY, SECOND, max_side=1792, ocr=fake_ocr)).response
    assert resp["status"] == "ok"
    assert resp["fields"]["total_amount"]["confidence"] == "high"
    assert resp["fields"]["total_amount"]["bbox"] == [412, 1880, 590, 1932]


def test_ocr_failure_keeps_the_model_result():
    from graph.verifier import verify_pages
    from tests.test_verifier import PRIMARY, SECOND, _page_b64

    async def broken_ocr(image):
        raise RuntimeError("paddle down")

    backend = FakeBackend(full_page(), full_page(absolute=True))
    resp = asyncio.run(verify_pages([_page_b64()], backend, PRIMARY, SECOND, max_side=1792, ocr=broken_ocr)).response
    assert resp["status"] == "ok" and "OCR witness unavailable" in resp["notes"]


def test_tva_read_as_tua_is_not_a_total():
    """Real OCR of the Rompetrol receipts: "TOTAL TVA BON" comes out as "TOTAL TUA BON"."""
    assert row_kind("TOTAL TUA BON 38.03") != "total"
    lines = ROMPETROL + [L("TOTAL TUA BON", 340, 870), L("38.03", 1020, 870)]
    assert [f.value for f in build_evidence(lines, 0).totals] == [219.1]


def test_ocr_without_amounts_falls_back_to_the_models_payment_check():
    """pag1: OCR can read neither the faded TOTAL nor the payment; the model's own read of
    CARD 245.18 must still catch the faded total."""
    from graph.verifier import verify_pages
    from tests.test_verifier import PRIMARY, SECOND, _lines_with_payment, _page_b64

    async def blind_ocr(image):
        return [("ROMPETROL", [300, 100, 600, 140]), ("MULTUMIM", [400, 2000, 600, 2030])]

    full = lambda **kw: full_page(total="245.1", **kw)  # noqa: E731
    backend = FakeBackend(full(), full(absolute=True), lines=_lines_with_payment("245.1", "245.18"),
                          zoom={"total_amount": "245.1", "currency": "LEI", "receipt_date": "03.11.2026",
                                "document_number": "F-0042"})
    resp = asyncio.run(verify_pages([_page_b64()], backend, PRIMARY, SECOND, max_side=1792, ocr=blind_ocr)).response
    assert resp["status"] == "unreadable" and "245.18" in resp["notes"]
