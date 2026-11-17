"""OCR witness: confirms/contradicts the model's total and currency with OCR lines.
Lines mimic the benchmark receipts (Rompetrol, MOL, OMV) at page-pixel positions."""
import asyncio

import pytest

from graph.ocr_witness import (
    LABELLED_NOTE, OcrLine, apply_currency_witness, apply_location_witness, apply_total_witness,
    build_evidence, evidence_from_labels, group_rows, needs_labels, prompt_rows, row_kind,
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


# Hotel invoice: the amount is printed under "Total de plata", then converted to EUR.
INVOICE = [
    L("Modalitate plata: Card", 251, 1122),
    L("Curs valutar 1 EUR = 5.3443 RON", 251, 1141),
    L("Semnatura si stampila furnizorului", 255, 1296), L("Semnatura de", 900, 1296),
    L("1120.41", 1113, 1296, 1190, 1316), L("118.84", 1270, 1296, 1330, 1316),
    L("primire", 900, 1315, 980, 1335), L("Total de plata", 1230, 1315, 1370, 1335),
    L("1239.25 RON", 1232, 1337, 1368, 1357),
    L("231.88 EUR", 1245, 1358, 1355, 1378),
    L("NET", 405, 1391, 440, 1411), L("TVA", 535, 1391, 570, 1411), L("TOTAL", 705, 1391, 765, 1411),
    L("11.00%", 255, 1432, 320, 1452), L("1080.41", 370, 1432, 450, 1452),
    L("118.84", 520, 1432, 585, 1452), L("1199.25", 700, 1432, 770, 1452),
]


def test_amount_printed_under_the_total_label():
    ev = build_evidence(INVOICE, 0)
    assert [f.value for f in ev.totals] == [1239.25]       # not the EUR conversion, not 1199.25
    assert [f.value for f in ev.currencies] == ["RON"]
    assert ev.paid is None                                 # "Card" without an amount
    fld = field(1239.25, "medium")
    assert apply_total_witness(fld, [1239.25, 1239.25], ev) is None
    assert fld["confidence"] == "high" and fld["bbox"] == [1232, 1337, 1368, 1357]


def test_text_under_the_total_label_is_not_an_amount():
    lines = [L("TOTAL", 300, 500, 400, 530), L("Va multumim 12.50", 300, 535, 600, 565)]
    assert build_evidence(lines, 0).totals == []


# ─── Location witness: OCR reads the text where the model located the total ──

AT_TOTAL = [412, 1880, 590, 1932]   # where full_page() puts the total


def test_location_confirms_the_digits_and_gives_an_exact_box():
    fld = field(1239.25, "medium")
    lines = [L("Gesamtbetrag", 1000, 1300, 1200, 1330), L("1.239,25 EUR", 1230, 1338, 1370, 1358)]
    assert apply_location_witness(fld, [[1220, 1330, 1380, 1365]], lines, 0) == (True, None)
    assert fld["confidence"] == "medium"                     # never raised: it may be a subtotal
    assert fld["bbox"] == [1230, 1338, 1370, 1358] and fld["page"] == 1


def test_location_contradicts_a_misread_digit():
    fld = field(1289.25, "medium")
    lines = [L("1239.25 RON", 1232, 1337, 1368, 1357)]
    judged, note = apply_location_witness(fld, [[1225, 1332, 1375, 1362]], lines, 0)
    assert judged and fld["confidence"] == "low" and "1239.25" in note


def test_location_without_ocr_text_there_judges_nothing():
    fld = field(1239.25, "medium")
    lines = [L("Hotel Sibiu", 100, 100, 400, 130), L("40.00", 700, 700, 770, 720)]
    assert apply_location_witness(fld, [[1225, 1332, 1375, 1362]], lines, 0) == (False, None)
    assert fld["confidence"] == "medium"


def test_unknown_layout_confirmed_by_location():
    """No TOTAL keyword OCR knows ("Gesamtbetrag") and read B fails: before, the total had no
    independent confirmation and the invoice came back unreadable."""
    from graph.verifier import verify_pages
    from tests.test_verifier import PRIMARY, _page_b64

    async def ocr(image):
        return [("Gesamtbetrag", [200, 1880, 380, 1932]), ("120,00", AT_TOTAL)]

    backend = FakeBackend(full_page(), full_page(), fail={(PRIMARY, "lines")})
    resp = asyncio.run(verify_pages([_page_b64()], backend, PRIMARY, None, max_side=1792, ocr=ocr)).response
    assert resp["status"] == "ok"
    assert resp["fields"]["total_amount"]["confidence"] == "medium"
    assert resp["fields"]["total_amount"]["bbox"] == AT_TOTAL


def test_unknown_layout_with_misread_total_is_unreadable():
    from graph.verifier import verify_pages
    from tests.test_verifier import PRIMARY, _page_b64

    async def ocr(image):
        return [("Gesamtbetrag", [200, 1880, 380, 1932]), ("128,00", AT_TOTAL)]

    backend = FakeBackend(full_page(), full_page(), fail={(PRIMARY, "lines")})
    resp = asyncio.run(verify_pages([_page_b64()], backend, PRIMARY, None, max_side=1792, ocr=ocr)).response
    assert resp["status"] == "unreadable" and "128.00" in resp["notes"]


# ─── Several TOTAL rows: OCR alone can't tell which is final ─────────────────

# Romanian invoice with a discount: the final total is the second TOTAL row.
DISCOUNT = [
    L("Total", 230, 1800, 330, 1830), L("120,00", 412, 1800, 590, 1830),
    L("Reducere 10%", 230, 1840, 400, 1870), L("-20,00", 412, 1840, 590, 1870),
    L("Total de plata", 230, 1880, 400, 1932), L("100,00", 412, 1880, 590, 1932),
]
DISCOUNT_LABELS = [{"row": 0, "kind": "subtotal"}, {"row": 1, "kind": "discount"}, {"row": 2, "kind": "total"}]


def test_totals_without_vat_are_not_final_totals():
    for net in ("Total HT 100,00", "TOTAL NETTO 100,00", "Total excl. VAT 100,00", "Total exkl. MwSt 100,00",
                "Total ohne MwSt 100,00", "Total imponibil 100,00"):
        assert row_kind(net) != "total", net
    for final in ("Total TTC 120,00", "TOTAL DE PLATA 120,00", "Total incl. VAT 120,00"):
        assert row_kind(final) == "total", final


@pytest.mark.parametrize("model_value", [120.0, 100.0])
def test_several_totals_confirm_nothing(model_value):
    """Before: OCR confirmed whichever TOTAL row the model read, even the one before the discount,
    as "high", which made Spendle raise a false "needs a look"."""
    ev = build_evidence(DISCOUNT, 0)
    assert [f.value for f in ev.totals] == [120.0, 100.0]
    fld = field(model_value, "high")
    note = apply_total_witness(fld, [model_value], ev)
    assert fld["confidence"] == "low" and "120.00, 100.00" in note


def test_payment_tells_which_total_is_final():
    ev = build_evidence(DISCOUNT + [L("CARD", 230, 1960), L("100,00", 412, 1960)], 0)
    fld = field(100.0, "high")
    assert apply_total_witness(fld, [100.0], ev) is None and fld["confidence"] == "high"
    fld = field(120.0, "high")
    apply_total_witness(fld, [120.0], ev)
    assert fld["confidence"] == "low"


def test_two_currencies_on_total_rows_are_not_confirmed():
    ev = build_evidence([L("TOTAL EUR", 230, 500), L("100,00", 600, 500),
                         L("TOTAL RON", 230, 560), L("497,50", 600, 560)], 0)
    cur = field("EUR", "high")
    assert "EUR, RON" in apply_currency_witness(cur, ev) and cur["confidence"] == "low"


# ─── When the AI labeler is asked ────────────────────────────────────────────

def test_labeler_only_when_the_rules_are_unsure():
    assert not needs_labels(build_evidence(ROMPETROL, 0), ROMPETROL)          # one TOTAL row
    assert needs_labels(build_evidence(DISCOUNT, 0), DISCOUNT)                # two different totals
    paid = DISCOUNT + [L("CARD", 230, 1960), L("100,00", 412, 1960)]
    assert not needs_labels(build_evidence(paid, 0), paid)                    # the payment settles it
    unknown = [L("Gesamtbetrag", 200, 500), L("119,00", 600, 500)]
    assert needs_labels(build_evidence(unknown, 0), unknown)                  # no keyword it knows
    no_amounts = [L("ROMPETROL", 300, 100), L("MULTUMIM", 300, 2000)]
    assert not needs_labels(build_evidence(no_amounts, 0), no_amounts)        # nothing to label


def test_prompt_rows_are_numbered_and_long_pages_keep_the_amount_rows():
    rows, text = prompt_rows(DISCOUNT)
    assert text.splitlines() == ["[0] Total 120,00", "[1] Reducere 10% -20,00", "[2] Total de plata 100,00"]
    long_page = [L(f"Produs {i}", 100, 100 + 40 * i) for i in range(200)] + [
        L("Total de plata", 100, 9000), L("55,00", 600, 9000)]
    rows, text = prompt_rows(long_page)
    assert len(rows) == 201 and text.splitlines() == ["[199] Produs 199", "[200] Total de plata 55,00"]


# ─── Evidence from the labeler's row kinds ───────────────────────────────────

def test_labels_take_the_digits_from_ocr():
    ev = evidence_from_labels(group_rows(DISCOUNT), DISCOUNT_LABELS, 0)
    assert ev.labelled and [f.value for f in ev.totals] == [100.0]
    assert ev.totals[0].bbox == [412, 1880, 590, 1932]


def test_labels_amount_under_its_label_and_payment():
    lines = [L("Gesamtbetrag", 200, 500, 400, 530), L("119,00 EUR", 200, 540, 400, 570),
             L("Bar", 200, 620), L("120,00", 600, 620), L("Rückgeld", 200, 660), L("1,00", 600, 660)]
    labels = [{"row": 0, "kind": "total"}, {"row": 2, "kind": "payment"}, {"row": 3, "kind": "change"}]
    ev = evidence_from_labels(group_rows(lines), labels, 0)
    assert [f.value for f in ev.totals] == [119.0] and ev.paid == 119.0
    assert [c.value for c in ev.currencies] == ["EUR"]


def test_unusable_labels_are_ignored():
    rows = group_rows(DISCOUNT)
    labels = [{"row": 9, "kind": "total"}, {"row": True, "kind": "total"}, {"row": "2", "kind": "total"},
              "total", {"row": 0, "kind": "total"}, {"row": 0, "kind": "subtotal"}]   # row 0: two kinds
    assert evidence_from_labels(rows, labels, 0).totals == []


def test_labelled_confirmation_is_medium_not_high():
    """The AI picked the row, so agreement is not "high"; the digits are still OCR's."""
    ev = evidence_from_labels(group_rows(DISCOUNT), DISCOUNT_LABELS, 0)
    fld = field(100.0, "high")
    assert apply_total_witness(fld, [100.0], ev) == LABELLED_NOTE
    assert fld["confidence"] == "medium" and fld["bbox"] == [412, 1880, 590, 1932]
    fld = field(120.0, "high")
    apply_total_witness(fld, [120.0], ev)
    assert fld["confidence"] == "low"


# ─── End to end: verify_pages with the labeler ───────────────────────────────

def _run_labelled(total, labels=None, fail=(), label_ocr=True, lines=DISCOUNT):
    from graph.verifier import verify_pages
    from tests.test_verifier import PRIMARY, _page_b64, amount_lines

    async def ocr(image):
        return [(l.text, l.bbox) for l in lines]

    backend = FakeBackend(full_page(total=total), full_page(total=total), lines=amount_lines(total),
                          zoom={"total_amount": total, "currency": "LEI", "receipt_date": "03.11.2026",
                                "document_number": "F-0042"},
                          labels={"rows": DISCOUNT_LABELS if labels is None else labels}, fail=fail)
    resp = asyncio.run(verify_pages([_page_b64()], backend, PRIMARY, None, max_side=1792, ocr=ocr,
                                    label_ocr=label_ocr)).response
    return resp, backend


def test_discount_invoice_total_before_discount_is_caught():
    """Every model read takes the amount before the discount; OCR + labeler say 100,00 is final."""
    resp, backend = _run_labelled("120,00")
    assert ("qwen3-vl:8b-thinking-q8_0", "labels") in backend.calls
    assert resp["status"] == "unreadable" and resp["fields"]["total_amount"]["confidence"] == "low"
    assert "100.00" in resp["notes"]


def test_discount_invoice_final_total_is_confirmed():
    resp, _ = _run_labelled("100,00")
    total = resp["fields"]["total_amount"]
    assert resp["status"] == "ok" and total["value"] == 100.0 and total["confidence"] == "medium"
    assert total["bbox"] == AT_TOTAL and LABELLED_NOTE in resp["notes"]


def test_labeler_sees_ocr_text_only_and_vision_reads_never_see_it():
    _, backend = _run_labelled("100,00")
    labelled = [i for i, (_, kind) in enumerate(backend.calls) if kind == "labels"]
    assert len(labelled) == 1 and backend.images[labelled[0]] is None
    assert "Reducere" in backend.prompts[labelled[0]]
    for i, (_, kind) in enumerate(backend.calls):
        if kind != "labels":
            assert backend.images[i] is not None and "Reducere" not in backend.prompts[i]


def test_labeler_not_called_when_the_rules_are_sure():
    resp, backend = _run_labelled("120,00", lines=[L("TOTAL:", 230, 1880, 360, 1932), L("120,00", 412, 1880, 590, 1932)])
    assert all(kind != "labels" for _, kind in backend.calls)
    assert resp["fields"]["total_amount"]["confidence"] == "high"


def test_labeler_failure_leaves_the_total_unconfirmed():
    from tests.test_verifier import PRIMARY
    resp, _ = _run_labelled("100,00", fail={(PRIMARY, "labels")})
    assert resp["status"] == "unreadable" and "OCR labeler failed" in resp["notes"]


def test_without_labeler_several_totals_are_unreadable():
    resp, backend = _run_labelled("100,00", label_ocr=False)
    assert all(kind != "labels" for _, kind in backend.calls)
    assert resp["status"] == "unreadable" and "nothing tells which one is final" in resp["notes"]


def test_unknown_language_confirmed_through_the_labeler():
    lines = [L("Gesamtbetrag", 230, 1880, 400, 1932), L("120,00", 412, 1880, 590, 1932)]
    resp, _ = _run_labelled("120,00", lines=lines, labels=[{"row": 0, "kind": "total"}])
    total = resp["fields"]["total_amount"]
    assert resp["status"] == "ok" and total["confidence"] == "medium" and total["bbox"] == AT_TOTAL
