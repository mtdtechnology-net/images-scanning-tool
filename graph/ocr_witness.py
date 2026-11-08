"""OCR as an independent witness for the receipt verifier.

The vision model never sees OCR text: it reads the image blind. OCR runs separately on the
same page, and its lines are used only to confirm or contradict what the model read:

- the amount on the TOTAL row,
- what was paid (CARD / NUMERAR / CASH rows minus REST),
- the currency printed on the TOTAL row.

Two technologies that fail differently (a vision model that may "see" a 0 as an 8 or invent a
plausible digit, an OCR engine that copies characters) agreeing is much stronger evidence
than several reads of the same model. OCR also gives exact boxes for the manager's highlight.
Everything here is plain code on OCR output; no model is involved.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from graph.normalize import normalize_amount

# Money as printed on receipts: always with 2 decimals ("219.10", "1.234,56"). Card numbers,
# codes and counters ("5372 11** **** 0673", "0109-00214") have none, so they never match.
_MONEY_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[.,\s]\d{3})*[.,]\d{2}(?![\d.,])|(?<![\d.,])\d+[.,]\d{2}(?![\d.,])")

_TOTAL_RE = re.compile(r"\bTOTAL\b")
# "TVA" is often OCR'd as "TUA" on thermal receipts ("TOTAL TUA BON 38.03").
_NOT_TOTAL_RE = re.compile(r"SUBTOTAL|\bT[VU]A\b|\bTAX|\bFARA\b|\bNET\b|\bDISCOUNT|\bREDUCERE")
_CHANGE_RE = re.compile(r"\bREST\b|\bCHANGE\b")
_PAYMENT_RE = re.compile(r"\bCARD\b|\bNUMERAR\b|\bCASH\b|\bCARTE\s+CREDIT\b|\bTICHET|\bVOUCHER\b")

# Currency words trusted from OCR (3+ letters or a symbol, so addresses and codes don't match).
_CURRENCY_WORDS = {"RON": "RON", "LEI": "RON", "LEU": "RON", "EUR": "EUR", "EURO": "EUR", "USD": "USD",
                   "GBP": "GBP", "CHF": "CHF", "HUF": "HUF", "PLN": "PLN", "CZK": "CZK", "BGN": "BGN",
                   "MDL": "MDL"}
_CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP", "$": "USD"}


@dataclass
class OcrLine:
    text: str
    bbox: list[int]  # [x0, y0, x1, y1] on the page image

    @property
    def yc(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])


@dataclass
class Found:
    value: object
    raw: str
    bbox: list[int]


@dataclass
class OcrEvidence:
    page: int                                   # 0-based
    totals: list[Found] = field(default_factory=list)      # amounts on TOTAL rows
    paid: Optional[float] = None                # payment rows − change rows
    currencies: list[Found] = field(default_factory=list)  # currencies on TOTAL rows

    @property
    def has_amounts(self) -> bool:
        """OCR found a TOTAL amount or a payment: it can judge the total."""
        return bool(self.totals) or self.paid is not None


def _plain(text: str) -> str:
    """Upper case without diacritics, so "TOTAL DE PLATĂ" and "TOTAL DE PLATA" match alike."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).upper()


def group_rows(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Group OCR lines into printed rows (same height on the page), each sorted left to right."""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda l: l.yc):
        if rows:
            row = rows[-1]
            ref = sum(l.yc for l in row) / len(row)
            tol = 0.6 * min(line.height, min(l.height for l in row))
            if abs(line.yc - ref) <= tol:
                row.append(line)
                continue
        rows.append([line])
    return [sorted(row, key=lambda l: l.bbox[0]) for row in rows]


def _rightmost_money(row: list[OcrLine]) -> Optional[Found]:
    """The amount printed at the right end of a row (where receipts put the value)."""
    for line in reversed(row):
        matches = _MONEY_RE.findall(line.text)
        if matches:
            value = normalize_amount(matches[-1])
            if value is not None:
                return Found(value, matches[-1].strip(), line.bbox)
    return None


def row_kind(text: str) -> Optional[str]:
    t = _plain(text)
    if _TOTAL_RE.search(t) and not _NOT_TOTAL_RE.search(t):
        return "total"
    if _CHANGE_RE.search(t):
        return "change"
    if _PAYMENT_RE.search(t):
        return "payment"
    return None


def _currencies(row: list[OcrLine]) -> list[Found]:
    found = []
    for line in row:
        for word in re.findall(r"[A-Za-z]+|[€£$]", _plain(line.text)):
            iso = _CURRENCY_WORDS.get(word) or _CURRENCY_SYMBOLS.get(word)
            if iso:
                found.append(Found(iso, word, line.bbox))
    return found


def build_evidence(lines: list[OcrLine], page: int) -> OcrEvidence:
    ev = OcrEvidence(page)
    payments: set[float] = set()
    change = 0.0
    for row in group_rows(lines):
        kind = row_kind(" ".join(l.text for l in row))
        if kind is None:
            continue
        money = _rightmost_money(row)
        if kind == "total":
            if money:
                ev.totals.append(money)
            ev.currencies.extend(_currencies(row))
        elif money and kind == "payment":
            payments.add(money.value)  # the same card payment is often printed twice
        elif money and kind == "change":
            change += money.value
    if payments:
        ev.paid = round(sum(payments) - change, 2)
    return ev


def _eq(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and abs(a - b) < 0.005


def _find(found: list[Found], value) -> Optional[Found]:
    for f in found:
        if (isinstance(value, float) and _eq(f.value, value)) or f.value == value:
            return f
    return None


def apply_total_witness(fld: dict, read_values: list[float], ev: OcrEvidence) -> Optional[str]:
    """Confirm or contradict the total the vision model read. Updates `fld` in place
    (value, raw, confidence, page, bbox) and returns a note when it isn't a plain confirmation."""
    value = fld["value"]
    on_total = _find(ev.totals, value) if value is not None else None
    paid_ok = ev.paid is None or _eq(ev.paid, value)

    def take(found: Found, confidence: str) -> None:
        fld.update(value=found.value, raw=found.raw, confidence=confidence, page=ev.page + 1, bbox=found.bbox)

    if on_total:
        # Model and OCR read the same amount on the TOTAL row: two technologies agree.
        was = fld["confidence"]
        if paid_ok:
            take(Found(value, fld["raw"] or on_total.raw, on_total.bbox),
                 "high" if was in ("high", "medium") else "medium")
            return "total_amount: model reads disagreed, confirmed by OCR" if was == "low" else None
        # A different payment amount alone doesn't overturn that (payment lines get misread
        # too, e.g. a thermal "0" taken for an "8"): keep it, at medium, with a note.
        take(Found(value, fld["raw"] or on_total.raw, on_total.bbox), "medium")
        return (f"total_amount: model and OCR read {value:.2f} on the TOTAL row; payment minus "
                f"change reads {ev.paid:.2f}")

    # The model's majority value is not what OCR reads on the TOTAL row. If one of the model's
    # reads matches OCR and the payment backs it, that value is printed twice on the receipt.
    for candidate in sorted(set(read_values), key=read_values.count, reverse=True):
        found = _find(ev.totals, candidate)
        if found and ev.paid is not None and _eq(ev.paid, candidate):
            take(found, "medium")
            return (f"total_amount: {candidate:.2f} confirmed by OCR (TOTAL row and payment); "
                    f"most model reads said {value:.2f}" if value is not None else
                    f"total_amount: {candidate:.2f} confirmed by OCR (TOTAL row and payment)")

    if value is None:
        if len({f.value for f in ev.totals}) == 1 and (ev.paid is None or _eq(ev.paid, ev.totals[0].value)):
            take(ev.totals[0], "low")
            return "total_amount: found by OCR only"
        return None

    if ev.totals:
        fld["confidence"] = "low"
        return (f"total_amount: the model read {value:.2f}, OCR reads "
                f"{', '.join(f'{f.value:.2f}' for f in ev.totals)} on the TOTAL row")

    # OCR found no readable TOTAL row (faded, cut): the payment is the only witness.
    if ev.paid is not None:
        if _eq(ev.paid, value):
            if fld["confidence"] == "low":
                fld["confidence"] = "medium"
            return "total_amount: TOTAL row unreadable for OCR, confirmed by the payment"
        fld["confidence"] = "low"
        return f"total_amount: the model read {value:.2f}, payment minus change (OCR) is {ev.paid:.2f}"
    return None


def apply_currency_witness(fld: dict, ev: OcrEvidence) -> Optional[str]:
    """Confirm the currency with the one OCR reads on the TOTAL row ("TOTAL LEI"). A currency
    the model did not find is filled in only from the TOTAL row, never from elsewhere."""
    if not ev.currencies:
        return None
    value = fld["value"]
    if value is None:
        f = ev.currencies[0]
        fld.update(value=f.value, raw=f.raw, confidence="medium", page=ev.page + 1, bbox=f.bbox)
        return "currency: found by OCR on the TOTAL row only"
    found = _find(ev.currencies, value)
    if found:
        fld.update(confidence="high", page=ev.page + 1, bbox=found.bbox)
        return None
    fld["confidence"] = "low"
    return f"currency: the model read {value}, OCR reads {ev.currencies[0].value} on the TOTAL row"
