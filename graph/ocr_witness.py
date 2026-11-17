"""OCR as an independent witness for the receipt verifier.

The vision model never sees OCR text: it reads the image blind. OCR runs separately on the
same page, and its lines are used only to confirm or contradict what the model read:

- the amount on the TOTAL row,
- what was paid (CARD / NUMERAR / CASH rows minus REST),
- the currency printed on the TOTAL row.

When those label rules can't tell which row is the final total (several TOTAL rows with different
amounts, or no keyword they know), the verifier may ask the model to label the OCR rows by meaning
(`prompt_rows` / `evidence_from_labels`). The model returns row numbers and kinds only; the amounts
still come from the OCR text, so it can't invent digits. Its choice of row is an AI judgment, so
evidence built from it never raises a field to "high".

When nothing works, a location witness reads the OCR text inside the model's box for the total:
no keywords, any layout.

Two technologies that fail differently (a vision model that may "see" a 0 as an 8 or invent a
plausible digit, an OCR engine that copies characters) agreeing is much stronger evidence
than several reads of the same model. OCR also gives exact boxes for the manager's highlight.
Everything here is plain code on OCR output (and on the labeler's row kinds); no model is called.
"""
import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Optional

from graph.normalize import normalize_amount

# Money as printed on receipts: always with 2 decimals ("219.10", "1.234,56"). Card numbers,
# codes and counters ("5372 11** **** 0673", "0109-00214") have none, so they never match.
_MONEY_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[.,\s]\d{3})*[.,]\d{2}(?![\d.,])|(?<![\d.,])\d+[.,]\d{2}(?![\d.,])")

_TOTAL_RE = re.compile(r"\bTOTAL\b")
# "TVA" is often OCR'd as "TUA" on thermal receipts ("TOTAL TUA BON 38.03").
# Totals without VAT: "Total HT", "Total netto", "Total excl. VAT", "Total exkl. MwSt", "imponibil".
_NOT_TOTAL_RE = re.compile(r"SUBTOTAL|\bT[VU]A\b|\bTAX|\bFARA\b|\bNET\b|\bDISCOUNT|\bREDUCERE"
                           r"|\bHT\b|\bNETTO\b|\bEXCL|\bEXKL|\bOHNE\b|\bIMPONIBIL")
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
    labelled: bool = False                      # the TOTAL rows were picked by the AI labeler

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


def _only_money(row: list[OcrLine]) -> bool:
    """A row with nothing but an amount and maybe a currency ("1239.25 RON")."""
    text = _plain(" ".join(l.text for l in row))
    if not _MONEY_RE.search(text):
        return False
    rest = _MONEY_RE.sub(" ", text)
    words = re.findall(r"[A-Z]+", rest)
    return all(w in _CURRENCY_WORDS for w in words) and not re.search(r"\d", rest)


def _value_below(rows: list[list[OcrLine]], i: int) -> Optional[list[OcrLine]]:
    """Invoices often print the amount under the label, in the same box:
        Total de plata
        1239.25 RON
    Returns the row right below a TOTAL label (close, overlapping it horizontally) when that row
    holds only an amount. Only the first such row counts: a second one is usually a conversion
    ("231.88 EUR")."""
    label = next(l for l in rows[i] if _TOTAL_RE.search(_plain(l.text)))
    if i + 1 >= len(rows):
        return None
    below = rows[i + 1]
    if below[0].yc - label.yc > 2.5 * label.height:
        return None
    width = label.bbox[2] - label.bbox[0]
    lo, hi = label.bbox[0] - width, label.bbox[2] + width
    near = [l for l in below if l.bbox[2] > lo and l.bbox[0] < hi]
    return near if near and _only_money(near) else None


def build_evidence(lines: list[OcrLine], page: int) -> OcrEvidence:
    ev = OcrEvidence(page)
    payments: set[float] = set()
    change = 0.0
    rows = group_rows(lines)
    for i, row in enumerate(rows):
        kind = row_kind(" ".join(l.text for l in row))
        if kind is None:
            continue
        money = _rightmost_money(row)
        if kind == "total":
            value_row = row
            if money is None:
                below = _value_below(rows, i)
                if below:
                    money, value_row = _rightmost_money(below), below
            if money:
                ev.totals.append(money)
            ev.currencies.extend(_currencies(row) + (_currencies(value_row) if value_row is not row else []))
        elif money and kind == "payment":
            payments.add(money.value)  # the same card payment is often printed twice
        elif money and kind == "change":
            change += money.value
    if payments:
        ev.paid = round(sum(payments) - change, 2)
    return ev


# ─── AI labeler: plain code around the model's row kinds ─────────────────────

MAX_PROMPT_ROWS = 150


def _distinct(found: list[Found]) -> list[float]:
    return list(dict.fromkeys(round(f.value, 2) for f in found))


def needs_labels(ev: OcrEvidence, lines: list[OcrLine]) -> bool:
    """The label rules can't name one final total: different amounts on several TOTAL rows (that
    no payment row settles), or no TOTAL row at all on a page that prints amounts."""
    values = _distinct(ev.totals)
    if len(values) == 1:
        return False
    if len(values) > 1:
        return not any(_eq(v, ev.paid) for v in values)
    return any(_MONEY_RE.search(l.text) for l in lines)


def prompt_rows(lines: list[OcrLine]) -> tuple[list[list[OcrLine]], str]:
    """The page's OCR rows, numbered, as the labeler sees them. A very long page keeps only the
    rows with an amount and the row above each (where a label printed over its amount sits)."""
    rows = group_rows(lines)
    texts = [" ".join(l.text for l in row) for row in rows]
    keep = list(range(len(rows)))
    if len(rows) > MAX_PROMPT_ROWS:
        money = [i for i, t in enumerate(texts) if _MONEY_RE.search(t)]
        keep = sorted({j for i in money for j in (i - 1, i) if j >= 0})[-MAX_PROMPT_ROWS:]
    return rows, "\n".join(f"[{i}] {texts[i][:200]}" for i in keep)


def evidence_from_labels(rows: list[list[OcrLine]], labels: list, page: int) -> OcrEvidence:
    """Evidence from the labeler's row kinds ({"row": i, "kind": ...}). The amounts are taken from
    the OCR text of each row, never from the model. Rows out of range or given two kinds are ignored."""
    kinds: dict[int, str] = {}
    conflicting: set[int] = set()
    for item in labels or []:
        if not isinstance(item, dict):
            continue
        i, kind = item.get("row"), item.get("kind")
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(rows) or not isinstance(kind, str):
            continue
        if kinds.get(i, kind) != kind:
            conflicting.add(i)
        kinds[i] = kind

    ev = OcrEvidence(page, labelled=True)
    payments: set[float] = set()
    change = 0.0
    for i in sorted(kinds.keys() - conflicting):
        kind, row = kinds[i], rows[i]
        money, value_row = _rightmost_money(row), row
        if money is None and kind == "total" and i + 1 < len(rows) and i + 1 not in kinds \
                and _only_money(rows[i + 1]):
            money, value_row = _rightmost_money(rows[i + 1]), rows[i + 1]  # amount under its label
        if money is None:
            continue
        if kind == "total":
            ev.totals.append(money)
            ev.currencies.extend(_currencies(value_row) + (_currencies(row) if value_row is not row else []))
        elif kind == "payment":
            payments.add(money.value)
        elif kind == "change":
            change += money.value
    if payments:
        ev.paid = round(sum(payments) - change, 2)
    return ev


# ─── Applying the evidence ───────────────────────────────────────────────────

LABELLED_NOTE = "total_amount: confirmed by OCR on the row the AI labeler picked as the final total"


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
    values = _distinct(ev.totals)
    if len(values) > 1:
        # Several different amounts on TOTAL rows (before/after a discount, net and gross, a total
        # and its conversion): OCR can't tell which one is final, and confirming whichever the
        # model read would turn its mistake into "high". Only a matching payment settles it.
        final = [f for f in ev.totals if _eq(f.value, ev.paid)]
        if not final:
            if value is not None:
                fld["confidence"] = "low"
            return (f"total_amount: several totals on the page ({', '.join(f'{v:.2f}' for v in values)}) "
                    "and nothing tells which one is final")
        ev = replace(ev, totals=final)

    on_total = _find(ev.totals, value) if value is not None else None
    paid_ok = ev.paid is None or _eq(ev.paid, value)
    capped = False

    def take(found: Found, confidence: str) -> None:
        nonlocal capped
        if ev.labelled and confidence == "high":  # an AI picked the row: not "high"
            confidence, capped = "medium", True
        fld.update(value=found.value, raw=found.raw, confidence=confidence, page=ev.page + 1, bbox=found.bbox)

    if on_total:
        # Model and OCR read the same amount on the TOTAL row: two technologies agree.
        was = fld["confidence"]
        if paid_ok:
            take(Found(value, fld["raw"] or on_total.raw, on_total.bbox),
                 "high" if was in ("high", "medium") else "medium")
            if was == "low":
                return "total_amount: model reads disagreed, confirmed by OCR"
            return LABELLED_NOTE if capped else None
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


def _intersection(a: list[int], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _area(b: list[int]) -> float:
    return max(1, b[2] - b[0]) * max(1, b[3] - b[1])


def apply_location_witness(fld: dict, boxes: list[list[int]], lines: list[OcrLine],
                           page: int) -> tuple[bool, Optional[str]]:
    """Layout-free witness: OCR reads the text where the model says the total is printed.

    Used when the label rules above find nothing (an unknown layout or language: "Gesamtbetrag",
    "Amount due", a total in a table column). It needs no keywords, only the model's box.
    It proves the model read the digits right, not that it picked the right line (it would
    confirm a subtotal too), so it never raises the confidence: it only counts as the
    independent confirmation a total needs. Returns (OCR judged the total, note)."""
    value = fld["value"]
    if not isinstance(value, float) or not boxes:
        return False, None
    seen: list[float] = []
    for box in boxes:
        w, h = box[2] - box[0], box[3] - box[1]
        # Model boxes are loose: look a little around them.
        grown = [box[0] - 0.15 * w - 5, box[1] - 0.5 * h - 5, box[2] + 0.15 * w + 5, box[3] + 0.5 * h + 5]
        for line in lines:
            if not _intersection(line.bbox, grown):
                continue
            amounts = [a for a in map(normalize_amount, _MONEY_RE.findall(line.text)) if a is not None]
            if any(_eq(a, value) for a in amounts):
                if fld["bbox"] is None:  # an exact box for the manager's highlight
                    fld.update(page=page + 1, bbox=line.bbox)
                return True, None
            if _intersection(line.bbox, box) >= 0.5 * _area(line.bbox):
                seen.extend(amounts)
    if seen:
        # OCR reads another amount right where the model read its total: a misread digit.
        fld["confidence"] = "low"
        return True, (f"total_amount: the model read {value:.2f}, OCR reads "
                      f"{', '.join(f'{a:.2f}' for a in dict.fromkeys(seen))} at the same place")
    return False, None


def apply_currency_witness(fld: dict, ev: OcrEvidence) -> Optional[str]:
    """Confirm the currency with the one OCR reads on the TOTAL row ("TOTAL LEI"). A currency
    the model did not find is filled in only from the TOTAL row, never from elsewhere."""
    if not ev.currencies:
        return None
    value = fld["value"]
    codes = list(dict.fromkeys(f.value for f in ev.currencies))
    if len(codes) > 1:
        # A total and its conversion ("TOTAL EUR" / "TOTAL RON"): OCR can't tell which currency
        # the document is in. "low" means Spendle doesn't compare it.
        if value is not None:
            fld["confidence"] = "low"
            return f"currency: several currencies on the total rows ({', '.join(codes)})"
        return None
    if value is None:
        f = ev.currencies[0]
        fld.update(value=f.value, raw=f.raw, confidence="medium", page=ev.page + 1, bbox=f.bbox)
        return "currency: found by OCR on the TOTAL row only"
    found = _find(ev.currencies, value)
    if found:
        fld.update(confidence="medium" if ev.labelled else "high", page=ev.page + 1, bbox=found.bbox)
        return None
    fld["confidence"] = "low"
    return f"currency: the model read {value}, OCR reads {ev.currencies[0].value} on the TOTAL row"
