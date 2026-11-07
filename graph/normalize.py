"""Normalization of values read from receipts, so reads can be compared exactly.

- amount:          "1.234,56" / "1,234.56" / "120,00 LEI" / "120,-"  -> 1234.56 / 120.0
- currency:        "LEI" / "lei" / "RON" / "€"                       -> "RON" / "EUR" (ISO 4217)
- date:            "03.11.2026" / "2026-11-03" / "3 noiembrie 2026"   -> "2026-11-03"
- document number: "Nr. F-0042"                                       -> "F-0042"

Every function returns None when the text holds no usable value.
"""
import datetime
import re
from typing import Optional

from graph.nodes import _parse_flexible_date

# ─── Amount ──────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"[-−]?\d[\d.,'’\s ]*\d|[-−]?\d")


def normalize_amount(text: Optional[str]) -> Optional[float]:
    """Parse a printed amount, whatever the thousands/decimal convention."""
    if text is None:
        return None
    s = str(text).strip()
    s = re.sub(r"[,.]\s*-+$", "", s)  # "120,-" / "120.-"
    matches = _NUMBER_RE.findall(s)
    if not matches:
        return None
    token = matches[-1]  # the value is printed after any label ("TOTAL 120,00")
    negative = token[0] in "-−"
    digits = re.sub(r"[^\d.,]", "", token)

    if "." in digits and "," in digits:
        decimal = "." if digits.rfind(".") > digits.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        digits = digits.replace(thousands, "").replace(decimal, ".")
    elif "." in digits or "," in digits:
        sep = "." if "." in digits else ","
        parts = digits.split(sep)
        if len(parts) > 2:
            digits = "".join(parts)  # "1.234.567" -> thousands only
        elif len(parts[1]) == 3 and parts[0] not in ("", "0") and len(parts[0]) <= 3:
            digits = "".join(parts)  # "1.234" -> 1234 (amounts carry 2 decimals)
        else:
            digits = parts[0] + "." + parts[1]

    try:
        value = float(digits)
    except ValueError:
        return None
    return round(-value if negative else value, 2)


# ─── Currency ────────────────────────────────────────────────────────────────

_CURRENCY_ALIASES = {
    "RON": ["RON", "LEI", "LEU", "LEI RON", "RON LEI"],
    "EUR": ["EUR", "EURO", "EUROS", "€"],
    "USD": ["USD", "US$", "$", "US DOLLAR", "DOLLAR", "DOLLARS"],
    "GBP": ["GBP", "£", "POUND", "POUNDS"],
    "CHF": ["CHF", "FR.", "SFR"],
    "HUF": ["HUF", "FT", "FORINT"],
    "PLN": ["PLN", "ZŁ", "ZL", "ZLOTY"],
    "CZK": ["CZK", "KČ", "KC"],
    "BGN": ["BGN", "ЛВ", "ЛВ.", "LEVA", "LV"],
    "MDL": ["MDL"],
    "TRY": ["TRY", "₺", "TL"],
    "SEK": ["SEK"],
    "NOK": ["NOK"],
    "DKK": ["DKK"],
    "JPY": ["JPY", "¥", "YEN"],
}
_ALIAS_TO_ISO = {alias: iso for iso, aliases in _CURRENCY_ALIASES.items() for alias in aliases}
# Symbols can be glued to the amount ("120,00€"), so look for them anywhere.
_SYMBOLS = {"€": "EUR", "£": "GBP", "$": "USD", "₺": "TRY", "¥": "JPY"}


def normalize_currency(text: Optional[str]) -> Optional[str]:
    """Map a printed currency onto its ISO 4217 code."""
    if text is None:
        return None
    s = re.sub(r"\s+", " ", str(text)).strip().upper()
    if not s:
        return None
    if s in _ALIAS_TO_ISO:
        return _ALIAS_TO_ISO[s]
    s_clean = re.sub(r"[\d.,:;()\-]+", " ", s).strip()
    s_clean = re.sub(r"\s+", " ", s_clean)
    if s_clean in _ALIAS_TO_ISO:
        return _ALIAS_TO_ISO[s_clean]
    for word in s_clean.split(" "):
        if word in _ALIAS_TO_ISO and len(word) >= 2:
            return _ALIAS_TO_ISO[word]
    for symbol, iso in _SYMBOLS.items():
        if symbol in s:
            return iso
    return None


# ─── Date ────────────────────────────────────────────────────────────────────

_DATE_SEARCH = [
    re.compile(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"),                     # 2026-11-03
    re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}"),                     # 03.11.2026
    re.compile(r"\d{1,2}[\s\-./]+[A-Za-zăîșțâĂÎȘȚÂ]{3,}\.?[\s\-./]+\d{4}"),  # 3 noiembrie 2026
    re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2}(?!\d)"),               # 03.11.26
]


def _plausible(d: datetime.date) -> bool:
    return 2000 <= d.year <= 2100


def normalize_date(text: Optional[str]) -> Optional[str]:
    """Parse a printed date (day first, as on Romanian/European receipts) into YYYY-MM-DD."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    candidates = [s] + [m.group(0) for p in _DATE_SEARCH for m in p.finditer(s)]
    for c in candidates:
        d = _parse_flexible_date(c.strip())
        if d is None:
            # US order (11/23/2026) is the only reading when the "month" is > 12.
            m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", c.strip())
            if m and int(m.group(2)) > 12 >= int(m.group(1)):
                try:
                    d = datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                except ValueError:
                    d = None
        if d is not None and _plausible(d):
            return d.isoformat()
    return None


# ─── Document number ─────────────────────────────────────────────────────────

_DOC_LABEL_RE = re.compile(
    r"^(?:(?:(?:bon\s+fiscal|bon|factur[aă](?:\s+fiscal[aă])?|invoice|receipt|chitan[tț][aă]|"
    r"document|doc|seria|serie|series|nr|no|num[aă]r|number)\b|n°|#)[\s.:#°]*)+",
    re.IGNORECASE,
)


def normalize_document_number(text: Optional[str]) -> Optional[str]:
    """Strip labels ("Nr.", "Factura nr", "#") and spacing from a document number."""
    if text is None:
        return None
    s = re.sub(r"\s+", " ", str(text)).strip()
    s = _DOC_LABEL_RE.sub("", s).strip(" .:#-")
    s = re.sub(r"\s+(?:nr|no)\b[\s.:]*", " ", s, flags=re.IGNORECASE)  # "ABC nr 0042" -> "ABC 0042"
    s = s.upper().strip()
    return s or None


def document_number_key(value: Optional[str]) -> Optional[str]:
    """Loose key: only letters/digits, leading zeros dropped in each number.
    "F-0042" and "F 42" share a key (small formatting difference)."""
    if not value:
        return None
    chunks = re.findall(r"[A-Z]+|\d+", value.upper())
    key = "".join(c.lstrip("0") or "0" if c.isdigit() else c for c in chunks)
    return key or None


# ─── Per-field dispatch ──────────────────────────────────────────────────────

FIELDS = ("total_amount", "currency", "receipt_date", "document_number")

NORMALIZERS = {
    "total_amount": normalize_amount,
    "currency": normalize_currency,
    "receipt_date": normalize_date,
    "document_number": normalize_document_number,
}


def loose_key(field: str, value):
    """Key under which two values count as the same up to formatting."""
    if value is None:
        return None
    if field == "document_number":
        return document_number_key(value)
    return value
