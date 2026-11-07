"""Prompts and JSON schemas for the receipt verifier (POST /api/verify-receipt).

The verifier only sees the page image: never OCR text, never what the employee declared.
"""

_FIELD = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "bbox_2d": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["text", "bbox_2d"],
}

# Read A (primary model) and read D (second model): the whole page, all four fields.
FULL_PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "enum": ["receipt", "invoice", "other", "unreadable"]},
        "receipt_count": {"type": "integer"},
        "total_amount": _FIELD,
        "currency": _FIELD,
        "receipt_date": _FIELD,
        "document_number": _FIELD,
    },
    "required": ["document_type", "receipt_count", "total_amount", "currency",
                 "receipt_date", "document_number"],
}

FULL_PAGE_PROMPT = """You are checking a scanned receipt or invoice. Read ONLY what is printed on the image.
Do not guess, do not calculate, do not fill in values that are not printed.

Return JSON with:
- document_type: "receipt" (fiscal receipt, bon fiscal), "invoice" (factură), "other" (not a receipt
  or invoice: order confirmation, proforma, booking confirmation, menu, random photo), or
  "unreadable" (too blurry, dark or cut off to read).
- receipt_count: how many separate receipts/invoices are visible on this image.
- total_amount: the FINAL amount paid, VAT included (TOTAL, TOTAL DE PLATĂ, TOTAL LEI, Total general,
  Amount due). NOT the subtotal, NOT the VAT/TVA line, NOT the cash handed over (NUMERAR/CASH when
  change is given), NOT the change (REST).
- currency: the currency as printed (e.g. LEI, RON, EUR, €). Many receipts print no currency:
  then return "". Never deduce it from the country, the language or the shop.
- receipt_date: the date the document was issued, as printed.
- document_number: the receipt/invoice number as printed (e.g. "Nr. bon", "Factura nr.", "Seria ... nr ...").

For each of the four fields return:
- "text": the exact characters printed, copied as they are (keep separators such as "1.234,56");
- "bbox_2d": [x1, y1, x2, y2], the box around that printed value on the image.
If a field is not printed on the document, return "text": "" and "bbox_2d": []."""

# Read B (primary model): every amount in the summary area, labelled. Code picks the total.
AMOUNT_KINDS = ["subtotal", "discount", "vat", "total", "payment", "change", "other"]

AMOUNT_LINES_SCHEMA = {
    "type": "object",
    "properties": {
        "amount_lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "amount_text": {"type": "string"},
                    "kind": {"type": "string", "enum": AMOUNT_KINDS},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["label", "amount_text", "kind", "bbox_2d"],
            },
        },
        "currency_text": {"type": "string"},
        "date_text": {"type": "string"},
        "document_number_text": {"type": "string"},
    },
    "required": ["amount_lines", "currency_text", "date_text", "document_number_text"],
}

AMOUNT_LINES_PROMPT = """This image is a receipt or invoice. Copy what is printed; do not calculate anything.

1. List every line of the summary section (after the list of products) that shows a money amount:
   subtotals, discounts, VAT/TVA lines, totals, payment lines (NUMERAR, CASH, CARD), change (REST).
   For each line return:
   - "label": the label as printed (e.g. "TOTAL", "SUBTOTAL", "TVA A 19%", "NUMERAR", "REST");
   - "amount_text": the amount exactly as printed;
   - "kind": one of "subtotal", "discount", "vat", "total" (final amount to pay, VAT included),
     "payment", "change", "other";
   - "bbox_2d": [x1, y1, x2, y2] around the amount.
2. "currency_text": the currency as printed; "" if no currency is printed (never deduce it).
3. "date_text": the issue date as printed ("" if none).
4. "document_number_text": the receipt/invoice number as printed ("" if none)."""

# Read C (primary model): a zoomed crop around one field found by read A.
ZOOM_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

ZOOM_TARGETS = {
    "total_amount": "the FINAL total amount paid (the TOTAL line, VAT included; not the subtotal, "
                    "the VAT line, the cash handed over or the change)",
    "currency": "the currency printed next to the total (e.g. LEI, RON, EUR, €); "
                "if no currency is printed, return an empty string",
    "receipt_date": "the date the document was issued",
    "document_number": "the receipt/invoice number",
}

ZOOM_PROMPT = """This image is a zoomed-in strip of a receipt or invoice.
Copy exactly the characters of {target}, as printed (keep separators such as "1.234,56").
Do not guess: if it is not fully visible or not readable in this strip, return "text": ""."""
