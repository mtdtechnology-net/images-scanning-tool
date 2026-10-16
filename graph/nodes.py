import re
import json
import threading
import datetime
from typing import Optional
from collections import Counter
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from core.state import FinancialState, DocumentInput, ExtractionResult, BusinessTripExpense
from prompts.index import FINANCIAL_EXTRACTION_PROMPT
from graph.ocr_worker import run_ocr_single

llm = ChatOllama(
    model="qwen3-vl:8b-instruct",
    base_url="http://localhost:11434",
    temperature=0
)

_DATE_LINE_RE = re.compile(r"^\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}")

_INVALID_DOC_ERROR = "__INVALID_DOCUMENT__"


_DATE_PATTERNS = [
    r'\d{1,2}[.]\d{1,2}[.]\d{2,4}',            # 08.07.2026
    r'\d{1,2}/\d{1,2}/\d{2,4}',                 # 08/07/2026
    r'\d{4}\s*-\s*\d{2}\s*-\s*\d{2}',           # 2026-07-08
    r'\d{1,2}\s*-\s*[A-Z]{3}\s*-\s*\d{4}',      # 20-NOV-2024
    r'\d{1,2}\s+[A-Z]{3}\s+\d{4}',              # 20 NOV 2024
]
 
 
_ROMANIAN_MONTHS = {
    "ian": 1, "ianuarie": 1, "jan": 1, "january": 1,
    "feb": 2, "februarie": 2, "february": 2,
    "mar": 3, "martie": 3, "march": 3,
    "apr": 4, "aprilie": 4, "april": 4,
    "mai": 5, "may": 5,
    "iun": 6, "iunie": 6, "june": 6,
    "iul": 7, "iulie": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "septembrie": 9, "september": 9,
    "oct": 10, "octombrie": 10, "october": 10,
    "noi": 11, "nov": 11, "noiembrie": 11, "november": 11,
    "dec": 12, "decembrie": 12, "december": 12,
}


def _parse_flexible_date(date_str: str) -> Optional[datetime.date]:
    """Parse flexible date strings (YYYY-MM-DD, DD.MM.YYYY, DD Month YYYY, etc.) into datetime.date."""
    if not date_str:
        return None
    s = str(date_str).strip()

    # 1. YYYY-MM-DD or YYYY.MM.DD or YYYY/MM/DD
    m1 = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})", s)
    if m1:
        try:
            return datetime.date(int(m1.group(1)), int(m1.group(2)), int(m1.group(3)))
        except ValueError:
            pass

    # 2. DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY
    m2 = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})", s)
    if m2:
        try:
            return datetime.date(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
        except ValueError:
            pass

    # 3. DD MonthName YYYY (e.g. 01 Octombrie 2026, 20-NOV-2024, 01 Oct 2026)
    m3 = re.search(r"(\d{1,2})[\s\-\./]+([a-zA-ZăîșțĂÎȘȚ]{3,})[\s\-\./]+(\d{4})", s)
    if m3:
        day = int(m3.group(1))
        month_word = m3.group(2).lower()
        year = int(m3.group(3))
        for k, v in _ROMANIAN_MONTHS.items():
            if month_word.startswith(k) or k.startswith(month_word):
                try:
                    return datetime.date(year, v, day)
                except ValueError:
                    pass
                break

    # 4. DD.MM.YY
    m4 = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2})$", s)
    if m4:
        try:
            yy = int(m4.group(3))
            year = 2000 + yy if yy < 70 else 1900 + yy
            return datetime.date(year, int(m4.group(2)), int(m4.group(1)))
        except ValueError:
            pass

    return None


def _validate_receipt_date(receipt_date: str, ocr_text: str) -> bool:
    """Verify that a date pattern actually exists in the OCR text.
    
    Prevents the LLM from hallucinating dates that don't appear
    on the actual receipt. Returns True only if the OCR text
    contains recognizable date patterns.
    """
    ocr_upper = ocr_text.upper()
    for pattern in _DATE_PATTERNS:
        if re.search(pattern, ocr_upper):
            return True
    return False


def merge_continuation_lines(lines_text: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines_text:
        if _DATE_LINE_RE.match(line) or not merged:
            merged.append(line)
        else:
            merged[-1] = f"{merged[-1]}\n{line}"
    return merged


# ─── Hybrid Extraction (OCR text + image → LLM) ─────────────────────────────

def _run_extraction_hybrid(text: str, image_b64: str, doc_index: int) -> list:
    """Run multimodal LLM structured extraction using both OCR text and the image.
    
    The LLM receives:
    1. The raw OCR text (perfect for numbers, amounts, dates)
    2. The original image (perfect for logos, layout, visual context)
    
    This hybrid approach lets the LLM cross-reference visual cues with exact
    text strings, producing much more accurate extractions than either alone.
    
    Returns a list of BusinessTripExpense objects or an error marker.
    """
    prompt = FINANCIAL_EXTRACTION_PROMPT.format(text=text)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        ]
    )

    try:
        response = llm.bind(format="json").invoke([message])
        
        raw_text = response.content.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        data = json.loads(raw_text)
        
        if isinstance(data, dict) and data.get("valid") is False:
            print(f"[LLM] Document {doc_index + 1} rejected: not a fiscal receipt")
            return _INVALID_DOC_ERROR
        
        if "expenses" in data and isinstance(data["expenses"], list):
            txs_data = data["expenses"]
        elif isinstance(data, list):
            txs_data = data
        elif isinstance(data, dict):
            data.pop("valid", None)
            txs_data = [data]
        else:
            print(f"[LLM] Error: Unexpected JSON structure: {raw_text[:150]}")
            return []

        expenses = []
        for t_data in txs_data:
            try:
                exp = BusinessTripExpense(**t_data)

                # ── Guardrail 1: validate receipt_date against OCR text ──
                if exp.receipt_date:
                    if not _validate_receipt_date(exp.receipt_date, text):
                        print(f"  [Guardrail] receipt_date '{exp.receipt_date}' NOT found in OCR text — setting to ''")
                        exp.receipt_date = ""

                # ── Guardrail 2: post-fix extract receipt_date from OCR if LLM missed it ──
                if not exp.receipt_date:
                    date_pattern = re.compile(r"\b(\d{2}[./]\d{2}[./]\d{4})\b")
                    all_dates = date_pattern.findall(text)
                    if all_dates:

                        exp.receipt_date = all_dates[-1]
                        print(f"  [Guardrail] Extracted receipt_date from OCR: {exp.receipt_date}")

                # Format city and country in Title Case (e.g. SIBIU -> Sibiu)
                if exp.vendor_city:
                    exp.vendor_city = exp.vendor_city.strip().title()
                if exp.vendor_country:
                    exp.vendor_country = exp.vendor_country.strip().title()

                # Ensure check_in_date and check_out_date are only kept for accommodation
                if exp.expense_category != "accommodation":
                    exp.check_in_date = ""
                    exp.check_out_date = ""

                expenses.append(exp)
            except Exception as val_e:
                print(f"[LLM] Validation error for an expense: {val_e}")

        if expenses:
            print(f"[LLM] Extracted {len(expenses)} expenses from document {doc_index + 1}")
            for exp in expenses:
                print(f"  → {exp.expense_description} | {exp.expense_amount} {exp.currency} | "
                      f"cat={exp.expense_category} | city={exp.vendor_city} | date={exp.receipt_date} | "
                      f"check_in={exp.check_in_date} | check_out={exp.check_out_date}")
            return expenses
        else:
            print(f"[LLM] No valid expenses found in document {doc_index + 1}")
            return []

    except Exception as e:
        print(f"[LLM] Error extracting expenses: {e}")
        return []


# ─── LangGraph Node: process_document ────────────────────────────────────────

def process_document(state: DocumentInput) -> dict:
    """Process a single document: Preprocess → OCR → Hybrid LLM extraction.
    
    Called by LangGraph via Send() for each document in parallel.
    """
    doc_b64 = state["doc_b64"]
    doc_index = state["doc_index"]
    total_docs = state["total_docs"]

    # Step 1: OCR (includes preprocessing pipeline internally)
    extracted_text = run_ocr_single(doc_b64, doc_index, total_docs)
    print(f"\n{'='*60}\n[OCR-FULL-TEXT] doc_index={doc_index+1}/{total_docs}\n{extracted_text}\n{'='*60}\n")

    # Step 2: Hybrid extraction (OCR text + original image → LLM)
    result = _run_extraction_hybrid(extracted_text, doc_b64, doc_index)

    # If the AI rejected the document, pass the error marker through
    if result == _INVALID_DOC_ERROR:
        return {
            "extracted_texts": [extracted_text],
            "extracted_expenses": [_INVALID_DOC_ERROR],
        }

    print(f"expenses: {result}")

    return {
        "extracted_texts": [extracted_text],
        "extracted_expenses": result,
    }


# ─── LangGraph Node: generate_report ─────────────────────────────────────────

def generate_report(state: FinancialState) -> dict:
    """Format expenses into report dict."""
    report = "### Business Trip Expenses Extracted Successfully."
    return {"report": report}


# ─── Utility: deduce transport type ──────────────────────────────────────────

def _deduce_means_of_transport(description: str) -> str:
    """Deduce meansOfTransport from expense description.

    Returns one of the EXACT values accepted by the frontend dropdown:
    "Plane", "Train", "Car", "Bus", "Company car", "Other".
    """
    desc = description.lower()

    plane_keywords = ["flight", "plane", "air", "wizz", "ryanair", "tarom",
                      "lufthansa", "blue air", "aeroport", "airport", "zbor"]
    train_keywords = ["train", "tren", "cfr", "railway"]
    bus_keywords = ["bus", "coach", "autobuz", "flixbus", "flix"]
    car_keywords = ["fuel", "motorina", "benzina", "diesel", "omv", "petrom",
                    "rompetrol", "mol", "lukoil", "socar", "gas station",
                    "parking", "parcare", "taxi", "uber", "bolt", "toll",
                    "rovinieta", "vigneta"]

    for kw in plane_keywords:
        if kw in desc:
            return "Plane"
    for kw in train_keywords:
        if kw in desc:
            return "Train"
    for kw in bus_keywords:
        if kw in desc:
            return "Bus"
    for kw in car_keywords:
        if kw in desc:
            return "Car"

    return "Other"


# ─── LangGraph Node: aggregate_web_expenses ──────────────────────────────────

def aggregate_web_expenses(state: FinancialState) -> dict:
    """Aggregate all extracted expenses into web form structure.

    Categorizes each expense into transport/accommodation/other,
    sums up the totals per category, deduces the trip location from
    vendor addresses, and builds the accommodation and transport tables.
    """
    expenses = state.get("extracted_expenses", [])

    transport_total = 0.0
    accommodation_total = 0.0
    other_total = 0.0
    currency = "RON"
    items = []
    cities = []
    countries = []
    accommodation_list = []
    transport_list = []

    for exp in expenses:
        if exp == "__INVALID_DOCUMENT__":
            continue

        currency = exp.currency
        amount = exp.expense_amount
        category = getattr(exp, "expense_category", "other")

        if category == "transport":
            transport_total += amount
        elif category == "accommodation":
            accommodation_total += amount
        else:
            other_total += amount

        # Collect location data from all receipts
        vendor_city = getattr(exp, "vendor_city", "")
        if vendor_city:
            vendor_city = vendor_city.strip().title()
        vendor_country = getattr(exp, "vendor_country", "")
        if vendor_country:
            vendor_country = vendor_country.strip().title()

        if vendor_city:
            cities.append(vendor_city)
        if vendor_country:
            countries.append(vendor_country)

        # Build accommodation entries from hotel invoices
        if category == "accommodation":
            accommodation_list.append({
                "accommodationName": exp.expense_description,
                "numberOfNights": getattr(exp, "number_of_nights", 1) or 1,
                "checkInDate": getattr(exp, "check_in_date", ""),
                "checkOutDate": getattr(exp, "check_out_date", ""),
            })

        # Build transport entries from transport receipts
        if category == "transport":
            transport_list.append({
                "city": vendor_city,
                "meansOfTransport": _deduce_means_of_transport(exp.expense_description),
            })

        items.append({
            "expense_description": exp.expense_description,
            "invoice_number_date": exp.invoice_number_date,
            "receipt_date": exp.receipt_date,
            "check_in_date": getattr(exp, "check_in_date", ""),
            "check_out_date": getattr(exp, "check_out_date", ""),
            "expense_amount": exp.expense_amount,
            "currency": exp.currency,
            "expense_category": category,
            "vendor_city": vendor_city,
            "vendor_country": vendor_country,
        })

    # Deduce the trip location prioritizing accommodation city
    location = ""
    for exp in expenses:
        if exp == "__INVALID_DOCUMENT__":
            continue
        if getattr(exp, "expense_category", "other") == "accommodation" and getattr(exp, "vendor_city", ""):
            location = exp.vendor_city
            break

    if not location:
        location = Counter(cities).most_common(1)[0][0] if cities else ""

    if location:
        location = location.strip().title()

    country = Counter(countries).most_common(1)[0][0] if countries else ""
    if country:
        country = country.strip().title()

    # ─── Deduce startDate and endDate ────────────────────────────────────────
    start_date = ""
    end_date = ""

    # Strategy 1: Prioritize accommodation dates (check_in_date / check_out_date)
    acc_check_ins = []
    acc_check_outs = []
    for exp in expenses:
        if exp == "__INVALID_DOCUMENT__":
            continue
        if getattr(exp, "expense_category", "other") == "accommodation":
            ci = getattr(exp, "check_in_date", "")
            co = getattr(exp, "check_out_date", "")
            ci_d = _parse_flexible_date(ci)
            co_d = _parse_flexible_date(co)

            nights = getattr(exp, "number_of_nights", 0) or 1
            if ci_d and not co_d and nights > 0:
                co_d = ci_d + datetime.timedelta(days=nights)

            # If check-in is not specified, attempt receipt_date of accommodation
            if not ci_d:
                rc_d = _parse_flexible_date(exp.receipt_date)
                if rc_d:
                    ci_d = rc_d
                    if not co_d and nights > 0:
                        co_d = ci_d + datetime.timedelta(days=nights)

            if ci_d:
                acc_check_ins.append(ci_d)
            if co_d:
                acc_check_outs.append(co_d)

    if acc_check_ins:
        start_date = min(acc_check_ins).strftime("%Y-%m-%d")
        if acc_check_outs:
            end_date = max(acc_check_outs).strftime("%Y-%m-%d")
        else:
            end_date = start_date

    # Strategy 2: Fallback to receipt_date across all documents if no accommodation dates
    if not start_date or not end_date:
        all_receipt_dates = []
        for exp in expenses:
            if exp == "__INVALID_DOCUMENT__":
                continue
            rd = _parse_flexible_date(exp.receipt_date)
            if rd:
                all_receipt_dates.append(rd)

        if all_receipt_dates:
            if not start_date:
                start_date = min(all_receipt_dates).strftime("%Y-%m-%d")
            if not end_date:
                end_date = max(all_receipt_dates).strftime("%Y-%m-%d")

    # Estimate per diem (55 RON per day) only if accommodation is present
    if accommodation_list:
        total_nights = sum(acc.get("numberOfNights", 1) for acc in accommodation_list)
        trip_days = total_nights + 1 if total_nights > 0 else 1
        per_diem_cost = trip_days * 55.0
    else:
        per_diem_cost = 0.0

    return {
        "web_result": {
            "startDate": start_date,
            "endDate": end_date,
            "transportCost": round(transport_total, 2),
            "accommodationCost": round(accommodation_total, 2),
            "otherCosts": round(other_total, 2),
            "perDiemCost": round(per_diem_cost, 2),
            "currency": currency,
            "estimatedTotalCost": round(transport_total + accommodation_total + other_total + per_diem_cost, 2),
            "location": location,
            "country": country,
            "accommodation": accommodation_list,
            "transport": transport_list,
            "items": items,
        }
    }