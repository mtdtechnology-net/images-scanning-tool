import re
import threading
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from core.state import FinancialState, DocumentInput, ExtractionResult
from prompts.index import FINANCIAL_EXTRACTION_PROMPT
from graph.ocr_worker import run_ocr_single

# (Lock-ul pentru LLM a fost eliminat pentru a permite paralelizarea cererilor către server)
# llm = ChatOllama(model="llama3.1:8b", temperature=0)

llm = ChatOllama(
    model="llama3.1:8b", #
    base_url="http://localhost:11434",
    temperature=0
)

_DATE_LINE_RE = re.compile(r"^\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}")

def merge_continuation_lines(lines_text: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines_text:
        if _DATE_LINE_RE.match(line) or not merged:
            merged.append(line)
        else:
            merged[-1] = f"{merged[-1]}\n{line}"
    return merged

def _run_extraction_on_text(text: str, doc_index: int) -> list:
    """Run LLM structured extraction on OCR text. Returns list of BusinessTripExpense objects."""
    prompt = FINANCIAL_EXTRACTION_PROMPT.format(text=text)
    message = HumanMessage(content=prompt)
    import json
    
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
        from core.state import BusinessTripExpense
        
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
                expenses.append(BusinessTripExpense(**t_data))
            except Exception as val_e:
                print(f"[LLM] Validation error for an expense: {val_e}")

        # Post-fix: extract receipt date deterministically from OCR text
        # Small models often miss it, so we use regex instead
        date_pattern = re.compile(r"\b(\d{2}[./]\d{2}[./]\d{4})\b")
        for exp in expenses:
            if not exp.receipt_date:
                all_dates = date_pattern.findall(text)
                if all_dates:
                    # Use the last date found (usually the receipt print date near the bottom)
                    exp.receipt_date = all_dates[-1]
                    print(f"  [Post-fix] Extracted receipt_date: {exp.receipt_date}")

        if expenses:
            print(f"[LLM] Extracted {len(expenses)} expenses from document {doc_index + 1}")
            return expenses
        else:
            print(f"[LLM] No valid expenses found in document {doc_index + 1}")
            return []

    except Exception as e:
        print(f"[LLM] Error extracting expenses: {e}")
        return []


_INVALID_DOC_ERROR = "__INVALID_DOCUMENT__"


def process_document(state: DocumentInput) -> dict:
    doc_b64 = state["doc_b64"]
    doc_index = state["doc_index"]
    total_docs = state["total_docs"]

    extracted_text = run_ocr_single(doc_b64, doc_index, total_docs)
    print(f"\n{'='*60}\n[OCR-FULL-TEXT] doc_index={doc_index+1}/{total_docs}\n{extracted_text}\n{'='*60}\n")

    result = _run_extraction_on_text(extracted_text, doc_index)

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

def generate_report(state: FinancialState) -> dict:
    """Format expenses into report dict."""
    report = "### Business Trip Expenses Extracted Successfully."
    return {"report": report}


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


def aggregate_web_expenses(state: FinancialState) -> dict:
    """Aggregate all extracted expenses into web form structure.

    Categorizes each expense into transport/accommodation/other,
    sums up the totals per category, deduces the trip location from
    vendor addresses, and builds the accommodation and transport tables.
    """
    from collections import Counter

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
        vendor_country = getattr(exp, "vendor_country", "")
        if vendor_city:
            cities.append(vendor_city)
        if vendor_country:
            countries.append(vendor_country)

        # Build accommodation entries from hotel invoices
        if category == "accommodation":
            accommodation_list.append({
                "accommodationName": exp.expense_description,
                "numberOfNights": getattr(exp, "number_of_nights", 1) or 1,
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
            "expense_amount": exp.expense_amount,
            "currency": exp.currency,
            "expense_category": category,
            "vendor_city": vendor_city,
            "vendor_country": vendor_country,
        })

    # Deduce the most likely trip location from all vendor addresses
    location = Counter(cities).most_common(1)[0][0] if cities else ""
    country = Counter(countries).most_common(1)[0][0] if countries else ""

    # Estimate per diem (50 RON per day)
    # Trip days is usually total nights + 1. If no nights, assume a 1-day trip.
    total_nights = sum(acc.get("numberOfNights", 1) for acc in accommodation_list)
    trip_days = total_nights + 1 if total_nights > 0 else 1
    per_diem_cost = trip_days * 50.0

    return {
        "web_result": {
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