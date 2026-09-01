import re
import threading
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from core.state import FinancialState, DocumentInput, ExtractionResult
from prompts.index import FINANCIAL_EXTRACTION_PROMPT
from graph.ocr_worker import run_ocr_single

llm_lock = threading.Lock()
# llm = ChatOllama(model="llama3.1:8b", temperature=0)

llm = ChatOllama(
    model="qwen2.5:1.5b", #
    base_url="http://192.168.100.56:11434",
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
        with llm_lock:
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