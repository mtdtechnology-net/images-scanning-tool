"""First pass (graph/nodes.py): why a document was rejected goes to the log, never to the client."""
import asyncio
import json
import logging
from types import SimpleNamespace

import graph.nodes as nodes
from core.state import LLMExtraction

EMPTY = {"expense_description": "", "invoice_number_date": "", "receipt_date": "", "expense_amount": 0,
         "currency": "", "expense_category": "", "vendor_city": "", "vendor_country": "",
         "number_of_nights": 0, "check_in_date": "", "check_out_date": ""}


class FakeLLM:
    def __init__(self, answer: dict):
        self.content = json.dumps(answer)

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self.content, response_metadata={})


def test_reject_reason_is_in_the_schema_right_after_valid():
    props = list(LLMExtraction.model_json_schema()["properties"])
    assert props[:2] == ["valid", "reject_reason"]


def test_rejection_logs_the_reason(monkeypatch, caplog):
    reason = "A translated conversion copy, not an invoice issued by the supplier."
    monkeypatch.setattr(nodes, "llm", FakeLLM({"valid": False, "reject_reason": reason, **EMPTY}))
    with caplog.at_level(logging.INFO, logger="graph.nodes"):
        result = asyncio.run(nodes.run_extraction_hybrid("text", "img", 0))
    assert result == nodes.INVALID_DOC_ERROR
    assert reason in caplog.text


def test_accepted_document_carries_no_reason(monkeypatch):
    answer = {**EMPTY, "valid": True, "reject_reason": "", "expense_description": "Hotel",
              "invoice_number_date": "Factura 1 / 03.11.2026", "expense_amount": 120.0, "currency": "RON",
              "expense_category": "accommodation"}
    monkeypatch.setattr(nodes, "llm", FakeLLM(answer))
    result = asyncio.run(nodes.run_extraction_hybrid("TOTAL 120,00", "img", 0))
    assert isinstance(result, list) and len(result) == 1
    assert not hasattr(result[0], "reject_reason")
