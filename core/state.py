from typing import TypedDict, Optional, Literal, Annotated
import operator
from pydantic import BaseModel, Field

CategoryType = Literal[
    "Income", "Rent", "Utilities", "Salaries", "Food",
    "Transport", "Subscriptions", "Taxes", "Transfer", "Fee", "Other"
]


class BusinessTripExpense(BaseModel):
    expense_description: str
    invoice_number_date: str
    receipt_date: str = ""
    expense_amount: float
    currency: str

class ExtractionResult(BaseModel):
    expenses: list[BusinessTripExpense]


class DocumentInput(TypedDict):
    """Input for a single parallel document processing branch.
    Each Send() creates one of these."""
    doc_b64: str
    doc_index: int
    total_docs: int
    company_cif: Optional[str]
    company_name: Optional[str]


class FinancialState(TypedDict):
    documents: list[str]

    extracted_texts: Annotated[list[str], operator.add]
    extracted_expenses: Annotated[list, operator.add]
    current_doc_index: int
    report: str

    company_cif: Optional[str]
    company_name: Optional[str]