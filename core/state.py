from pydantic import BaseModel


class BusinessTripExpense(BaseModel):
    expense_description: str
    invoice_number_date: str
    receipt_date: str = ""
    expense_amount: float
    currency: str
    expense_category: str = "other"  # "transport" | "accommodation" | "other"
    vendor_city: str = ""            # city extracted from vendor address
    vendor_country: str = ""         # country extracted from vendor address
    number_of_nights: int = 0        # only for accommodation invoices
    check_in_date: str = ""          # check-in / arrival date (for accommodation)
    check_out_date: str = ""         # check-out / departure date (for accommodation)


class LLMExtraction(BaseModel):
    """Exact shape the LLM must return. Its JSON schema is passed to Ollama as
    `format`, so decoding is constrained to valid JSON with every field present.
    `valid` comes first so the model decides it before filling the fields."""
    valid: bool
    expense_description: str
    invoice_number_date: str
    receipt_date: str
    expense_amount: float
    currency: str
    expense_category: str
    vendor_city: str
    vendor_country: str
    number_of_nights: int
    check_in_date: str
    check_out_date: str
