import pytest

from graph.normalize import (
    document_number_key, normalize_amount, normalize_currency, normalize_date,
    normalize_document_number,
)


@pytest.mark.parametrize("text, expected", [
    ("120,00", 120.0),
    ("120.00", 120.0),
    ("120", 120.0),
    ("1.234,56", 1234.56),
    ("1,234.56", 1234.56),
    ("1 234,56", 1234.56),
    ("1 234,56", 1234.56),
    ("1'234.50", 1234.5),
    ("1.234.567,89", 1234567.89),
    ("1,234,567.89", 1234567.89),
    ("12.345.678", 12345678.0),
    ("1.234", 1234.0),      # one separator + 3 digits = thousands
    ("1234,5", 1234.5),
    ("0,50", 0.5),
    ("120,00 LEI", 120.0),
    ("TOTAL 120,00", 120.0),
    ("€ 45.90", 45.9),
    ("45,90€", 45.9),
    ("120,-", 120.0),
    ("-5,00", -5.0),
    ("", None),
    ("N/A", None),
    (None, None),
])
def test_normalize_amount(text, expected):
    assert normalize_amount(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("LEI", "RON"), ("lei", "RON"), ("Lei", "RON"), ("RON", "RON"), ("ron", "RON"), ("LEU", "RON"),
    ("120,00 LEI", "RON"), ("TOTAL LEI", "RON"),
    ("EUR", "EUR"), ("€", "EUR"), ("euro", "EUR"), ("45,90€", "EUR"),
    ("$", "USD"), ("USD", "USD"), ("£", "GBP"), ("Ft", "HUF"), ("zł", "PLN"), ("Kč", "CZK"),
    ("", None), ("xyz", None), (None, None),
])
def test_normalize_currency(text, expected):
    assert normalize_currency(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("03.11.2026", "2026-11-03"),
    ("3.11.2026", "2026-11-03"),
    ("03/11/2026", "2026-11-03"),
    ("03-11-2026", "2026-11-03"),
    ("2026-11-03", "2026-11-03"),
    ("2026.11.03", "2026-11-03"),
    ("03.11.26", "2026-11-03"),
    ("3 noiembrie 2026", "2026-11-03"),
    ("03 NOV 2026", "2026-11-03"),
    ("03-NOV-2026", "2026-11-03"),
    ("Data: 03.11.2026 14:32", "2026-11-03"),
    ("03.11.2026 14:32:05", "2026-11-03"),
    ("11/23/2026", "2026-11-23"),   # only valid as month/day
    ("31.02.2026", None),
    ("03.11.1899", None),
    ("", None),
    ("no date", None),
    (None, None),
])
def test_normalize_date(text, expected):
    assert normalize_date(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Nr. F-0042", "F-0042"),
    ("F-0042", "F-0042"),
    ("Factura nr. 123", "123"),
    ("Factură fiscală nr: 2026/77", "2026/77"),
    ("Seria ABC nr 0042", "ABC 0042"),
    ("#0042", "0042"),
    ("Bon fiscal 0042", "0042"),
    ("BONUS123", "BONUS123"),
    ("abc note 5", "ABC NOTE 5"),
    ("", None),
    ("Nr.", None),
    (None, None),
])
def test_normalize_document_number(text, expected):
    assert normalize_document_number(text) == expected


def test_document_number_key_ignores_formatting():
    assert document_number_key("F-0042") == document_number_key("F 42") == "F42"
    assert document_number_key("ABC 0042") == "ABC42"
    assert document_number_key("F-0042") != document_number_key("F-0043")
    assert document_number_key(None) is None
