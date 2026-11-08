"""First-pass date guardrails (graph/nodes.py): dates with dashes must survive."""
import pytest

from graph.nodes import _last_ocr_date, _parse_flexible_date, _validate_receipt_date

ROMPETROL = """Z:0099 BF:0658 | NR.AMEF:0001
DATA: 13-02-2026 ORA: 18:54:34"""


@pytest.mark.parametrize("ocr", [
    ROMPETROL,
    "NUMAR BON: 0109-00214 | 27.01.2026 | 17:37:16",
    "DATA: 14/06/2026 | ORA: 10:42:19",
    "2026-01-27 17:37:02",
])
def test_date_in_ocr_text_is_recognised(ocr):
    assert _validate_receipt_date("any", ocr)


def test_no_date_in_ocr_text():
    assert not _validate_receipt_date("13-02-2026", "TOTAL LEI 219.10 | CARD 219.10")


def test_last_ocr_date_with_dashes():
    assert _last_ocr_date(ROMPETROL) == "13-02-2026"
    assert _last_ocr_date("A 01.02.2026 B 03/04/2026") == "03/04/2026"
    assert _last_ocr_date("NUMAR BON: 0109-00214") == ""  # not a date


def test_dash_date_parses_for_trip_dates():
    assert _parse_flexible_date("13-02-2026").isoformat() == "2026-02-13"
