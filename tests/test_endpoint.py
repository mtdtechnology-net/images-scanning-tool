"""POST /api/verify-receipt contract, with the Ollama backend replaced by canned answers."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from graph.verifier_ollama import VerifierError
from tests.test_verifier import PRIMARY, SECOND, FakeBackend, full_page


class EndpointBackend(FakeBackend):
    began = ended = 0

    def __init__(self, fail_load=False):
        super().__init__(full_page(), full_page(absolute=True))
        self.fail_load = fail_load

    async def begin(self):
        EndpointBackend.began += 1

    async def end(self):
        EndpointBackend.ended += 1

    async def load(self, model):
        if self.fail_load:
            raise VerifierError(f"Verifier model '{model}' is not available in Ollama (not pulled?).")
        await super().load(model)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1000, 2000), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "VERIFY_IDLE_GRACE_S", 0)
    # Canned answers are keyed on these model names, whatever the env configures.
    monkeypatch.setattr(main, "VERIFY_MODEL", PRIMARY)
    monkeypatch.setattr(main, "VERIFY_SECOND_MODEL", SECOND)
    return TestClient(main.app)  # no `with`: skips the OCR/Ollama warm-up


def test_verify_receipt_contract(client, monkeypatch):
    monkeypatch.setattr(main, "OllamaBackend", EndpointBackend)
    EndpointBackend.began = EndpointBackend.ended = 0
    resp = client.post("/api/verify-receipt", files={"file": ("r.png", _png(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert list(body) == ["status", "model", "models", "duration_ms", "pages", "fields", "notes"]
    assert body["status"] == "ok"
    assert body["model"] == PRIMARY
    assert body["pages"] == [{"page": 1, "width": 1000, "height": 2000}]
    assert set(body["fields"]) == {"total_amount", "currency", "receipt_date", "document_number"}
    for f in body["fields"].values():
        assert set(f) == {"value", "raw", "confidence", "page", "bbox"}
    assert body["fields"]["total_amount"]["value"] == 120.0
    assert (EndpointBackend.began, EndpointBackend.ended) == (1, 1)  # scanner model restored


def test_backend_failure_is_status_error_503(client, monkeypatch):
    monkeypatch.setattr(main, "OllamaBackend", lambda: EndpointBackend(fail_load=True))
    EndpointBackend.ended = 0
    resp = client.post("/api/verify-receipt", files={"file": ("r.png", _png(), "image/png")})
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "60"
    body = resp.json()
    assert body["status"] == "error" and "not pulled" in body["notes"]
    assert all(f["confidence"] == "none" for f in body["fields"].values())
    assert EndpointBackend.ended == 1


def test_unsupported_file_is_400(client):
    resp = client.post("/api/verify-receipt", files={"file": ("r.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
