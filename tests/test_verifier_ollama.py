"""OllamaBackend: unloading waits until the model has really left the GPU."""
import asyncio
from types import SimpleNamespace

from graph.verifier_ollama import OllamaBackend


class FakeClient:
    def __init__(self, listed_for: int):
        self.listed_for = listed_for  # /api/ps calls that still list the model after unload
        self.calls: list[str] = []

    async def generate(self, model, prompt, keep_alive, **kw):
        self.calls.append(f"generate {model} keep_alive={keep_alive}")

    async def ps(self):
        self.calls.append("ps")
        self.listed_for -= 1
        models = [SimpleNamespace(model="qwen3-vl:8b", name="qwen3-vl:8b")] if self.listed_for >= 0 else []
        return SimpleNamespace(models=models)


def test_unload_waits_until_model_leaves_ps():
    backend = OllamaBackend(vram_settle=0)
    backend.client = FakeClient(listed_for=2)
    backend.loaded = ["qwen3-vl:8b"]
    asyncio.run(backend.unload("qwen3-vl:8b"))
    assert backend.client.calls == ["generate qwen3-vl:8b keep_alive=0", "ps", "ps", "ps"]
    assert backend.loaded == []


def test_unload_gives_up_after_timeout():
    backend = OllamaBackend(unload_wait=0.3, vram_settle=0)
    backend.client = FakeClient(listed_for=1000)
    asyncio.run(backend.unload("qwen3-vl:8b"))  # returns (with a warning) instead of hanging
