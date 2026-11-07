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


class ChatClient:
    def __init__(self, content="{}", done_reason="stop"):
        self.content, self.done_reason, self.kwargs = content, done_reason, None

    async def chat(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content, thinking="hmm"),
            prompt_eval_count=10, eval_count=5, done_reason=self.done_reason)


def test_thinking_models_sample_instead_of_greedy_decoding():
    backend = OllamaBackend()
    backend.client = ChatClient()
    backend._thinking = {"qwen3-vl:8b": True, "qwen2.5vl:7b": False}
    asyncio.run(backend.chat("qwen3-vl:8b", "p", "img", {}))
    opts = backend.client.kwargs["options"]
    assert backend.client.kwargs["think"] is True
    assert opts["temperature"] > 0 and opts["top_k"] == 20 and opts["top_p"] == 0.95
    asyncio.run(backend.chat("qwen2.5vl:7b", "p", "img", {}))
    assert backend.client.kwargs["options"]["temperature"] == 0
    assert "think" not in backend.client.kwargs


def test_running_out_of_tokens_is_reported_clearly():
    import pytest
    from graph.verifier_ollama import VerifierError
    backend = OllamaBackend()
    backend.client = ChatClient(content="", done_reason="length")
    backend._thinking = {"qwen3-vl:8b": True}
    with pytest.raises(VerifierError, match="tokens .*while thinking"):
        asyncio.run(backend.chat("qwen3-vl:8b", "p", "img", {}))
