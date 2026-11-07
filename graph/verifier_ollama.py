"""Ollama access for the receipt verifier: model loading/swapping, the 100%-on-GPU
check and the chat call. The reading logic itself lives in graph/verifier.py."""
import asyncio
import logging
import time
from typing import Optional

import httpx
from ollama import AsyncClient, ResponseError

from core.config import (
    OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL, OLLAMA_NUM_CTX,
    VERIFY_GPU_MODE, VERIFY_KEEP_ALIVE, VERIFY_NUM_CTX, VERIFY_NUM_PREDICT,
    VERIFY_THINK_TEMPERATURE, VERIFY_THINK_TOP_K, VERIFY_THINK_TOP_P, VERIFY_TIMEOUT,
)

logger = logging.getLogger(__name__)


class VerifierError(Exception):
    """The verification could not run (Ollama down, model missing, timeout...).
    The endpoint answers status "error" and the caller retries later."""


class NotOnGpu(VerifierError):
    pass


_TRANSPORT_ERRORS = (httpx.HTTPError, ConnectionError, OSError)


def _same_model(name: Optional[str], model: str) -> bool:
    if not name:
        return False
    strip = lambda n: n[: -len(":latest")] if n.endswith(":latest") else n  # noqa: E731
    return strip(name) == strip(model)


class OllamaBackend:
    """One instance per verification. Tracks which verifier models it loaded so `end()`
    can unload them and bring the live-scan model back (swap mode)."""

    def __init__(
        self,
        host: str = OLLAMA_BASE_URL,
        gpu_mode: str = VERIFY_GPU_MODE,
        scanner_model: str = OLLAMA_MODEL,
        num_ctx: int = VERIFY_NUM_CTX,
        num_predict: int = VERIFY_NUM_PREDICT,
        keep_alive=VERIFY_KEEP_ALIVE,
        timeout: float = VERIFY_TIMEOUT,
        unload_wait: float = 30.0,
        vram_settle: float = 1.0,
    ) -> None:
        self.client = AsyncClient(host=host, timeout=timeout)
        self.unload_wait = unload_wait    # max seconds to wait for a model to leave /api/ps
        self.vram_settle = vram_settle    # extra pause for the runner to free its VRAM
        self.gpu_mode = gpu_mode
        self.scanner_model = scanner_model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.loaded: list[str] = []
        self.gpu_fraction: dict[str, float] = {}  # model -> share of its memory on the GPU
        self._thinking: dict[str, bool] = {}
        self._scanner_unloaded = False

    async def begin(self) -> None:
        """Free the VRAM held by the live-scan model (swap mode only)."""
        if self.gpu_mode == "swap":
            await self._unload(self.scanner_model)
            self._scanner_unloaded = True

    async def load(self, model: str) -> None:
        """Load `model` with the verifier context size and check it is 100% on the GPU."""
        try:
            await self.client.generate(
                model=model, prompt="", keep_alive=self.keep_alive, options={"num_ctx": self.num_ctx}
            )
        except ResponseError as e:
            if e.status_code == 404:
                raise VerifierError(f"Verifier model '{model}' is not available in Ollama (not pulled?).") from e
            raise VerifierError(f"Ollama could not load '{model}': {e.error}") from e
        except _TRANSPORT_ERRORS as e:
            raise VerifierError(f"Ollama unreachable while loading '{model}': {e}") from e
        self.loaded.append(model)

        fraction = await self._gpu_share(model)
        if fraction is not None:
            self.gpu_fraction[model] = fraction
            logger.info("Verifier model %s loaded, %.0f%% on GPU", model, fraction * 100)
            if fraction < 0.999:
                await self.unload(model)
                raise NotOnGpu(
                    f"Verifier model '{model}' is only {fraction:.0%} on GPU (part of it would run on "
                    "the CPU). Use a smaller quantization or lower VERIFY_NUM_CTX / VERIFY_IMAGE_MAX_SIDE."
                )
        self._thinking[model] = await self._supports_thinking(model)

    async def unload(self, model: str) -> None:
        await self._unload(model)
        if model in self.loaded:
            self.loaded.remove(model)

    async def end(self) -> None:
        """Unload the verifier models and reload the live-scan model (swap mode).
        Never raises: this runs in `finally` blocks."""
        if self.gpu_mode != "swap":
            return
        for model in list(self.loaded):
            await self.unload(model)
        if self._scanner_unloaded:
            try:
                await self.client.generate(
                    model=self.scanner_model, prompt="", keep_alive=OLLAMA_KEEP_ALIVE,
                    options={"num_ctx": OLLAMA_NUM_CTX},
                )
                self._scanner_unloaded = False
            except Exception:
                logger.exception("Could not reload the live-scan model %s", self.scanner_model)

    async def chat(self, model: str, prompt: str, image_b64: str, schema: dict) -> str:
        """One structured call with one image. Returns the raw message content."""
        thinking = self._thinking.get(model, False)
        options = {"num_ctx": self.num_ctx, "num_predict": self.num_predict}
        if thinking:
            options.update(temperature=VERIFY_THINK_TEMPERATURE, top_p=VERIFY_THINK_TOP_P, top_k=VERIFY_THINK_TOP_K)
        else:
            options["temperature"] = 0
        try:
            resp = await self.client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
                format=schema,
                options=options,
                keep_alive=self.keep_alive,
                **({"think": True} if thinking else {}),
            )
        except ResponseError as e:
            raise VerifierError(f"Ollama error from '{model}': {e.error}") from e
        except _TRANSPORT_ERRORS as e:
            raise VerifierError(f"Ollama call to '{model}' failed: {type(e).__name__}") from e
        thought = len(resp.message.thinking or "") if thinking else 0
        logger.info("Verifier %s: prompt_tokens=%s output_tokens=%s thinking_chars=%s done_reason=%s",
                    model, resp.prompt_eval_count, resp.eval_count, thought, resp.done_reason)
        content = resp.message.content or ""
        if not content.strip() and resp.done_reason == "length":
            raise VerifierError(f"'{model}' used all {self.num_predict} tokens (VERIFY_NUM_PREDICT) "
                                "before answering" + (" while thinking" if thinking else ""))
        return content

    async def _unload(self, model: str) -> None:
        """Unload `model` and wait until Ollama no longer lists it. Ollama answers the
        keep_alive=0 request before the runner has released its VRAM; loading the next
        model straight away would then see too little free memory and spill to the CPU."""
        try:
            await self.client.generate(model=model, prompt="", keep_alive=0)
        except Exception as e:  # not loaded / not pulled / Ollama down: nothing to free
            logger.debug("Unload of %s skipped: %s", model, e)
            return
        deadline = time.monotonic() + self.unload_wait
        while time.monotonic() < deadline:
            try:
                ps = await self.client.ps()
            except Exception:
                return
            if not any(_same_model(m.model, model) or _same_model(m.name, model) for m in ps.models):
                await asyncio.sleep(self.vram_settle)  # the runner process exits right after
                return
            await asyncio.sleep(0.25)
        logger.warning("Model %s still listed by Ollama %.0fs after unloading", model, self.unload_wait)

    async def _gpu_share(self, model: str) -> Optional[float]:
        try:
            ps = await self.client.ps()
        except Exception:
            logger.warning("Could not read Ollama /api/ps to check GPU placement of %s", model)
            return None
        for m in ps.models:
            if _same_model(m.model, model) or _same_model(m.name, model):
                if m.size:
                    return (m.size_vram or 0) / m.size
        logger.warning("Model %s not listed in Ollama /api/ps after loading", model)
        return None

    async def _supports_thinking(self, model: str) -> bool:
        try:
            info = await self.client.show(model)
            return "thinking" in (info.capabilities or [])
        except Exception:
            return False
