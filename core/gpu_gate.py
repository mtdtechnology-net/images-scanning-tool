"""Live scans have priority on the GPU over background verifications.

- Live scan endpoints wrap their stream in `live_scans.scan()`.
- A verification waits until no scan has run for a grace period (`wait_idle`), then runs
  through `run_preemptible`, which cancels it as soon as a new scan starts. The caller
  answers status "error" and the Spendle queue retries later.

Single process only (the service runs one uvicorn worker): the state lives in memory.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Optional, TypeVar

from core.config import VERIFY_CONCURRENCY

T = TypeVar("T")


class GpuBusy(Exception):
    """Live scans kept the GPU busy for longer than the verifier is willing to wait."""


class Preempted(Exception):
    """A live scan started while the verification was running."""


class LiveScanTracker:
    def __init__(self) -> None:
        self._active = 0
        self._last_end: Optional[float] = None
        self._idle = asyncio.Event()
        self._idle.set()
        self._busy = asyncio.Event()

    @property
    def active(self) -> int:
        return self._active

    @asynccontextmanager
    async def scan(self):
        """Mark a live scan as running for the duration of the block."""
        self._active += 1
        self._idle.clear()
        self._busy.set()
        try:
            yield
        finally:
            self._active -= 1
            self._last_end = time.monotonic()
            if self._active == 0:
                self._busy.clear()
                self._idle.set()

    async def wait_idle(self, grace: float, timeout: float) -> None:
        """Return once no scan is running and none ended in the last `grace` seconds.
        Raise GpuBusy if that doesn't happen within `timeout` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            remaining = deadline - now
            if self._active == 0:
                quiet = float("inf") if self._last_end is None else now - self._last_end
                if quiet >= grace:
                    return
                if remaining <= 0:
                    raise GpuBusy()
                await asyncio.sleep(min(grace - quiet, remaining))
            else:
                if remaining <= 0:
                    raise GpuBusy()
                try:
                    await asyncio.wait_for(self._idle.wait(), remaining)
                except asyncio.TimeoutError:
                    raise GpuBusy() from None

    async def run_preemptible(self, aw: Awaitable[T]) -> T:
        """Run `aw`, cancelling it and raising Preempted as soon as a live scan starts."""
        task = asyncio.ensure_future(aw)
        if self._active:
            task.cancel()
            raise Preempted()
        watcher = asyncio.ensure_future(self._busy.wait())
        try:
            done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return task.result()
            task.cancel()
            try:
                await task
            except BaseException:  # the cancellation itself, or whatever the task died of
                pass
            raise Preempted()
        finally:
            watcher.cancel()
            if not task.done():
                task.cancel()


live_scans = LiveScanTracker()
verify_semaphore = asyncio.Semaphore(VERIFY_CONCURRENCY)
