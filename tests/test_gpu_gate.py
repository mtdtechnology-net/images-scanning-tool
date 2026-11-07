"""Live scans keep priority: verifications wait for idle and are preempted by new scans."""
import asyncio

import pytest

from core.gpu_gate import GpuBusy, LiveScanTracker, Preempted


def test_wait_idle_returns_at_once_when_nothing_ran():
    async def main():
        await asyncio.wait_for(LiveScanTracker().wait_idle(grace=5, timeout=1), 0.5)
    asyncio.run(main())


def test_wait_idle_times_out_while_a_scan_runs():
    async def main():
        t = LiveScanTracker()
        async with t.scan():
            with pytest.raises(GpuBusy):
                await t.wait_idle(grace=0, timeout=0.05)
    asyncio.run(main())


def test_wait_idle_waits_for_scan_end_plus_grace():
    async def main():
        t = LiveScanTracker()
        loop = asyncio.get_running_loop()

        async def scan():
            async with t.scan():
                await asyncio.sleep(0.05)

        task = asyncio.create_task(scan())
        await asyncio.sleep(0)
        start = loop.time()
        await t.wait_idle(grace=0.05, timeout=1)
        assert loop.time() - start >= 0.09
        await task
    asyncio.run(main())


def test_run_preemptible_returns_result_when_undisturbed():
    async def main():
        t = LiveScanTracker()

        async def work():
            await asyncio.sleep(0.01)
            return 42
        assert await t.run_preemptible(work()) == 42
    asyncio.run(main())


def test_new_scan_preempts_and_cancels_verification():
    async def main():
        t = LiveScanTracker()
        cancelled = asyncio.Event()

        async def work():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def scan_soon():
            await asyncio.sleep(0.02)
            async with t.scan():
                await asyncio.sleep(0.05)

        scan_task = asyncio.create_task(scan_soon())
        with pytest.raises(Preempted):
            await t.run_preemptible(work())
        assert cancelled.is_set()
        await scan_task
    asyncio.run(main())


def test_run_preemptible_refuses_to_start_during_a_scan():
    async def main():
        t = LiveScanTracker()

        async def work():
            return 1
        async with t.scan():
            with pytest.raises(Preempted):
                await t.run_preemptible(work())
    asyncio.run(main())
