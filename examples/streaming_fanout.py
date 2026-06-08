"""Streaming + fan-out demo (no real network calls).

A STREAM source yields three URL-like strings; a fan-out SINGLE
downstream "fetches" each one with a simulated delay. Each
:py:class:`JobEvent` is printed as it arrives so you can watch the
interleaving: the downstream begins processing item 0 before the
source finishes yielding item 2, and at most two ``fetch`` invocations
are in flight at once (``max_concurrency=2``).

Run:

    uv run python examples/streaming_fanout.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from streamwright import Pipeline

pipeline = Pipeline("streaming-fanout-demo")


@pipeline.step()
async def urls(ctx: Any, value: Any) -> AsyncIterator[str]:
    """Stream three URLs with a small delay between each."""
    for i in range(3):
        url = f"https://example.com/item-{i}"
        yield url
        await asyncio.sleep(0.05)


@pipeline.step(fan_out_from="urls", max_concurrency=2)
async def fetch(ctx: Any, url: str) -> dict[str, Any]:
    """Pretend to fetch each URL — sleeps to simulate I/O."""
    await asyncio.sleep(0.1)
    return {"url": url, "status": 200}


async def main() -> None:
    async for event in pipeline.run("start"):
        print(f"  {type(event).__name__}: {event}")


if __name__ == "__main__":
    asyncio.run(main())
