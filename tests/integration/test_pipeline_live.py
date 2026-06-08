"""End-to-end live pipeline integration test.

Builds the documented streaming topology against three real providers:

    scenes (STREAM, Anthropic Haiku via stream_json_array)
       ├── visual (SINGLE fan-out, Moonshot Kimi K2.5)
       └── narration (SINGLE fan-out, OpenAI gpt-5.2)
              │
              └── merged (MERGE on scene index)

Asserts both that the pipeline completes successfully and the
streaming-first ordering claim: the merged output for scene 0 lands
*before* the upstream ``scenes`` step finishes. To give that assertion
real headroom, the prompt asks Anthropic for longer briefs per scene
(so the upstream stream takes noticeably longer than each downstream
single completion).

Skips unless all three API keys are present.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import streamwright
from streamwright import Message, Pipeline
from streamwright.types import (
    JobEvent,
    PipelineDone,
    StepDone,
    StepOutput,
)

SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "title": {"type": "string"},
        "brief": {"type": "string"},
    },
    "required": ["index", "title", "brief"],
}

_SOURCE_DOC = (
    "A sunrise over a quiet harbor town. Fishermen prepare nets while "
    "seagulls circle overhead and the first light catches the water."
)
_SCENES_SYSTEM = (
    "You are a scene planner. Always emit exactly 2 scene objects, each "
    "with index (0 or 1), title (short), and brief (3 to 5 sentences "
    "describing the visual)."
)
_MAX_TOKENS = 80


async def _collect_with_timestamps(
    events: AsyncIterator[JobEvent],
) -> list[tuple[float, JobEvent]]:
    """Drain the event stream, recording perf_counter() per event."""
    collected: list[tuple[float, JobEvent]] = []
    async for event in events:
        collected.append((time.perf_counter(), event))
    return collected


def _build_pipeline() -> Pipeline:
    pipeline = Pipeline("scenes-to-narrated-visuals")

    @pipeline.step()
    async def scenes(ctx: Any, source_doc: str) -> AsyncIterator[dict[str, Any]]:
        """Stream exactly 2 scene objects from Anthropic Haiku."""
        provider, model = await ctx.llm("anthropic/claude-haiku-4-5")
        messages = [
            Message(role="system", content=_SCENES_SYSTEM),
            Message(
                role="user",
                content=(
                    f"Source doc:\n{source_doc}\n\n"
                    "Emit exactly 2 scene objects with index 0 and 1."
                ),
            ),
        ]
        yielded = 0
        async for obj in provider.stream_json_array(
            model=model,
            messages=messages,
            schema=SCENE_SCHEMA,
            max_tokens=_MAX_TOKENS,
        ):
            yield obj
            yielded += 1
            if yielded >= 2:
                break

    @pipeline.step(fan_out_from="scenes", max_concurrency=2)
    async def visual(ctx: Any, scene: dict[str, Any]) -> dict[str, Any]:
        """One-sentence SVG concept for a scene's brief, via Moonshot."""
        provider, model = await ctx.llm("moonshot/kimi-k2.5")
        result = await provider.complete(
            model=model,
            messages=[
                Message(
                    role="user",
                    content=(
                        "In one sentence, describe an SVG concept for this "
                        f"scene brief: {scene['brief']}"
                    ),
                )
            ],
            max_tokens=_MAX_TOKENS,
        )
        return {"index": int(scene["index"]), "visual": result.text}

    @pipeline.step(fan_out_from="scenes", max_concurrency=2)
    async def narration(ctx: Any, scene: dict[str, Any]) -> dict[str, Any]:
        """One-sentence narration for a scene title, via OpenAI."""
        provider, model = await ctx.llm("openai/gpt-5.2")
        result = await provider.complete(
            model=model,
            messages=[
                Message(
                    role="user",
                    content=(
                        "Write exactly one sentence of narration for a scene "
                        f"titled: {scene['title']}"
                    ),
                )
            ],
            max_tokens=_MAX_TOKENS,
        )
        return {"index": int(scene["index"]), "narration": result.text}

    @pipeline.merge("visual", "narration", key=lambda v: v["index"])
    async def merged(
        ctx: Any,
        key: int,
        visual_value: dict[str, Any],
        narration_value: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Join visual + narration for the same scene index."""
        yield {
            "index": key,
            "visual": visual_value["visual"],
            "narration": narration_value["narration"],
        }

    return pipeline


async def test_pipeline_end_to_end_streams_across_providers(
    anthropic_key: str,
    moonshot_key: str,
    openai_key: str,
) -> None:
    """Three-provider pipeline runs end-to-end and preserves streaming-first ordering."""
    del anthropic_key, moonshot_key, openai_key

    pipeline = _build_pipeline()

    started = time.perf_counter()
    try:
        timeline = await _collect_with_timestamps(pipeline.run(_SOURCE_DOC))
    finally:
        await streamwright.aclose()
    elapsed = time.perf_counter() - started

    # Loose wall-clock bound — generous to absorb provider variance.
    assert elapsed < 60.0, f"pipeline took {elapsed:.1f}s, exceeded 60s budget"

    # PipelineDone fired exactly once at the end.
    assert any(isinstance(ev, PipelineDone) for _, ev in timeline), (
        "no PipelineDone event observed"
    )

    # At least 2 merged outputs (one per scene).
    merged_outputs = [
        (t, ev)
        for t, ev in timeline
        if isinstance(ev, StepOutput) and ev.step == "merged"
    ]
    assert len(merged_outputs) >= 2, (
        f"expected ≥2 merged outputs, got {len(merged_outputs)}: "
        f"{[ev for _, ev in merged_outputs]}"
    )

    # Streaming-first claim: the first merged output for scene 0 arrives
    # before the scenes STREAM step signals StepDone. The Anthropic
    # stream takes longer to emit the full 2-scene array than the
    # downstream single completions take per scene, so merged(0) should
    # land before scenes finishes.
    scene_zero_merges = [
        (t, ev)
        for t, ev in merged_outputs
        if isinstance(ev.value, dict) and ev.value.get("index") == 0
    ]
    assert scene_zero_merges, (
        "no merged output for scene index 0 — cannot evaluate streaming-first claim"
    )
    first_merge_zero_ts = scene_zero_merges[0][0]

    scenes_done_ts = next(
        (
            t
            for t, ev in timeline
            if isinstance(ev, StepDone) and ev.step == "scenes"
        ),
        None,
    )
    assert scenes_done_ts is not None, "scenes step never emitted StepDone"

    assert first_merge_zero_ts < scenes_done_ts, (
        "streaming-first violated: first merged(scene 0) arrived at "
        f"t={first_merge_zero_ts:.3f}s but scenes StepDone arrived earlier at "
        f"t={scenes_done_ts:.3f}s. Downstream completions outpaced the upstream "
        "stream's remaining scenes — increase scenes prompt length or shrink "
        "downstream max_tokens to restore the comparison's headroom."
    )
