"""Anthropic smoke test — exercise AnthropicProvider against the real API.

Runs two calls (``complete`` then ``stream``) with the cheapest registered
Anthropic model and prints latency / token usage for each. Intended to be
invoked by hand:

    uv run python examples/smoke/anthropic_smoke.py

Reads ``ANTHROPIC_API_KEY`` from the environment (or ``.env`` at the repo
root via python-dotenv). Exits 1 with a clear message if the key is
missing or any call raises.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import traceback

from dotenv import load_dotenv

import streamwright
from streamwright import Done, Message, TextDelta, UsageEvent

PROVIDER_SPEC = "anthropic/claude-haiku-4-5"
ENV_KEY = "ANTHROPIC_API_KEY"
PROMPT = "Say hi in exactly 5 words."
MAX_TOKENS = 20


async def _run() -> None:
    provider, model = streamwright.get_provider(PROVIDER_SPEC)
    messages = [Message(role="user", content=PROMPT)]

    # --- complete() -------------------------------------------------------
    t0 = time.perf_counter()
    result = await provider.complete(
        model=model, messages=messages, max_tokens=MAX_TOKENS
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"[complete] text={result.text!r}")
    print(
        f"[complete] tokens_in={result.usage.tokens_in} "
        f"tokens_out={result.usage.tokens_out} "
        f"latency_ms={latency_ms:.1f}"
    )

    # --- stream() ---------------------------------------------------------
    print("[stream] ", end="", flush=True)
    t0 = time.perf_counter()
    chunks: list[str] = []
    tokens_in = 0
    tokens_out = 0
    finish_reason: str | None = None
    async for event in provider.stream(
        model=model, messages=messages, max_tokens=MAX_TOKENS
    ):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
            chunks.append(event.text)
        elif isinstance(event, UsageEvent):
            tokens_in = event.tokens_in
            tokens_out = event.tokens_out
        elif isinstance(event, Done):
            finish_reason = event.finish_reason
    latency_ms = (time.perf_counter() - t0) * 1000
    print(
        f"\n--- assembled={''.join(chunks)!r} tokens_in={tokens_in} "
        f"tokens_out={tokens_out} latency_ms={latency_ms:.1f} "
        f"finish={finish_reason!r}"
    )


def _load_env_quietly() -> None:
    """Suppress dotenv's parse-error warning; print our own hint instead.

    python-dotenv's parse warning ("could not parse statement starting at
    line N") fires unconditionally via the ``dotenv.main`` logger and is
    not affected by ``verbose=False`` (verified in the installed source).
    Mute the logger, then if ``load_dotenv()`` reports it loaded nothing
    AND a ``.env`` file actually exists, print a clearer hint that
    points at the usual culprit (special characters needing quoting).
    """
    logging.getLogger("dotenv.main").setLevel(logging.ERROR)
    loaded = load_dotenv()
    if not loaded and os.path.exists(".env"):
        print(
            '[smoke] .env exists but no variables loaded. If a value '
            'contains "=", "#", "$", or spaces, wrap it in double '
            'quotes (KEY="value with $special chars").',
            file=sys.stderr,
        )


async def main() -> int:
    _load_env_quietly()
    if not os.environ.get(ENV_KEY):
        print(
            f"ERROR: {ENV_KEY} is not set. Copy .env.example to .env at the "
            "repo root and fill in your key, or export the variable in your "
            "shell.",
            file=sys.stderr,
        )
        return 1
    try:
        await _run()
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        await streamwright.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
