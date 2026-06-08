# streamwright

**Streaming-first orchestration for multi-LLM pipelines. Fan out across providers, merge by key, and stream partial results to your users in seconds instead of minutes.**

[![PyPI version](https://img.shields.io/pypi/v/streamwright.svg)](https://pypi.org/project/streamwright/)
[![Python versions](https://img.shields.io/pypi/pyversions/streamwright.svg)](https://pypi.org/project/streamwright/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-178%20passing-brightgreen)

---

streamwright is a small, dependency-light Python library for orchestrating
asynchronous multi-LLM pipelines as a directed acyclic graph (DAG) of steps.
Its defining feature is that downstream steps begin consuming from upstream the
moment upstream starts yielding, not when it finishes. User-visible latency
tracks the *first* useful result rather than the slowest one. It ships with
adapters for Anthropic, OpenAI, and Moonshot behind one uniform `Provider`
interface, plus fan-out with concurrency caps, keyed merges across parallel
branches, retries with exponential backoff, and clean cancellation.

## Table of contents

- [Why streamwright](#why-streamwright)
- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Use cases](#use-cases)
- [Calling an LLM](#calling-an-llm)
- [Provider support](#provider-support)
- [API overview](#api-overview)
- [How it compares](#how-it-compares)
- [Configuration](#configuration)
- [Status](#status)
- [Contributing](#contributing)
- [License](#license)

## Why streamwright

Picture five LLM calls in a pipeline. Each takes 30 seconds. Composed
sequentially, your users wait two and a half minutes staring at a spinner, even
though the first useful piece of output was ready after 30 seconds. The problem
is sequential composition: every step waits for the previous one to fully finish
before any work begins.

streamwright treats every step as a producer and consumer connected by a bounded
async queue. As soon as an upstream STREAM step yields its first item, a
downstream fan-out step can start processing it, and it emits its own results as
they arrive. The pipeline still produces every output the sequential version
would have. It simply reaches the first useful result an order of magnitude
faster, and it keeps the throughput high by overlapping work that used to run
back to back.

If you have ever wired up `asyncio.gather` by hand, lost track of which task
owns which result, and then bolted on retries and cancellation until the code
became unreadable, streamwright is the structured version of that pattern.

## Features

- **Streaming-first scheduler.** Downstream steps consume upstream output as it
  is produced, so latency tracks the first result.
- **Three step kinds.** SINGLE (one value in, one value out), STREAM (yields
  many items), and MERGE (joins parallel branches by key).
- **Fan-out with concurrency caps.** Spread a stream across workers with a
  per-step semaphore so you never exceed a provider rate limit.
- **Keyed merges.** Join the outputs of several branches by a key function, with
  a strict mode that flags incomplete keys and a lenient mode that drops them.
- **Multi-provider abstraction.** Anthropic, OpenAI, and Moonshot behind one
  `Provider` interface. Switch providers by changing a string.
- **Token-honest usage.** Visible and hidden reasoning tokens are reported
  separately so spend tracking and context budgeting stay accurate.
- **Retries and cancellation.** Transient provider errors retry with exponential
  backoff. Closing the event stream early cancels every in-flight task cleanly,
  with no leaked work.
- **Typed and tested.** Fully type-annotated, `mypy --strict` clean, and covered
  by 178 unit tests. No durable-execution infrastructure required, just
  `pip install`.

## Install

```bash
pip install streamwright
# or with uv
uv add streamwright
```

From source:

```bash
pip install git+https://github.com/arjunkarth1k/streamwright-python
```

streamwright requires Python 3.12 or newer. Its only runtime dependencies are
`anyio`, `httpx`, `pydantic`, and `typing-extensions`.

## Quick start

This pipeline has no LLM calls at all, so you can run it with nothing installed
beyond streamwright. It demonstrates the core behavior: a downstream step
processing items while the upstream step is still producing them.

```python
import asyncio
from collections.abc import AsyncIterator
from typing import Any

from streamwright import Pipeline

pipeline = Pipeline("demo")


@pipeline.step()
async def urls(ctx: Any, _input: Any) -> AsyncIterator[str]:
    """Stream three URLs with a small delay between each."""
    for i in range(3):
        yield f"https://example.com/item-{i}"
        await asyncio.sleep(0.05)


@pipeline.step(fan_out_from="urls", max_concurrency=2)
async def fetch(ctx: Any, url: str) -> dict[str, Any]:
    """Pretend to fetch. Sleeps to simulate I/O."""
    await asyncio.sleep(0.1)
    return {"url": url, "status": 200}


async def main() -> None:
    async for event in pipeline.run("start"):
        print(f"{type(event).__name__}: {event}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it and you will see the downstream `fetch` step begin processing `item-0`
*before* the upstream `urls` step yields `item-2`. That interleaving is the whole
point:

```
StepStarted: StepStarted(step='urls')
StepStarted: StepStarted(step='fetch')
StepStreaming: StepStreaming(step='urls')
StepOutput: StepOutput(step='urls', value='https://example.com/item-0', key=0)
StepOutput: StepOutput(step='urls', value='https://example.com/item-1', key=1)
StepOutput: StepOutput(step='fetch', value={'url': 'https://example.com/item-0', 'status': 200}, key=0)
StepOutput: StepOutput(step='urls', value='https://example.com/item-2', key=2)
StepOutput: StepOutput(step='fetch', value={'url': 'https://example.com/item-1', 'status': 200}, key=1)
StepDone: StepDone(step='urls')
StepOutput: StepOutput(step='fetch', value={'url': 'https://example.com/item-2', 'status': 200}, key=2)
StepDone: StepDone(step='fetch')
PipelineDone: PipelineDone()
```

## Core concepts

### Steps

A pipeline is a set of steps connected into a DAG. You declare a step by
decorating an async function with `@pipeline.step()`. The first argument is
always a `Context` (`ctx`), and the second is the input value. There are three
kinds of step, and streamwright infers the kind from how you write the function:

- **SINGLE.** The function returns one value. One input produces one output.
- **STREAM.** The function is an async generator (it uses `yield`). One input
  produces many outputs, each emitted as soon as it is ready. A STREAM step is
  what enables downstream interleaving.
- **MERGE.** Declared with `@pipeline.merge(...)`. It joins the outputs of two
  or more upstream steps by a key, and fires once per key that every source has
  contributed to.

### Fan-out

When a step sets `fan_out_from="upstream"`, streamwright invokes it once per item
the upstream step emits, rather than once for the whole batch. Add
`max_concurrency=N` to cap how many invocations run at the same time. This is the
knob you use to respect provider rate limits while still running in parallel.

### Keys

Every value flowing through the pipeline carries a key. Entry inputs are keyed
`0`, and a STREAM source assigns `0, 1, 2, ...` to the items it yields. Keys
propagate downstream so a MERGE step can line up results that belong together.
MERGE supports a strict mode (the default), which emits a `StepFailed` event at
end of run if a key never received a value from every source, and a lenient mode,
which logs a warning and drops the partial set.

### Events

`pipeline.run(input)` returns an async iterator of events. You stream these
straight to your user or to your telemetry system:

- `StepStarted`, `StepStreaming`, `StepOutput`, `StepDone`, `StepFailed`
- `PipelineDone` when the whole run completes
- `Telemetry` for usage and timing data

### Providers

`ctx.llm("provider/model")` resolves a `"provider/model"` string to a cached
provider instance plus the model id. Provider instances are pooled one per
provider per process, so repeated calls reuse the same HTTP client. Call
`await streamwright.aclose()` at shutdown to release those clients cleanly.

## Use cases

streamwright fits any workload where you fan a task out across many model calls
and want the user to see results as they land. A few concrete examples:

- **Streaming research and summarization.** Stream a list of topics, summarize
  each with a fast model in parallel, and render each summary to the user the
  instant it returns rather than waiting for the slowest topic.
- **RAG over many documents.** Fan a retrieved document set out across a
  per-chunk extraction step, then MERGE the structured fields back together by
  document id.
- **Multi-provider ensembles.** Send the same prompt to Anthropic, OpenAI, and
  Moonshot concurrently, then MERGE their answers by request id to compare,
  vote, or pick the cheapest acceptable response.
- **Parallel structured extraction.** Use `stream_json_array` to coerce a model
  into emitting a JSON array of objects and consume each object as soon as it
  parses, instead of blocking on the full response.
- **Fan-out classification or labeling.** Spread thousands of items across a
  bounded pool of model calls with `max_concurrency`, stream labels back, and
  retry only the transient failures.
- **Cost-aware routing.** Resolve a cheap model for the easy branch and an
  expensive one for the hard branch, all in the same pipeline, switching by
  changing the `"provider/model"` string.

## Calling an LLM

Here is a pipeline that streams a list of topics and summarizes each one with a
fast model, fanning out across three concurrent calls.

```python
import asyncio

import streamwright
from streamwright import Message, Pipeline

pipeline = Pipeline("summarize")


@pipeline.step()
async def topics(ctx, _input) -> "AsyncIterator[str]":
    for topic in ("photosynthesis", "tectonic plates", "neutron stars"):
        yield topic


@pipeline.step(fan_out_from="topics", max_concurrency=3)
async def summarize(ctx, topic: str) -> dict:
    provider, model = await ctx.llm("anthropic/claude-haiku-4-5")
    result = await provider.complete(
        model=model,
        messages=[Message(role="user", content=f"One-sentence summary of {topic}.")],
        max_tokens=80,
    )
    return {"topic": topic, "summary": result.text}


async def main() -> None:
    async for event in pipeline.run("start"):
        print(event)
    await streamwright.aclose()  # release pooled HTTP clients at shutdown


if __name__ == "__main__":
    asyncio.run(main())
```

Set the relevant environment variable before running: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, or `MOONSHOT_API_KEY`.

### Streaming tokens directly

When you want raw token deltas rather than a completed result, call
`provider.stream(...)` and iterate the events:

```python
provider, model = streamwright.get_provider("openai/gpt-5.2")
async for event in provider.stream(
    model=model,
    messages=[Message(role="user", content="Explain async I/O in two sentences.")],
    max_tokens=120,
):
    match event:
        case streamwright.TextDelta(text=chunk):
            print(chunk, end="", flush=True)
        case streamwright.Done(finish_reason=reason):
            print(f"\n[done: {reason}]")
```

Reasoning-family models (the GPT-5 family and Moonshot Kimi K2.x) also emit
`ReasoningDelta` events for hidden chain-of-thought, which you can display
separately, suppress, or route to a different surface.

## Provider support

| Provider | Models | Streaming | Tools | Structured | Vision | Cache | Context |
|---|---|---|---|---|---|---|---|
| Anthropic | claude-opus-4-7 | yes | yes | yes | yes | yes | 1M |
| | claude-sonnet-4-6 | yes | yes | yes | yes | yes | 1M |
| | claude-haiku-4-5 | yes | yes | yes | yes | yes | 200K |
| OpenAI | gpt-5.5 | yes | yes | yes | yes | yes | 1M |
| | gpt-5.5-pro | no | yes | yes | yes | yes | 1M |
| | gpt-5.2 | yes | yes | yes | yes | yes | 400K |
| Moonshot | kimi-k2.6 | yes | yes | no | yes | partial | 256K |
| | kimi-k2.5 | yes | yes | no | no | partial | 256K |

Moonshot's prompt-cache API requires a separate two-call handshake that this
adapter does not yet implement. Passing `cache=True` to the Moonshot adapter
raises `NotImplementedError`. See [docs/ROADMAP.md](docs/ROADMAP.md) for the full
list of planned work.

## API overview

Everything you need is exported from the top-level `streamwright` package.

| Symbol | Purpose |
|---|---|
| `Pipeline` | Declare steps and run the DAG. |
| `step`, `merge` | Decorators for SINGLE/STREAM steps and MERGE steps. |
| `get_provider(spec)` | Resolve `"provider/model"` to a cached provider and model id. |
| `aclose()` | Close every pooled provider HTTP client at shutdown. |
| `Message`, `Tool` | Conversation turns and tool definitions sent to a provider. |
| `CompletionResult`, `Usage` | Result and token usage of a non-streaming call. |
| `TextDelta`, `ReasoningDelta`, `ToolCallDelta`, `UsageEvent`, `Done` | Streaming events from `provider.stream(...)`. |
| `Provider` | The provider interface, for writing your own adapter. |
| `UnknownModelError`, `CapabilityError` | Provider-layer errors. |

Each provider exposes three calls: `complete(...)` for a single response,
`stream(...)` for token-level streaming, and `stream_json_array(...)` for
streaming a JSON array of objects as each one parses.

## How it compares

| | streamwright | LangGraph | Inngest | Temporal | asyncio.gather |
|---|---|---|---|---|---|
| Primary use case | Streaming multi-LLM pipelines | Stateful agent graphs | Durable event-driven workflows | Long-running durable workflows | Concurrent in-process tasks |
| LLM-native | yes | yes (via LangChain) | no | no | no |
| First-class streaming | yes, downstream consumes upstream as items arrive | partial, graph-level | realtime channel, not token-shaped | workflow streams on top of signals | do it yourself |
| Multi-provider abstraction | built in | via LangChain | none | none | none |
| Fan-out with concurrency caps | yes, per-step semaphore | yes | yes | yes | manual |
| Join/merge by key | yes, strict or lenient | via state reducers | via event correlation | via workflow logic | manual |
| Durable across crashes | no, single process | yes, checkpointers | yes | yes | no |
| Infrastructure required | none | none | server | server and workers | none |
| Best when | you stream partial results to a user as a multi-LLM pipeline runs | you build stateful agents with checkpointing | you need durable workflows that survive crashes | you run mission-critical workflows for hours to years | you have a handful of independent async tasks |

Each of these projects is excellent at what it was built for. streamwright fills
the specific gap of streaming-first multi-LLM orchestration without requiring
durable-execution infrastructure.

## Configuration

streamwright reads provider credentials from the environment. Set whichever you
use:

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic models |
| `OPENAI_API_KEY` | OpenAI models |
| `MOONSHOT_API_KEY` | Moonshot models |

You can keep these in a `.env` file at your project root. Copy
[.env.example](.env.example) to `.env` and fill in the keys you have. The
provider constructors also accept an explicit `api_key=` or a preconfigured
`httpx.AsyncClient` if you would rather not use environment variables.

## Status

streamwright is **v0.1**: it works, it is well tested (178 tests passing), and it
is in active development. The provider abstraction, scheduler, and pipeline DSL
are stable for everyday use. The remaining items in
[docs/ROADMAP.md](docs/ROADMAP.md), such as native structured outputs,
pricing and telemetry wiring, per-invocation events, composite keys for nested
streams, and pipeline-level deadlines, are planned for subsequent releases.

A live integration suite under `tests/integration/` exercises each provider and a
multi-provider pipeline against the real APIs. Run it with `just integration`
(it requires API keys, see
[CONTRIBUTING.md](CONTRIBUTING.md#running-integration-tests)).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development
setup, code style, and the test and check workflow. The standing collaboration
rules live in [AGENTS.md](AGENTS.md), and they apply to humans too.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Authors

- **Arjun Karthik** ([akart07@gmail.com](mailto:akart07@gmail.com))
- **Prakhar Dagur** ([prakharevan@gmail.com](mailto:prakharevan@gmail.com))
