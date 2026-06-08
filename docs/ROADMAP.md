# Roadmap

Items deferred from prior work. Each one has been discussed and triaged
but not yet implemented.

## Per-invocation events

Today the scheduler emits ``StepStarted`` and ``StepDone`` exactly once
per step, regardless of how many fan-out invocations ran. That's the
right granularity for most observability tooling (one row per step in
a trace, one timer per step in metrics) so the existing events stay.
But debugging a slow tail invocation or counting real-time parallelism
needs finer signal. Add ``StepInvocationStarted`` and
``StepInvocationDone`` events that carry both the step name and the
invocation's key, and emit them per-invocation alongside the
per-step lifecycle events. Existing consumers ignore unfamiliar
variants, so this is backwards-compatible.

## Multi-entry pipelines

The DAG validator accepts pipelines with multiple entry steps (no
upstream dependency). Today the scheduler feeds the same
``input_value`` to every entry step — the simplest semantic but
probably not what callers want. Realistic use cases need either
per-entry inputs (eg ``pipeline.run({"a": ..., "b": ...})``) or a
mechanism for one entry to run conditionally based on a runtime
predicate. Design decision pending a real use case so we don't
over-fit on a guess.

## Pipeline-level deadline and watchdog

A single runner blocked forever — eg ``await put()`` to a downstream
whose consumer crashed without setting ``_DONE`` — would hang
``Pipeline.run()`` indefinitely. Add ``max_runtime: float | None``
on ``Pipeline.run`` that cancels everything if the wall-clock budget
is exceeded, plus a per-step inactivity watchdog that detects runners
not making progress between events. Production deployments where a
hung pipeline is worse than a failed pipeline need this; development
environments can leave it ``None``.

## Composite keys for nested STREAM steps

The scheduler currently propagates the upstream item's key unchanged
through every yielded sub-item of a downstream STREAM step. Concretely:
if STREAM ``A`` yields three items keyed 0/1/2 and STREAM ``B`` fans out
from ``A``, every value ``B`` yields while handling ``A``-item-``k``
carries key ``k`` — regardless of how many sub-items ``B`` produces.

The visible failure is that **STREAM → STREAM → MERGE pipelines are
unexpressible** today: the MERGE sees all of ``B``'s sub-items from one
``A``-item colliding under the same key, so they can't be joined to a
parallel stream by sub-position.

**Planned fix**: composite (tuple) keys. A downstream STREAM step's
``n``-th yield carries the key ``(upstream_key, n)`` instead of
``upstream_key``. SINGLE downstream steps keep emitting ``upstream_key``
(no sub-position). Existing single-level pipelines are unaffected; the
``StepOutput.key`` field already accepts ``Hashable``, so the tuple
form is type-compatible.

Implementation sketch:

1. ``_invoke_stream`` already tracks a per-invocation ``position``
   counter; for non-source STREAM steps, emit
   ``item_key = (input_key, position)`` instead of ``input_key``.
2. MERGE's ``key`` callable receives the value, not the key — so MERGE
   authors writing across nested-stream pipelines can still extract the
   join dimension from the value. No MERGE-side changes required.
3. Add a regression test that mirrors the current limitation test and
   asserts the new composite-key behavior. Update the module docstring
   on ``streamwright/scheduler.py`` to remove the limitation callout.

## Native structured-output APIs

`BaseProvider.stream_json_array` currently coerces the model with a
prompt-injected system instruction. Models still drift. Replace with
provider-native structured-output:

- **OpenAI** — set `response_format = {"type": "json_schema", "json_schema": {...}}`
  on `/v1/chat/completions` when a schema is supplied. Strict-mode JSON
  schema rejects malformed outputs server-side, not via parser luck.
- **Anthropic** — use a single forced `tool_choice` with a tool whose
  `input_schema` is the desired schema. The model's tool input IS the
  structured output; collect it via `ToolCallDelta` instead of parsing
  free text.
- **Moonshot** — confirm support; likely follows the OpenAI pattern.

The current JSON-array parser stays useful for general streaming JSON
even after structured outputs land. Keep both code paths.

## Pricing data

`ModelPricing` exists as a dataclass placeholder with all-`None` fields.
`streamwright.types.Telemetry` already has a `cost_usd: float` field
ready to be populated. To wire this end-to-end:

1. Add a single source-of-truth file (`streamwright/providers/pricing.toml`
   or similar — TOML is human-editable and the project already depends
   on Python 3.11+ stdlib `tomllib`).
2. Load it lazily at first capability lookup, attach `ModelPricing` to
   each `ModelCapabilities` entry.
3. Add a `cost(usage: Usage, model: str) -> float` helper on the
   capabilities module.
4. Wire `Telemetry` emission in the scheduler once it exists; the
   scheduler is the right place to know per-step usage and emit
   telemetry events with `cost_usd` populated.

Defer until at least one billing-aware use case lands.

## Worked example: accumulating ToolCallDelta

`ToolCallDelta` events stream partial tool-input JSON across chunks.
Callers must concatenate `partial_input` themselves and parse the
combined string with `json.loads` once the tool call is complete
(usually signaled by a `Done` event or a tool index switching).

A short worked example in the README (or a dedicated examples page) is
needed. Sketch:

```python
import json
from collections import defaultdict
from streamwright import ToolCallDelta, Done, get_provider

provider, model = get_provider("anthropic/claude-sonnet-4-6")
inputs: dict[str, list[str]] = defaultdict(list)
names: dict[str, str] = {}

async for ev in provider.stream(model=model, messages=[...], tools=[...]):
    match ev:
        case ToolCallDelta(id=tid, name=tname, partial_input=chunk):
            inputs[tid].append(chunk)
            if tname:
                names[tid] = tname
        case Done():
            for tid, parts in inputs.items():
                args = json.loads("".join(parts))
                print(f"call {tid} -> {names[tid]}({args!r})")
```

Add as a runnable script under `examples/` when the scheduler can wire
this into a real pipeline step.

## Provider.stream async-generator docstring callout

`Provider.stream` is declared `async def ... -> AsyncIterator[StreamEvent]`,
which is genuinely ambiguous in Python: it could be a coroutine that
returns an `AsyncIterator`, OR an async generator function. The
convention in this library is that adapters implement it as an async
generator (`async def` + `yield`), so the call site is
`async for event in provider.stream(...)` — no `await` on the call.

The docstring on `Provider.stream` and `Provider.stream_json_array`
should explicitly state this so users don't try
`stream = await provider.stream(...)` and get confused. One-line
addition; low priority.

## Moonshot reasoning-token visibility

Kimi's chat completion API conflates visible output tokens with hidden
reasoning tokens in `completion_tokens`. There is no
`completion_tokens_details.reasoning_tokens` field as of the current
API docs (<https://platform.kimi.ai/docs/api/chat>). Until upstream
exposes the split, `MoonshotProvider` reports `reasoning_tokens=0` and
`Usage.tokens_out` includes hidden reasoning. The reasoning *content*
(text) is now surfaced via `ReasoningDelta` events and
`CompletionResult.reasoning_text`, so callers can see what the model
thought even if they can't get a per-channel token count from the
provider.

Two paths forward, both deferred:

1. **Upstream**: file a feature request with Moonshot to populate
   `completion_tokens_details.reasoning_tokens` on responses for K2.x
   reasoning models, matching OpenAI's shape. Zero adapter change
   needed once it lands — `_split_usage` already reads that field.
2. **Local fallback**: bundle a Kimi-aware tokenizer (eg `tiktoken`
   with a Kimi-compatible encoding, or Moonshot's published
   tokenizer if released) and count `ReasoningDelta.text` lengths
   client-side. Adds a dependency and a per-token-count CPU cost.
   Only worth it if the upstream feature request stalls and a real
   billing-tracking use case lands.

## Anthropic extended thinking → ReasoningDelta + Usage.reasoning_tokens

Claude's extended-thinking feature returns hidden thinking content via
an opt-in `thinking` parameter on the Messages API. When this
capability is wired into `AnthropicProvider` (currently not exposed as
a `Provider.complete` / `Provider.stream` kwarg), the streaming SSE
parser should route `thinking_delta` events to
:py:class:`ReasoningDelta` (already exists, see
`streamwright/providers/base.py`) and the non-streaming parser should
populate :py:attr:`CompletionResult.reasoning_text` from any `thinking`
content blocks. Token counts go on :py:attr:`Usage.reasoning_tokens`
from the thinking-block token count Anthropic returns, matching the
OpenAI / GPT-5 semantics in `OpenAIProvider._split_usage`.

## GPT-5 forbidden params: logprobs, logit_bias, top_logprobs

Microsoft Learn's reasoning-models page lists `logprobs`, `logit_bias`,
and `top_logprobs` as also unsupported on GPT-5 reasoning variants.
None are currently on the `Provider.complete` / `Provider.stream` kwarg
surface, so no immediate code change. If they're ever added as adapter
kwargs, extend `_GPT_5_FORBIDDEN_SAMPLERS` in `openai.py` to include
them.
