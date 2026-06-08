# Changelog

All notable changes to streamwright will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-05-23

Initial public release.

### Added

#### Provider layer
- `Provider` Protocol with `complete`, `stream`, and `stream_json_array`
  methods, plus a concrete `BaseProvider` that adapters subclass.
- `AnthropicProvider`, `OpenAIProvider`, `MoonshotProvider` adapters
  against the real Messages, Chat Completions, and OpenAI-compatible
  endpoints respectively. SSE streaming, tool-use, prompt caching, and a
  uniform `cache: bool` intent parameter (with per-provider semantics:
  Anthropic `cache_control: ephemeral`; OpenAI `prompt_cache_key`
  derived from a sha256 of the messages; Moonshot raises
  `NotImplementedError` until the two-call context-cache handshake
  lands).
- Capability tables verified against current docs (May 2026):
  Anthropic Opus 4.7 / Sonnet 4.6 / Haiku 4.5 (with the
  `claude-haiku-4-5-20251001` dated snapshot); OpenAI gpt-5.5,
  gpt-5.5-pro (streaming disabled), gpt-5.2; Moonshot Kimi k2.6 / k2.5.
- `get_provider("provider/model")` registry returning a cached
  `(Provider, model_id)` tuple, plus `get_capabilities` for capability
  lookups.
- `streamwright.aclose()` for clean shutdown of cached provider HTTP
  clients.
- Streaming JSON-array parser (`JsonArrayBuffer`) and shared SSE event
  parser (`iter_sse_events`).

#### Pipeline DSL and scheduler
- `Pipeline` with `@pipeline.step(...)`, `@pipeline.merge(...)`, and a
  declarative DAG that validates at `pipeline.run()` call time.
- `Pipeline.run(input)` returns an `AsyncIterator[JobEvent]` —
  `StepStarted`, `StepStreaming` (STREAM steps only), `StepOutput`,
  `StepDone`, `StepFailed`, `PipelineDone`, and `Telemetry`.
- Streaming-first execution: downstream steps consume from upstream
  STREAM sources as items arrive (not when upstream completes).
- Fan-out via `fan_out_from=` and `max_concurrency=`; STREAM sources
  assign position keys 0, 1, 2, … that propagate through the DAG.
- MERGE steps joining N upstream sources by a `key=` callable, with
  `mode="strict"` (default — emits `StepFailed` on incomplete key-sets
  at end-of-run) or `mode="lenient"` (warns and drops).
- Per-step retries with exponential backoff (`with_retries`) that
  retry on `httpx.TimeoutException`, `asyncio.TimeoutError`,
  5xx/429 HTTP statuses, and `ProviderError.retryable=True`, and
  abort immediately on fatal errors (including `UnknownModelError`
  and `CapabilityError`).
- Cancellation safety: closing the `Pipeline.run` iterator early
  cancels every in-flight runner without leaking tasks.
- Bounded inter-step queues (default maxsize 32) provide natural
  backpressure when a downstream is slower than its upstream.

#### Context
- Per-invocation `Context` exposing `job_id`, `step_name`,
  `emit(event)` for custom telemetry, `log(msg, **kwargs)` with
  job and step IDs attached automatically, and async
  `llm("provider/model")` that resolves via the provider registry.

#### Tests
- 158 tests covering the provider adapters (via `httpx.MockTransport` —
  no real network), the JSON parser, the SSE parser, the capability
  tables, the scheduler's lifecycle, retry, MERGE, cancellation,
  backpressure, and documented limitations.

### Documentation
- `README.md`, `docs/ROADMAP.md`, `CONTRIBUTING.md`, `AGENTS.md`.
- `examples/streaming_fanout.py` — runnable demo of the streaming +
  fan-out value prop.

[Unreleased]: https://github.com/arjunkarth1k/streamwright-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/arjunkarth1k/streamwright-python/releases/tag/v0.1.0
