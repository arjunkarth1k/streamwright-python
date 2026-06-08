# Contributing to streamwright

Thanks for considering a contribution. This is a young project and we
prefer high-trust collaboration — a few short rules and a willingness to
ask questions go a long way.

## Development setup

```bash
git clone https://github.com/arjunkarth1k/streamwright-python
cd streamwright
uv sync --dev      # installs runtime + dev dependencies into .venv
just check         # runs ruff, mypy strict, and pytest
```

Don't have `just`? The recipes are one-liners in `justfile`; run them
directly with `uv run ...` if you prefer.

## Code style

- **Ruff** enforces formatting and a small lint surface (`E`, `F`, `I`,
  `UP`, `B`). `just format` will format in place; `just lint` checks.
- **mypy strict** is enabled. No `Any` without an explanatory comment.
- **Line length 100**. Type-hint everything. Short docstrings on public
  modules, classes, and functions.

## Tests

New behavior needs a test. We use `pytest` with `pytest-asyncio` in auto
mode. Mock external services with `httpx.MockTransport` — no test should
make real network calls.

Run the suite:

```bash
just test       # uv run pytest
just check      # the full lint + typecheck + tests gate
```

## Running integration tests

Some smoke and integration tests hit real provider APIs and need keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MOONSHOT_API_KEY`). They are
off by default and never run on the standard `just check` / `just test`
gates.

**Local — recommended:** copy `.env.example` to `.env` at the repo root
and fill in your real keys. The smoke scripts and the integration test
conftest load it automatically via `python-dotenv`. Then:

```bash
just smoke         # runs all three provider smoke scripts in sequence
just integration   # runs the live integration test suite
```

The integration suite gates collection on either
`STREAMWRIGHT_RUN_INTEGRATION=1` or pytest's `-m integration` marker
expression. Per-provider tests with their key missing skip cleanly
rather than failing — partial-key local setups are fine.

**Cost guardrail:** each full run is under $0.10 at current pricing.
Both `just smoke` and `just integration` cap every model call at small
`max_tokens` budgets, but they still make real network calls and
spend real money.

**CI policy:** the `.github/workflows/integration.yml` job runs only
on (1) manual `workflow_dispatch` from the GitHub Actions UI by
someone with write access, or (2) a pushed release tag matching
`v*.*.*`. The default `tests` workflow stays hermetic and never
touches a paid endpoint.

For the CI workflow to succeed, three repository secrets must be
configured first: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and
`MOONSHOT_API_KEY`. Add them at
<https://github.com/arjunkarth1k/streamwright-python/settings/secrets/actions>
(repo settings → Secrets and variables → Actions). Tests whose key is
absent skip cleanly rather than failing the run, so partial-secret
configurations are valid for incremental rollout.

> `.env` is gitignored and must stay that way — never commit real keys.

## Standing instructions

`AGENTS.md` at the repo root has the agent collaboration rules — they
apply to humans too:

1. Flag problems, don't work around them.
2. Surface ideas under a `## Suggestions` heading at the end of your
   change — don't smuggle them in.
3. Ask when ambiguous.
4. Verify stale knowledge against current docs.
5. Run the checks before declaring done.

## Pull requests

Small, scoped PRs over large ones. Include a brief summary, any
relevant issue links, and the output of `just check` (or note any
intentional skip and why). Reviewers reserve the right to ask for
splits.

## License

By contributing, you agree your contributions are licensed under
[Apache License 2.0](LICENSE).
