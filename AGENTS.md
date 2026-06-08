# Repository Guidelines

## Agent Standing Instructions
Before completing any task in this repo, agents must:

1. **Flag problems, don't work around them.** If you notice bugs, type errors,
   dead code, leaky abstractions, design smells, or missing edge cases in
   existing code, stop and report it before continuing. Don't silently patch
   around issues.
2. **Surface ideas, don't smuggle them in.** If you have suggestions for
   optimizations, better APIs, simpler implementations, or architectural
   improvements beyond what was asked, list them under a `## Suggestions`
   heading at the end of your response. Do not apply them without approval.
3. **Ask when ambiguous.** If instructions are unclear, contradictory, or
   conflict with the existing repo, ask before guessing.
4. **Verify stale knowledge.** If a dependency, library version, or API
   behavior may have changed since your training cutoff, say so and verify
   against current docs or source rather than assuming.
5. **Run the checks.** After making changes, run the relevant `just` recipes
   (test, lint, typecheck) and report results. If something fails, fix it or
   flag it - never paper over a failure.
6. **Test missing-credential paths via pytest fixtures, not shell unset.**
   When verifying behavior under "no API key" conditions, use pytest's
   `monkeypatch.delenv("KEY_NAME", raising=False)` plus an isolated working
   directory (tmpdir + chdir) so `.env` files in the real working tree can't
   reload the keys. Do NOT rely on shell-level `unset` or PowerShell
   `Remove-Item Env:KEY` — any script that calls `load_dotenv()` will
   re-populate the env from `.env` and defeat the test. If a real no-key
   invocation is unavoidable, temporarily move `.env` aside
   (`mv .env .env.bak`) and restore it in a `try/finally` so an interruption
   never leaves the dev env broken.

The bar is high-trust collaboration: prefer slowing down and raising a flag
over shipping something subtly broken.

## Project Structure & Module Organization
`streamwright/` contains the library code. Core orchestration stubs live in
`pipeline.py`, `scheduler.py`, `context.py`, and `types.py`. Provider adapters
live under `streamwright/providers/`, with shared protocol definitions in
`providers/base.py`. Tests live in `tests/`, using names like
`test_pipeline.py`. Example assets and runnable samples should go in
`examples/`; keep placeholder-only directories with `.gitkeep`.

## Build, Test, and Development Commands
Use `uv` for dependency management and `just` for task running. Install `just`
via `winget install Casey.Just` (Windows), `brew install just` (macOS), or
`cargo install just` / your distro's package manager (Linux). See
https://github.com/casey/just.

- `just --list`: list available recipes.
- `just install`: install runtime and development dependencies.
- `just test`: run the pytest suite (`uv run pytest`).
- `just lint`: run Ruff lint checks (`uv run ruff check .`).
- `just typecheck`: run strict mypy checks on `streamwright` and `tests`.
- `just format`: format Python files with Ruff.
- `just check`: run lint, typecheck, and tests in sequence.

Without `just`, run the raw command bodies directly, for example `uv run pytest`
or `uv run mypy streamwright tests`.

## Coding Style & Naming Conventions
Target Python 3.12+. Use 4-space indentation, full type hints, and short
docstrings for public modules, classes, and functions. Keep line length at 100
characters. Ruff enforces `E`, `F`, `I`, `UP`, and `B` rules, including import
sorting. Use `snake_case` for functions, variables, modules, and test files;
use `PascalCase` for classes and dataclasses. Do not implement placeholder
logic in stubs; use `raise NotImplementedError` and leave focused `TODO`
comments for unresolved design decisions.

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio` in auto mode. Add tests under
`tests/` with file names matching `test_*.py` and test functions named
`test_*`. Async tests may be written directly without explicit event-loop
fixtures. For new behavior, include at least one focused test for success paths
and one for meaningful error or cancellation behavior when applicable.

## Commit & Pull Request Guidelines
The current history only has an initial commit, so follow concise imperative
commit messages such as `Add provider protocol stubs` or `Configure mypy`.
Keep commits scoped to one logical change. Pull requests should include a short
summary, relevant issue links, and verification output for `just test`,
`just lint`, and `just typecheck`. Include screenshots only for future
documentation or UI changes.

## Security & Configuration Tips
Do not commit secrets, API keys, `.env` files, virtual environments, or tool
caches. Provider implementations should read credentials from configuration or
environment variables once the configuration layer is designed.
