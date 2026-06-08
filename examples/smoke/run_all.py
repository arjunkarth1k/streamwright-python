"""Run every provider smoke script in sequence and print a summary.

Wraps ``anthropic_smoke.py``, ``openai_smoke.py``, and
``moonshot_smoke.py`` so a single command can exercise all three.

The smoke scripts themselves treat any failure (including a missing API
key) as exit 1, per spec. This runner detects the missing-key case
*before* invoking each subprocess by checking the relevant env var
itself (after a single ``load_dotenv()`` so a local ``.env`` is
honored), so the per-provider summary line can distinguish:

* ``PASS`` — subprocess exited 0
* ``SKIPPED-no-key`` — required env var is unset; subprocess not run
* ``FAIL (exit N)`` — subprocess exited non-zero

The overall runner exits non-zero only if at least one provider FAILed;
missing-key skips are not failures. Cross-platform: invokes the smoke
scripts via the current Python interpreter so the same recipe works on
Windows, macOS, and Linux.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("anthropic_smoke.py", "ANTHROPIC_API_KEY"),
    ("openai_smoke.py", "OPENAI_API_KEY"),
    ("moonshot_smoke.py", "MOONSHOT_API_KEY"),
)
_HERE = Path(__file__).resolve().parent


def main() -> int:
    # Quiet python-dotenv's parse-error warning, then print our own
    # clearer hint if the load reported nothing despite a .env being
    # present. See anthropic_smoke.py:_load_env_quietly for the
    # rationale — verbose=False does not suppress this warning;
    # logger.ERROR is the only knob that does.
    logging.getLogger("dotenv.main").setLevel(logging.ERROR)
    loaded = load_dotenv()
    if not loaded and os.path.exists(".env"):
        print(
            '[smoke] .env exists but no variables loaded. If a value '
            'contains "=", "#", "$", or spaces, wrap it in double '
            'quotes (KEY="value with $special chars").',
            file=sys.stderr,
        )
    results: list[tuple[str, str]] = []
    for script, env_key in _SCRIPTS:
        print(f"\n=== {script} ===", flush=True)
        if not os.environ.get(env_key):
            print(
                f"{env_key} is not set; skipping (no key)",
                file=sys.stderr,
                flush=True,
            )
            results.append((script, "SKIPPED-no-key"))
            continue
        path = _HERE / script
        completed = subprocess.run([sys.executable, str(path)], check=False)
        status = "PASS" if completed.returncode == 0 else f"FAIL (exit {completed.returncode})"
        results.append((script, status))

    print("\n=== summary ===")
    any_failed = False
    for script, status in results:
        print(f"  {script}: {status}")
        if status.startswith("FAIL"):
            any_failed = True
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
