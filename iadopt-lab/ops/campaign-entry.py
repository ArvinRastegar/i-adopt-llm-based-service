"""Campaign entry point that can actually be asked to stop.

Identical to `main.py` except for signal disposition. A process started in the
background by a non-interactive shell inherits SIGINT set to SIG_IGN, so neither the
shell nor Python can act on it; a forwarded Ctrl-C is silently discarded. SIGTERM is
never ignored, but Python's default SIGTERM handling terminates the interpreter
without running `finally` blocks, which is precisely where the campaign stores answers
it has already received and paid for.

This restores SIGINT and routes SIGTERM into the same KeyboardInterrupt path that an
interactive Ctrl-C uses, so an unattended supervisor can stop the campaign without
discarding evidence.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _interrupt(signum: int, frame: object) -> None:
    """Turn a termination request into the interruption the runner already handles."""
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGTERM, _interrupt)

from iadopt_lab.cli import main  # noqa: E402  - import after handlers are installed

if __name__ == "__main__":
    raise SystemExit(main())
