"""Campaign entry point that can actually be asked to stop."""

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
