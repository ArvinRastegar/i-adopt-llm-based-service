"""Command line: ``iadopt-linking run --out results/<name>`` and its phases."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .compare import compare_runs, render_comparison
from .errors import RunError, UsageError
from .experiment import load_population, log, query_keys, run
from .recorder import StoreError
from .rerank import DEFAULT_MAX_DESCRIPTION_CHARS, ScoreError
from .retrieve import RetrievalTimeout
from .score import PoolError

# Everything a run stops with on purpose: a named reason, exit code 2, no traceback.
STOPS = (RunError, RetrievalTimeout, PoolError, ScoreError, StoreError)


def _positive(kind: type) -> Callable[[str], Any]:
    """Build an argparse type that accepts only a finite number above zero.

    Args: kind: int or float.
    Returns: a parser that raises ArgumentTypeError for anything else.
    Raises: None.
    Side effects: none.
    """
    def parse(text: str) -> Any:
        """Read one option value.

        Args: text: the command-line text.
        Returns: the number.
        Raises: argparse.ArgumentTypeError when it is not a finite number above zero.
        Side effects: none.
        """
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a {kind.__name__}") from None
        if not (math.isfinite(value) and value > 0):
            raise argparse.ArgumentTypeError(f"{text!r} must be greater than 0")
        return value
    return parse


def build_parser() -> argparse.ArgumentParser:
    """Define the command line.

    Args: none.
    Returns: the argument parser.
    Raises: None.
    Side effects: none.
    """
    parser = argparse.ArgumentParser(
        prog="iadopt-linking", description=__doc__,
        epilog="Exit codes: 0 success; 2 a check or limit stopped the run, or the request is invalid.")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="show the population and the queries, call nothing")
    full = commands.add_parser(
        "run", help="retrieve, score, check and report; resumes where a previous run stopped")
    for command in (plan, full):
        command.add_argument("--only", help="comma-separated variable file stems (a pilot)")
        command.add_argument("--lab-root", help="iadopt-lab root (default: sibling folder)")
    compare = commands.add_parser(
        "compare", help="compare the stored evidence of two runs (calls nothing)")
    compare.add_argument("--first", required=True, help="first results folder")
    compare.add_argument("--second", required=True, help="second results folder")
    compare.add_argument("--out", help="write the Markdown here instead of printing it")
    full.add_argument("--out", required=True, help="results folder")
    full.add_argument("--max-retrieval-minutes", type=_positive(float), default=30.0)
    full.add_argument("--max-description-chars", type=_positive(int),
                      default=DEFAULT_MAX_DESCRIPTION_CHARS)
    full.add_argument("--device", choices=("mps", "cpu"), help="reranker device (default: mps)")
    devices = full.add_mutually_exclusive_group()
    devices.add_argument("--skip-device-check", action="store_true",
                         help="do not cross-check scores on the other device")
    devices.add_argument("--recheck-devices", action="store_true",
                         help="repeat the cross-device check even if one passed for these scores")
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    """Run the chosen command.

    Args: args: parsed arguments.
    Returns: 0 on success.
    Raises: UsageError for an invalid request; the STOPS errors when a run is stopped.
    Side effects: depends on the command.
    """
    if args.command == "compare":
        text = render_comparison(compare_runs(args.first, args.second),
                                 (Path(args.first).name, Path(args.second).name))
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    if args.command == "plan":
        _, slots, counts = load_population(args.lab_root, only)
        log(f"population counts: {counts}")
        log(f"{len(slots)} slots, {len(query_keys(slots))} distinct queries x 3 sources")
        return 0
    run(args.out, lab_root=args.lab_root, only=only,
        max_retrieval_minutes=args.max_retrieval_minutes,
        max_description_chars=args.max_description_chars, device=args.device,
        check_devices=not args.skip_device_check, recheck_devices=args.recheck_devices)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command.

    Args: argv: arguments; defaults to sys.argv.
    Returns: 0 on success, 2 when a check, a limit or the request stopped the run.
    Raises: SystemExit from argparse for bad usage.
    Side effects: depends on the command.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except UsageError as exc:
        parser.error(str(exc))
    except STOPS as exc:
        log(f"STOPPED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
