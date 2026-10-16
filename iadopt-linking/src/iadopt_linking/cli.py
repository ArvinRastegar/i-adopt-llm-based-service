"""Command line: ``iadopt-linking run --out results/<name>`` and its phases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .chat_choice import CHAT_PASSES, MODEL, REASONING_OFF
from .chat_choice import build_tasks as build_chat_tasks
from .chat_choice import estimate as estimate_chat
from .chat_experiment import run_chat
from .compare import compare_runs, render_comparison
from .errors import RunError, UsageError
from .experiment import load_population, log, query_keys, run
from .jev import ARM_CHOICE, ARM_NOUL, PASSES, PassSpec, pass_spec
from .jev_experiment import PRICE_PER_INPUT_TOKEN_USD, run_jev
from .jev_inputs import build_tasks, estimate, load_baseline
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
    jev_plan = commands.add_parser(
        "jev-plan", help="build every Jev request offline, show one of each kind and the cost "
                         "estimate, call nothing")
    jev_full = commands.add_parser(
        "jev-run", help="ask Jev about the baseline's candidates, check, compare with the "
                        "baseline and report; resumes where a previous run stopped")
    for command in (jev_plan, jev_full):
        command.add_argument("--baseline", required=True, help="finished baseline results folder")
        command.add_argument("--only", help="comma-separated variable file stems (a pilot)")
        command.add_argument("--lab-root", help="iadopt-lab root (default: sibling folder)")
        command.add_argument("--passes", help="comma-separated pass names (default: all)")
    jev_full.add_argument("--out", required=True, help="Jev results folder")
    jev_full.add_argument("--max-cost-usd", required=True, help="hard spending cap in USD")
    jev_full.add_argument("--env-file", help="credential file defining OPENROUTER_API_KEY "
                                             "(default: .env at the repository root)")
    jev_full.add_argument("--workers", type=_positive(int), default=4,
                          help="parallel requests (default 4)")
    jev_full.add_argument("--run-id", help="run name (default: the folder name)")
    psnc_plan = commands.add_parser(
        "psnc-plan", help="build every PSNC chat request offline, show one and the size of the "
                          "run, call nothing")
    psnc_full = commands.add_parser(
        "psnc-run", help="ask a PSNC chat model (reasoning off) about the baseline's candidates, "
                         "check, compare with the baseline and Jev and report; resumes where a "
                         "previous run stopped")
    for command in (psnc_plan, psnc_full):
        command.add_argument("--baseline", required=True, help="finished baseline results folder")
        command.add_argument("--only", help="comma-separated variable file stems (a pilot)")
        command.add_argument("--lab-root", help="iadopt-lab root (default: sibling folder)")
        command.add_argument("--passes", help="comma-separated pass names (default: all four "
                                              "Choice passes)")
        command.add_argument("--model", default=MODEL, choices=sorted(REASONING_OFF),
                             help="PSNC model id with a measured way to turn reasoning off "
                                  f"(default {MODEL})")
    psnc_full.add_argument("--out", required=True, help="results folder of the PSNC run")
    psnc_full.add_argument("--max-requests", type=_positive(int), required=True,
                           help="hard limit on the requests this invocation may send")
    psnc_full.add_argument("--env-file", help="credential file defining PSNC_API_KEY "
                                              "(default: .env at the repository root)")
    psnc_full.add_argument("--workers", type=_positive(int), default=1,
                           help="parallel requests (default 1: answers depend slightly on what "
                                "else the server is processing)")
    psnc_full.add_argument("--run-id", help="run name (default: the folder name)")
    psnc_full.add_argument("--reference", help="finished Jev results folder on the same "
                                               "candidates, to compare with")
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


def _specs(names: str | None) -> list[PassSpec]:
    """Read the pass names of a command line.

    Args: names: comma-separated pass names, or None for all.
    Returns: the passes in the order given (or PASSES).
    Raises: UsageError for an unknown name.
    Side effects: none.
    """
    if not names:
        return list(PASSES)
    try:
        return [pass_spec(name.strip()) for name in names.split(",") if name.strip()]
    except KeyError as exc:
        raise UsageError(f"{exc.args[0]}; passes are {[p.name for p in PASSES]}") from None


def _jev_plan(args: argparse.Namespace, only: list[str] | None) -> None:
    """Show what a Jev run would send and cost, without calling anything.

    Args: args: parsed arguments; only: variable stems or None.
    Returns: None.
    Raises: RunError if the baseline is not usable.
    Side effects: prints; reads the baseline and the corpus.
    """
    baseline = load_baseline(args.baseline, lab_root=args.lab_root, only=only)
    tasks, asked = build_tasks(baseline, _specs(args.passes), session_id="iadopt-linking/plan")
    summary = estimate(tasks, PRICE_PER_INPUT_TOKEN_USD)
    log(f"{asked['slots']} slots, {asked['empty_pools']} without candidates (not asked)")
    log(f"estimate: {json.dumps(summary, sort_keys=True)}")
    for arm in (ARM_CHOICE, ARM_NOUL):
        first = next((t for t in tasks if t.spec.arm == arm), None)
        if first is not None:
            log(f"first {arm} request ({first.spec.name}, {first.slot.slot_id}, sha256 "
                f"{first.sha256[:12]}):\n{json.dumps(first.body, indent=2, ensure_ascii=False)}")


def _chat_specs(names: str | None) -> list[PassSpec]:
    """Read the pass names of a PSNC command line.

    Args: names: comma-separated pass names, or None for all of the chat passes.
    Returns: the passes in the order given (or CHAT_PASSES).
    Raises: UsageError for a name that is not a chat pass.
    Side effects: none.
    """
    if not names:
        return list(CHAT_PASSES)
    by_name = {spec.name: spec for spec in CHAT_PASSES}
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise UsageError(f"No such pass: {unknown[0]!r}; passes are {list(by_name)}")
    return [by_name[name] for name in wanted]


def _chat_plan(args: argparse.Namespace, only: list[str] | None) -> None:
    """Show what a PSNC run would send, without calling anything.

    Args: args: parsed arguments; only: variable stems or None.
    Returns: None.
    Raises: RunError if the baseline is not usable.
    Side effects: prints; reads the baseline and the corpus.
    """
    baseline = load_baseline(args.baseline, lab_root=args.lab_root, only=only)
    tasks, asked = build_chat_tasks(baseline, _chat_specs(args.passes), model=args.model)
    log(f"{asked['slots']} slots, {asked['empty_pools']} without candidates (not asked)")
    log(f"size: {json.dumps(estimate_chat(tasks), sort_keys=True)}")
    first = next(iter(tasks), None)
    if first is not None:
        log(f"first request ({first.spec.name}, {first.slot.slot_id}, sha256 "
            f"{first.sha256[:12]}):\n{json.dumps(first.body, indent=2, ensure_ascii=False)}")


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
    if args.command == "jev-plan":
        _jev_plan(args, only)
        return 0
    if args.command == "jev-run":
        run_jev(args.out, baseline_dir=args.baseline, max_cost_usd=args.max_cost_usd,
                env_file=args.env_file, lab_root=args.lab_root, only=only,
                passes=[s.name for s in _specs(args.passes)] if args.passes else None,
                workers=args.workers, run_id=args.run_id)
        return 0
    if args.command == "psnc-plan":
        _chat_plan(args, only)
        return 0
    if args.command == "psnc-run":
        run_chat(args.out, baseline_dir=args.baseline, max_requests=args.max_requests,
                 model=args.model, env_file=args.env_file, lab_root=args.lab_root, only=only,
                 passes=[s.name for s in _chat_specs(args.passes)] if args.passes else None,
                 workers=args.workers, run_id=args.run_id, reference=args.reference)
        return 0
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
