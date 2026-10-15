"""The Jev experiment: plan, ask, check, analyse and report, with a health gate before any result.

Phases append to one results folder, so a run can be interrupted and resumed and a finished folder
can have its report rebuilt without a service call. ``plan.json`` binds the folder to the code,
baseline, prompt, passes and analysis plan that started it, and ``provenance.json`` journals every
invocation. The worker that receives a response stores it, with its hash, before it does anything
that could fail, and judging a response cannot stop it being stored (an unexpected error while
reading it becomes the recorded reason), so a paid answer is not lost to a parsing or accounting
failure; only a failing disk can lose one. The credential never reaches a file: the client holds it
and nothing here can read it back.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from iadopt_linking_eval import evaluate_slot, summarise
from iadopt_linking_eval.paired import DEFAULT_RESAMPLES

from .artifacts import file_sha256, write_json, write_text
from .errors import RunError, UsageError
from .experiment import log, save_journal, timed
from .gold import resolve_lab_root
from .health import ensure_checks
from .jev import (
    ARM_CHOICE,
    CHOICE_INSTRUCTIONS,
    MODEL,
    NONE_DESCRIPTION,
    NOUL_INSTRUCTIONS,
    ORDER_SEED,
    PASSES,
    PRIMARY_PASS,
    PROMPT_VERSION,
    JevError,
    PassSpec,
    pass_spec,
    prompt_fingerprint,
)
from .jev_analysis import BOOTSTRAP_SEED, agreement_with, analyse, exact_margins
from .jev_health import (
    answered_ok,
    history_spend,
    jev_checks,
    parse_answer,
    read_answers,
    reported_cost,
    unverified,
)
from .jev_inputs import Baseline, Task, build_tasks, estimate, load_baseline
from .jev_report import (
    confidence_csv,
    outcomes_csv,
    probabilities_csv,
    render_comparison,
    render_report,
)
from .openrouter import FATAL, DecisionsClient, Exchange, api_key_from
from .provenance import (
    JOURNAL_SCHEMA,
    code_identity,
    corpus_identity,
    evidence_code_identity,
    now_utc,
    summed_seconds,
)
from .recorder import JsonlStore, latest

EXPERIMENT_ID = "jev-entity-linking-v1"
# Failures in a row, with no success at all, after which asking more would be waste.
BREAKER = 10
TITLE = "Jev (typesafe/jev-1.13) over the baseline's pooled Wikidata, QUDT and OM candidates"
# OpenRouter's endpoint listing for typesafe/jev-1.13, read on 2026-10-15; input tokens only.
PRICE_PER_INPUT_TOKEN_USD = "0.000000042"
PRICE_SOURCE = "https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints (2026-10-15)"
# The modules that decide what is asked and how it is read; the report-side modules may change and
# the report be rebuilt from the stored responses.
JEV_EVIDENCE_FILES = (
    "iadopt_linking/gold.py", "iadopt_linking/jev.py", "iadopt_linking/jev_inputs.py",
    "iadopt_linking/openrouter.py", "iadopt_linking/population.py", "iadopt_linking/recorder.py",
    "iadopt_linking/rerank.py", "iadopt_linking/score.py", "iadopt_linking_eval/identity.py")
PUBLISHED_JEV = ("responses.jsonl", "plan.json", "provenance.json", "summary.json", "outcomes.csv",
                 "confidence.csv", "probabilities.csv", "report.md", "comparison.md")
PLAN_FIELDS = (
    ("evidence_code_sha256", "different evidence-producing code"),
    ("baseline", "a different baseline"), ("prompt", "a different prompt"),
    ("passes", "different passes"), ("model", "a different model"),
    ("only", "a different set of variables"), ("analysis", "a different analysis plan"),
    ("corpus", "a different corpus release"), ("run_id", "a different run name"))
# Written into plan.json before the first request, so the analysis cannot be chosen after the
# data are seen. Everything not listed as primary is exploratory.
ANALYSIS_PLAN = {
    "primary": {
        "pass": PRIMARY_PASS, "population": "reachable: gold in Wikidata, QUDT or OM",
        "decision": "forced choice: the candidate with the highest probability, none_of_these "
                    "ignored, ties by source order then rank (iadopt_linking_eval.choose)",
        "estimand": "top-1 accuracy of Jev minus the Qwen3 baseline, same slots and candidates",
        "interval": "percentile bootstrap over terms",
        "test": "exact McNemar on slots, descriptive because slots cluster by term"},
    "bootstrap": {"resamples": DEFAULT_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": "0.95"},
    "strict_identity_is_the_headline": True,
    "pass_or_fail_criterion": None,
    "confidence_signals": [
        "jev_selected_probability", "jev_confidence", "jev_margin", "jev_candidate_mass",
        "jev_pass_agreement"],
    "exploratory": [
        "every pass other than the primary", "the pointwise arm", "consensus of the Choice passes",
        "abstention", "position bias", "confidence thresholds and escalation",
        "QUDT/Wikidata equivalence", "subpopulations by gold vocabulary"],
    "prompt_changes_after_first_corpus_request": "none"}


@dataclass(frozen=True)
class JevPaths:
    """The files of one Jev results folder."""

    root: Path

    @property
    def responses(self) -> Path:
        """Return the stored-exchanges JSONL path.

        Args: none.
        Returns: path of responses.jsonl.
        Raises: None.
        Side effects: none.
        """
        return self.root / "responses.jsonl"

    def named(self, name: str) -> Path:
        """Return a file inside the results folder.

        Args: name: file name.
        Returns: its path.
        Raises: None.
        Side effects: none.
        """
        return self.root / name

    def has_evidence(self) -> bool:
        """Say whether the folder already holds responses or a journal.

        Args: none.
        Returns: True when either file exists and is not empty.
        Raises: None.
        Side effects: filesystem checks.
        """
        return any(path.is_file() and path.stat().st_size > 0
                   for path in (self.responses, self.named("provenance.json")))


def jev_evidence_identity() -> str:
    """Hash the modules that decide what Jev is asked and how its answers are read.

    Args: none.
    Returns: a hexadecimal SHA-256 over JEV_EVIDENCE_FILES.
    Raises: OSError if a listed module is missing.
    Side effects: filesystem reads.
    """
    return evidence_code_identity(files=JEV_EVIDENCE_FILES)


def prompt_plan(max_description_chars: int | None) -> dict[str, Any]:
    """Describe the prompt design for the plan.

    Args: max_description_chars: the baseline's description cap.
    Returns: version, fingerprint, the three texts Jev is given, the seed and the cap.
    Raises: None.
    Side effects: none.
    """
    return {"version": PROMPT_VERSION, "fingerprint": prompt_fingerprint(),
            "choice_instructions": CHOICE_INSTRUCTIONS, "none_description": NONE_DESCRIPTION,
            "noul_instructions": NOUL_INSTRUCTIONS, "order_seed": ORDER_SEED,
            "max_description_chars": max_description_chars,
            "shown_per_candidate": ["label", "description"]}


def bind_plan(paths: JevPaths, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Tie a results folder to the run that starts it, or check that this run continues it.

    Args: paths: results folder; expected: the plan this invocation would write.
    Returns: the stored plan.
    Raises: RunError when plan.json cannot be read or differs in any PLAN_FIELDS entry, or when
    the folder holds evidence but no plan.
    Side effects: writes plan.json for a new folder.
    """
    path = paths.named("plan.json")
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunError(f"{path} cannot be read ({exc}); a results folder without its plan "
                           "cannot be continued. Use a new --out folder.") from exc
        for field, phrase in PLAN_FIELDS:
            if previous.get(field) != expected[field]:
                raise RunError(f"{paths.root} was started with {phrase}; resuming would mix "
                               "evidence. Use a new --out folder.")
        return previous
    if paths.has_evidence():
        raise RunError(f"{paths.root} holds evidence but no plan.json; nothing says what "
                       "produced it. Use a new --out folder.")
    write_json(path, dict(expected))
    return dict(expected)


def load_jev_journal(path: Path, *, evidence_exists: bool) -> dict[str, Any]:
    """Read a results folder's journal, or start one.

    Args: path: provenance.json; evidence_exists: whether the folder already holds responses.
    Returns: the journal; a new empty one when the folder is new.
    Raises: RunError for evidence without a journal or a journal that cannot be read.
    Side effects: filesystem read.
    """
    if not path.is_file():
        if evidence_exists:
            raise RunError(f"{path.parent} holds evidence but no {path.name}; without it nothing "
                           "says what produced that evidence. Use a new --out folder.")
        return {"schema": JOURNAL_SCHEMA, "invocations": []}
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"{path} cannot be read ({exc}); it is evidence and is not recreated") from exc
    if (not isinstance(journal, dict) or journal.get("schema") != JOURNAL_SCHEMA
            or not isinstance(journal.get("invocations"), list)):
        raise RunError(f"{path} is not a version {JOURNAL_SCHEMA} journal")
    return journal


def begin_invocation(journal: dict[str, Any], *, evidence_sha256: str,
                     client_settings: Mapping[str, Any], cap_usd: Decimal | None,
                     workers: int) -> dict[str, Any]:
    """Open a journal entry for this invocation.

    Args: journal: the journal; evidence_sha256: hash of the evidence-producing code;
    client_settings: how the client behaves (never a credential); cap_usd: the spending cap;
    workers: parallel requests.
    Returns: the new entry, already appended; its outcome is "running" until the invocation ends.
    Raises: None.
    Side effects: runs git read-only; hashes the source files.
    """
    invocation: dict[str, Any] = {
        "started_at": now_utc(), "finished_at": None, "outcome": "running",
        "code": code_identity(), "evidence_code_sha256": evidence_sha256,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "client": dict(client_settings), "cap_usd": None if cap_usd is None else format(cap_usd, "f"),
        "workers": workers, "seconds": {}, "asking": None}
    journal["invocations"].append(invocation)
    return invocation


def exchange_record(task: Task, exchange: Exchange, parse_error: str | None) -> dict[str, Any]:
    """Form the stored record of one request and what came back.

    Args: task: the request; exchange: what the client ended with; parse_error: why a 200 response
    is not a valid answer, or None.
    Returns: a JSON-ready record. The response text is stored whole beside its hash; the request
    is identified by its hash and the order it showed, and can be regenerated from the plan.
    Raises: None.
    Side effects: none.
    """
    text = exchange.text
    return {
        "slot_id": task.slot.slot_id, "pass": task.spec.name, "arm": task.spec.arm,
        "order_name": task.spec.order, "shown": list(task.shown), "request_sha256": task.sha256,
        "request_chars": task.chars, "sent_at": exchange.sent_at, "status": exchange.status,
        "attempts": exchange.attempts, "retries": list(exchange.retries),
        "latency_s": round(exchange.latency_s, 3), "request_s": round(exchange.request_s, 3),
        "error": exchange.error, "parse_error": parse_error,
        "response_sha256": None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "response_text": text}


def ask_phase(paths: JevPaths, tasks: Sequence[Task], client: Any, *, workers: int,
              cap_usd: Decimal | None, spent_before: Decimal,
              counters: Counter[str] | None = None) -> Counter[str]:
    """Send every request that has no usable answer yet.

    Each worker stores the record of the response it received, under a lock, before anything that
    could fail: judging the response is wrapped, so even an unexpected error while reading it
    leaves the response on disk with that error as the reason. A worker is also the one that
    applies the three stop rules, so none depends on timing: the service refusing the credential or
    the account, total spend passing the cap (rejected responses included, they were paid for),
    and the first BREAKER requests all failing. After a stop no worker sends another request.

    Args: paths: results folder; tasks: all requests of the run; client: anything with
    ``post(body) -> Exchange``; workers: parallel requests; cap_usd: stop when total spend passes
    it, or None; spent_before: what every stored response already cost; counters: filled in place
    as the run goes, so a caller that sees this function raise still knows what it did.
    Returns: counters: requested, answered, failed, retried, not_sent.
    Raises: RunError for a stop rule, for stored responses that no longer match their hash (the
    folder was altered, so nothing is asked), or when requests remain without a usable answer.
    Side effects: network I/O; one appended, fsynced line per finished request.
    """
    store = JsonlStore(paths.responses)
    history = store.read()
    altered = unverified(history)
    if altered:
        raise RunError(f"{len(altered)} stored responses do not match their hash; the folder "
                       "was altered after the responses were stored and is no longer evidence")
    recorded = latest(history, lambda r: (r["slot_id"], r["pass"]))
    pending = [t for t in tasks if not answered_ok(recorded.get(t.key))]
    stats: Counter[str] = counters if counters is not None else Counter()
    stats["requested"] = len(pending)
    if not pending:
        return stats
    started = time.monotonic()
    halt = threading.Event()
    lock = threading.Lock()
    book: dict[str, Any] = {"spent": spent_before, "failed": 0, "answered": 0, "stop": None}

    def send(task: Task) -> tuple[Exchange, str | None] | None:
        """Send one request unless the run has been halted, store the response, apply the rules.

        Args: task: the request.
        Returns: the exchange and why a 200 response is not a valid answer (None if it is), or
        None when the run was halted before this request was sent.
        Raises: OSError if the response cannot be written to disk.
        Side effects: network I/O; appends one record; may halt the run.
        """
        if halt.is_set():
            return None
        try:
            exchange = client.post(task.body)
        except Exception as exc:
            # The client does not raise for network or HTTP problems, so this is something
            # unexpected. Name only its type (a message could carry request details), keep going so
            # that the answers already paid for are still stored, and let the stop rules decide.
            exchange = Exchange(None, None, f"unexpected {type(exc).__name__} while sending", 1, (),
                                0.0, 0.0, now_utc())
        parse_error, cost = None, Decimal(0)
        if exchange.status == 200 and exchange.text is not None:
            cost = reported_cost(exchange.text) or Decimal(0)
            try:
                parse_answer(task, exchange.text)
            except JevError as exc:
                parse_error = str(exc)
            except Exception as exc:
                parse_error = f"unexpected {type(exc).__name__} while reading the response"
        failed = exchange.status != 200 or exchange.error is not None or parse_error is not None
        with lock:
            store.append(exchange_record(task, exchange, parse_error))
            book["spent"] += cost
            book["failed" if failed else "answered"] += 1
            stats["failed" if failed else "answered"] += 1
            stats["retried"] += exchange.attempts > 1
            if book["stop"] is None:
                if exchange.status in FATAL:
                    book["stop"] = (f"the service refused the credential or the account (HTTP "
                                    f"{exchange.status}); no further request can succeed")
                elif cap_usd is not None and book["spent"] > cap_usd:
                    book["stop"] = (f"spend ${format(book['spent'], 'f')} passed the cap "
                                    f"${format(cap_usd, 'f')}")
                elif book["failed"] >= BREAKER and not book["answered"]:
                    book["stop"] = (f"the first {book['failed']} requests all failed (last: "
                                    f"{exchange.error or parse_error}); stopping rather than "
                                    "paying for the rest")
                if book["stop"] is not None:
                    halt.set()
        return exchange, parse_error

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(send, task): task for task in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            sent = future.result()
            task = futures[future]
            if sent is None:
                stats["not_sent"] += 1
                continue
            exchange, parse_error = sent
            if exchange.status != 200 or exchange.error is not None or parse_error is not None:
                log(f"ask FAILED {task.spec.name} {task.slot.slot_id!r}: "
                    f"{exchange.error or parse_error}")
            elif done % 25 == 0 or done == len(pending):
                elapsed = time.monotonic() - started
                log(f"ask {done}/{len(pending)} [{elapsed:.0f}s elapsed, ~"
                    f"{elapsed / done * (len(pending) - done):.0f}s left, "
                    f"${format(book['spent'], 'f')} spent]")
    finally:
        halt.set()
        pool.shutdown(wait=True, cancel_futures=True)
    if book["stop"] is not None:
        raise RunError(book["stop"])
    if stats["failed"]:
        raise RunError(f"{stats['failed']} of {len(pending)} requests have no usable answer; "
                       "rerun the same command to retry only those")
    return stats


def notes_for(summary: Mapping[str, Any], counts: Mapping[str, int], asked: Mapping[str, int],
              margin_ties: tuple[int, int], rejected: Mapping[str, int], cap: int | None,
              has_repeat: bool) -> list[str]:
    """Write the report's notes from this run's numbers.

    Args: summary: the run summary; counts: population counts; asked: slots and empty pools;
    margin_ties: (primary selections that tie with their runner-up, primary selections that have
    a runner-up); rejected: reason -> number of stored responses that failed the validity rule
    and were asked again; cap: the description cap Jev saw; has_repeat: whether an identical
    repeat pass was run.
    Returns: the notes.
    Raises: KeyError for a malformed summary.
    Side effects: none.
    """
    pops = summary["primary"]["populations"]
    unreachable = pops.get("unreachable", {"n": 0})["n"]
    ties, with_runner_up = margin_ties
    shown = "uncapped" if cap is None else f"capped at {cap} characters"
    notes = [
        f"Jev saw, for every candidate, the label and the description ({shown}): the information "
        "the Qwen3 reranker saw, with no vocabulary, identifier, rank or score. The term and the "
        "variable's definition are the baseline's.",
        f"{asked['empty_pools']} slots have no candidate; they were not asked and count as "
        "misses, as in the baseline.",
        f"{counts.get('excluded_label_only', 0)} label-only nodes and "
        f"{counts.get('excluded_non_vocabulary', 0)} non-vocabulary IRIs are outside the "
        "population, as in the baseline.",
        f"{unreachable} slots have their gold in a vocabulary these sources cannot reach; they "
        "count as misses in the \"All controlled-vocabulary slots\" row and are left out of the "
        "primary population (the reachable slots).",
        f"Jev reports probabilities rounded to two decimals: {ties} of {with_runner_up} primary "
        "selections that have a runner-up tie with it and were decided by the baseline's rule "
        "(source order, then rank).",
        "Gold terms are the corpus labels of the gold nodes; for QUDT slots they are often QUDT's "
        "own labels, so QUDT accuracy on these terms may overstate accuracy on LLM-predicted "
        "wording.",
        "The service answers live: another run would not give identical probabilities"
        + (" (see the repeated pass)" if has_repeat else "")
        + ", and a later release of the model is a different model."]
    if rejected:
        reasons = "; ".join(f"{count} x {reason}" for reason, count in sorted(rejected.items()))
        notes.append(
            f"{sum(rejected.values())} stored response(s) were rejected by the validity rule and "
            f"the request was asked again, as the protocol says ({reasons}); rejected responses "
            "stay in responses.jsonl.")
    return notes


def build_summary(baseline: Baseline, results: Mapping[str, Any], analysis: Mapping[str, Any]
                  ) -> dict[str, Any]:
    """Assemble the run summary from the judged passes.

    Args: baseline: the verified baseline; results: pass_results by pass name; analysis: analyse
    output.
    Returns: the summary: the primary pass and the baseline summarised by the same code, and the
    analysis.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    jev = [r["outcome"] for r in results[PRIMARY_PASS]]
    qwen = [evaluate_slot(baseline.records[slot.slot_id]) for slot in baseline.slots]
    primary = summarise(jev)
    # The evaluator subtracts floats; Jev's scores are decimals, so its margins are counted exactly.
    primary["margins"] = exact_margins(results[PRIMARY_PASS])
    return {"scorer_version": primary["scorer_version"], "experiment": EXPERIMENT_ID,
            "primary_pass": PRIMARY_PASS, "primary": primary, "baseline": summarise(qwen),
            "analysis": dict(analysis)}


def build_outputs(paths: JevPaths, baseline: Baseline, tasks: Sequence[Task],
                  plan: Mapping[str, Any], journal: Mapping[str, Any], specs: Sequence[PassSpec],
                  cap_usd: Decimal | None, created_at: str
                  ) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Judge the stored responses and render everything that is published, writing nothing.

    This is the whole of the analysis as a function of stored evidence: a committed folder can be
    rebuilt with it and compared byte for byte with what is there.

    Args: paths: results folder (only responses.jsonl is read); baseline: the verified baseline;
    tasks: all requests; plan: the stored plan; journal: the provenance journal; specs: the passes;
    cap_usd: the spending cap; created_at: the manifest's timestamp, which the report prints.
    Returns: the manifest without its file hashes, the summary, and the text of outcomes.csv,
    confidence.csv, probabilities.csv, report.md and comparison.md.
    Raises: RunError if any check failed; KeyError for a malformed record.
    Side effects: filesystem read of responses.jsonl.
    """
    history = JsonlStore(paths.responses).read()
    recorded = dict(latest(history, lambda r: (r["slot_id"], r["pass"])))
    parsed, problems = read_answers(tasks, recorded)
    inputs = {"slots": len(baseline.slots),
              "candidates": sum(len(baseline.records[s.slot_id]["pool"]) for s in baseline.slots),
              "baseline_files": len(baseline.manifest["files"]),
              "baseline_records": len(baseline.records),
              "baseline_slots": None if plan["only"] else baseline.manifest["population"]["slots"]}
    checks = jev_checks(tasks, recorded, parsed, problems, history=history, inputs=inputs,
                        cap_usd=cap_usd)
    ensure_checks(checks, final=True)
    by_key = {task.key: task for task in tasks}
    results, analysis = analyse(baseline.slots, baseline.records, specs, by_key, parsed,
                                resamples=plan["analysis"]["bootstrap"]["resamples"],
                                seed=plan["analysis"]["bootstrap"]["seed"])
    summary = build_summary(baseline, results, analysis)
    primary = results[PRIMARY_PASS]
    jev = [r["outcome"] for r in primary]
    qwen = [evaluate_slot(baseline.records[s.slot_id]) for s in baseline.slots]
    choice = [results[s.name] for s in specs if s.arm == ARM_CHOICE and s.name != PRIMARY_PASS]
    agreement = agreement_with(primary, choice)
    spec_map = {s.name: s for s in specs}
    paths_by_slot = {s.slot_id: s.source_path for s in baseline.slots}
    margins = [r["margin"] for r in primary if r["margin"] is not None]
    last = journal["invocations"][-1]
    sent = sorted(r["sent_at"] for r in history)
    manifest: dict[str, Any] = {
        "title": TITLE, "experiment": EXPERIMENT_ID, "run_id": plan["run_id"],
        "created_at": created_at, "pilot_variables": plan["only"], "code": last["code"],
        "evidence_code_sha256": plan["evidence_code_sha256"], "corpus": plan["corpus"],
        "baseline": plan["baseline"],
        "model": {"requested": MODEL, "served": sorted({a["model"] for a in parsed.values()}),
                  "providers": sorted({str(a["provider"]) for a in parsed.values()}),
                  "price_per_input_token_usd": plan["model"]["price_per_input_token_usd"],
                  "price_source": plan["model"]["price_source"]},
        "prompt": plan["prompt"], "passes": plan["passes"], "python": last["python"],
        "platform": last["platform"], "invocations": _summaries(journal["invocations"]),
        "timings_s": summed_seconds(journal["invocations"]),
        "evidence_period": {"first_request": sent[0], "last_request": sent[-1]},
        "spend": _spend(parsed, recorded, history, cap_usd, plan),
        "population": {"counts": dict(baseline.counts), "slots": len(baseline.slots)},
        "checks": checks,
        "notes": notes_for(summary, baseline.counts, plan["asked"],
                           (sum(m == 0 for m in margins), len(margins)),
                           Counter(r["parse_error"] for r in history if r.get("parse_error")),
                           baseline.max_description_chars,
                           any(r.arm == ARM_CHOICE and r.name != PRIMARY_PASS and r.order
                               == pass_spec(PRIMARY_PASS).order for r in specs))}
    texts = {
        "outcomes.csv": outcomes_csv(jev, qwen, results, agreement, paths_by_slot, spec_map),
        "confidence.csv": confidence_csv(results, spec_map),
        "probabilities.csv": probabilities_csv(results, spec_map, baseline.records),
        "report.md": render_report(manifest, summary, jev, spec_map),
        "comparison.md": render_comparison(manifest, summary)}
    return manifest, summary, texts


def report_phase(paths: JevPaths, baseline: Baseline, tasks: Sequence[Task],
                 plan: Mapping[str, Any], journal: dict[str, Any], specs: Sequence[PassSpec],
                 cap_usd: Decimal | None) -> dict[str, Any]:
    """Judge the stored responses and write summary, tables, reports and, last, the manifest.

    Args: paths: results folder; baseline: the verified baseline; tasks: all requests; plan: the
    stored plan; journal: the provenance journal; specs: the passes; cap_usd: the spending cap.
    Returns: the manifest.
    Raises: RunError if any check failed; KeyError for a malformed record.
    Side effects: writes provenance.json, summary.json, outcomes.csv, confidence.csv,
    probabilities.csv, report.md, comparison.md and manifest.json.
    """
    manifest, summary, texts = build_outputs(paths, baseline, tasks, plan, journal, specs, cap_usd,
                                             now_utc())
    save_journal(paths, journal)
    write_json(paths.named("summary.json"), summary)
    for name, text in texts.items():
        write_text(paths.named(name), text)
    manifest["files"] = {name: file_sha256(paths.named(name)) for name in PUBLISHED_JEV}
    write_json(paths.named("manifest.json"), manifest)
    return manifest


def _spend(parsed: Mapping[Any, Mapping[str, Any]], recorded: Mapping[Any, Mapping[str, Any]],
           history: Sequence[Mapping[str, Any]], cap_usd: Decimal | None, plan: Mapping[str, Any]
           ) -> dict[str, Any]:
    """Account for what the run cost.

    Args: parsed: parsed answers; recorded: the latest stored exchange per task; history: every
    stored exchange, rejected ones included; cap_usd: the spending cap; plan: the stored plan.
    Returns: answered requests with their tokens, the exact cost reported by the service over every
    stored response (a rejected response was paid for too), the cap, the pre-run estimate, and how
    many answers needed an HTTP retry.
    Raises: KeyError for a malformed answer.
    Side effects: none.
    """
    total, _ = history_spend(history)
    return {"requests": len(parsed),
            "responses_stored": sum(r.get("response_text") is not None for r in history),
            "rejected_responses": sum(r.get("parse_error") is not None for r in history),
            "input_tokens": sum(a["input_tokens"] for a in parsed.values()),
            "output_tokens": sum(a["output_tokens"] for a in parsed.values()),
            "cost_usd": format(total, "f"),
            "cap_usd": None if cap_usd is None else format(cap_usd, "f"),
            "estimate_usd": plan["estimate"]["estimated_cost_usd"],
            "answered_after_retry": sum(recorded[k]["attempts"] > 1 for k in parsed),
            "failed_attempts_repeated": sum(len(recorded[k]["retries"]) for k in parsed)}


def _summaries(invocations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Condense the journal for the manifest.

    Args: invocations: journal invocations.
    Returns: one entry per invocation: times, outcome, what it asked, where it ran, which code.
    Raises: KeyError for a malformed entry.
    Side effects: none.
    """
    return [{"started_at": i["started_at"], "finished_at": i["finished_at"],
             "outcome": i["outcome"], "seconds": {k: round(v, 1) for k, v in i["seconds"].items()},
             "asking": i["asking"], "workers": i["workers"], "cap_usd": i["cap_usd"],
             "python": i["python"], "platform": i["platform"],
             "source_sha256": i["code"]["source_sha256"], "git_dirty": i["code"]["git_dirty"]}
            for i in invocations]


def _parse_cap(max_cost_usd: str | Decimal | None) -> Decimal | None:
    """Read the spending cap.

    Args: max_cost_usd: a decimal string or Decimal, or None for no cap.
    Returns: the cap.
    Raises: RunError for a value that is not a positive decimal.
    Side effects: none.
    """
    if max_cost_usd is None:
        return None
    try:
        cap = Decimal(str(max_cost_usd))
    except InvalidOperation:
        raise RunError(f"the cost cap {max_cost_usd!r} is not a number") from None
    if not cap.is_finite() or cap <= 0:
        raise RunError("the cost cap must be a positive amount")
    return cap


def run_jev(out_dir: Path | str, *, baseline_dir: Path | str, max_cost_usd: str | Decimal | None,
            env_file: str | None = None, lab_root: Path | str | None = None,
            only: Sequence[str] | None = None, passes: Sequence[str] | None = None,
            workers: int = 4, run_id: str | None = None, resamples: int = DEFAULT_RESAMPLES,
            client_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Run the whole Jev experiment: verify, plan, ask, check, analyse, report.

    Args: out_dir: results folder; baseline_dir: the finished baseline to compare against and take
    candidates from; max_cost_usd: spending cap in USD, required in practice (a billed call needs
    an explicit limit); env_file: credential file defining OPENROUTER_API_KEY, used only when
    requests are pending; lab_root: iadopt-lab root; only: variable stems for a pilot; passes:
    pass names (default all); workers: parallel requests; run_id: name recorded in the manifest
    and sent as the OpenRouter session id; resamples: bootstrap replicates; client_factory:
    injectable for tests.
    Returns: the manifest.
    Raises: RunError when the baseline is not usable, the plan differs, the estimate exceeds the
    cap, a request cannot be answered or a check fails; UsageError for an unknown variable;
    KeyError for an unknown pass name.
    Side effects: network I/O, files in out_dir; removes an earlier manifest.json at once, so an
    invocation that stops leaves no manifest behind.
    """
    paths = JevPaths(Path(out_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    root = resolve_lab_root(lab_root)
    baseline = load_baseline(baseline_dir, lab_root=root, only=only)
    specs = [pass_spec(name) for name in passes] if passes else list(PASSES)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise UsageError(f"a pass can be run once; the list has duplicates: {names}")
    if PRIMARY_PASS not in names:
        raise UsageError(f"the primary pass {PRIMARY_PASS!r} must be among the passes")
    name = run_id or paths.root.name
    tasks, asked = build_tasks(baseline, specs, session_id=f"iadopt-linking/{name}")
    cap = _parse_cap(max_cost_usd)
    evidence_sha256 = jev_evidence_identity()
    plan = bind_plan(paths, {
        "experiment": EXPERIMENT_ID, "run_id": name, "evidence_code_sha256": evidence_sha256,
        "baseline": baseline.identity, "corpus": corpus_identity(root),
        "model": {"requested": MODEL, "price_per_input_token_usd": PRICE_PER_INPUT_TOKEN_USD,
                  "price_source": PRICE_SOURCE},
        "prompt": prompt_plan(baseline.max_description_chars),
        "passes": [[s.name, s.arm, s.order] for s in specs],
        "only": sorted(only) if only else None,
        "analysis": {**ANALYSIS_PLAN, "bootstrap": {**ANALYSIS_PLAN["bootstrap"],
                                                    "resamples": resamples}},
        "asked": asked, "tasks": len(tasks),
        "estimate": estimate(tasks, PRICE_PER_INPUT_TOKEN_USD)})
    journal = load_jev_journal(paths.named("provenance.json"), evidence_exists=paths.has_evidence())
    paths.named("manifest.json").unlink(missing_ok=True)
    stored = JsonlStore(paths.responses).read()
    recorded = latest(stored, lambda r: (r["slot_id"], r["pass"]))
    spent_before, _ = history_spend(stored)
    waiting = [t for t in tasks if not answered_ok(recorded.get(t.key))]
    pending = len(waiting)
    cost_estimate = Decimal(estimate(waiting, PRICE_PER_INPUT_TOKEN_USD)["estimated_cost_usd"])
    log(f"population: {len(baseline.slots)} slots, {asked['empty_pools']} without candidates; "
        f"{len(tasks)} requests in {len(specs)} passes, {pending} pending; estimate for those "
        f"${format(cost_estimate, 'f')} (upper bound), spent so far ${format(spent_before, 'f')}, "
        f"cap {'none' if cap is None else '$' + format(cap, 'f')}")
    if pending and cap is None:
        raise RunError("a billed run needs an explicit spending cap (--max-cost-usd)")
    if pending and cap is not None and spent_before + cost_estimate > cap:
        raise RunError(f"the upper-bound estimate ${format(cost_estimate, 'f')} plus the "
                       f"${format(spent_before, 'f')} already spent exceeds the cap "
                       f"${format(cap, 'f')}; raise --max-cost-usd or ask for fewer passes")
    client = None
    if pending:
        client = client_factory() if client_factory else DecisionsClient(
            api_key_from(env_file or _default_env(root), os.environ))
    invocation = begin_invocation(
        journal, evidence_sha256=evidence_sha256, workers=workers, cap_usd=cap,
        client_settings=client.settings() if client is not None and hasattr(client, "settings")
        else {})
    save_journal(paths, journal)
    counters: Counter[str] = Counter()
    try:
        try:
            with timed(invocation["seconds"], "asking"):
                if client is not None:
                    ask_phase(paths, tasks, client, workers=workers, cap_usd=cap,
                              spent_before=spent_before, counters=counters)
        finally:
            invocation["asking"] = dict(sorted(counters.items())) or None
        save_journal(paths, journal)
        invocation["outcome"], invocation["finished_at"] = "completed", now_utc()
        manifest = report_phase(paths, baseline, tasks, plan, journal, specs, cap)
        log(f"report written to {paths.named('report.md')}")
        return manifest
    except BaseException as exc:
        invocation["outcome"] = f"stopped: {type(exc).__name__}: {exc}"[:300]
        invocation["finished_at"] = now_utc()
        save_journal(paths, journal)
        raise


def _default_env(lab_root: Path) -> str | None:
    """Find the repository's credential file when none was named.

    Args: lab_root: iadopt-lab root; its parent is the repository root.
    Returns: the path of ``.env`` there, or None when there is none.
    Raises: None.
    Side effects: a filesystem check.
    """
    candidate = lab_root.parent / ".env"
    return str(candidate) if candidate.is_file() else None
