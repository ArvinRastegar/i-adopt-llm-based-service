"""The baseline experiment: plan, retrieve, score, check, report, with a health gate per phase.

Phases append to evidence files in one results folder, so any phase can be rerun and resumes where
it stopped. ``plan.json`` binds the folder to the code, corpus, population and configuration that
started it, and ``provenance.json`` journals every invocation. A failed check stops the run with a
named reason before the next phase; a report is written only when every check that must run passed.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iadopt_linking_eval import evaluate_slot, summarise

from .artifacts import file_sha256, write_json, write_text
from .devices import check_slots, device_check, device_passed, reference_device
from .errors import RunError, UsageError
from .gold import GoldVariable, load_gold_links, resolve_lab_root
from .health import (
    SOURCE_NAMES,
    build_checks,
    ensure_checks,
    evidence_period,
    notes_for,
    retrieval_summary,
)
from .http import PoliteClient
from .population import Slot, build_population
from .provenance import (
    EVIDENCE_FILES,
    begin_invocation,
    corpus_identity,
    evidence_code_identity,
    invocation_summaries,
    load_journal,
    merge_releases,
    model_summary,
    now_utc,
    probe_releases,
    summed_seconds,
)
from .recorder import JsonlStore, latest
from .report import outcomes_csv, render_report
from .rerank import DEFAULT_MAX_DESCRIPTION_CHARS, Scorer
from .retrieve import is_ok, retrieve
from .score import score_all
from .sources import RESULT_LIMIT, SOURCES

__all__ = [
    "EVIDENCE_FILES", "RunError", "UsageError", "evidence_code_identity", "load_population",
    "query_keys", "run",
]

USER_AGENT = "iadopt-linking/0.1 (I-ADOPT corpus entity-linking benchmark; research use)"
TITLE = "Baseline: Qwen3-Reranker-0.6B over pooled Wikidata, QUDT and OM candidates"
# What a published results folder contains; the manifest carries a hash of each.
PUBLISHED = ("candidates.jsonl", "scores.jsonl", "plan.json", "provenance.json", "summary.json",
             "outcomes.csv", "report.md")
# Compared on every resume: a results folder continues only the run that started it.
PLAN_FIELDS = (("evidence_code_sha256", "different evidence-producing code"),
               ("only", "a different set of variables"), ("config", "a different configuration"),
               ("corpus", "a different corpus release"), ("query_keys", "a different set of queries"))


@dataclass(frozen=True)
class Paths:
    """The files of one results folder."""

    root: Path

    @property
    def candidates(self) -> Path:
        """Return the retrieval JSONL path.

        Args: none.
        Returns: path of candidates.jsonl.
        Raises: None.
        Side effects: none.
        """
        return self.root / "candidates.jsonl"

    @property
    def scores(self) -> Path:
        """Return the scores JSONL path.

        Args: none.
        Returns: path of scores.jsonl.
        Raises: None.
        Side effects: none.
        """
        return self.root / "scores.jsonl"

    def named(self, name: str) -> Path:
        """Return a file inside the results folder.

        Args: name: file name.
        Returns: its path.
        Raises: None.
        Side effects: none.
        """
        return self.root / name

    def has_evidence(self) -> bool:
        """Say whether the folder already holds retrieval, scoring or provenance data.

        Args: none.
        Returns: True when any of those files exists and is not empty.
        Raises: None.
        Side effects: filesystem checks.
        """
        return any(path.is_file() and path.stat().st_size > 0
                   for path in (self.candidates, self.scores, self.named("provenance.json")))


def log(message: str) -> None:
    """Print a timestamped progress line immediately.

    Args: message: text to print.
    Returns: None.
    Raises: None.
    Side effects: writes to stdout and flushes.
    """
    print(f"{now_utc()[11:19]} {message}", flush=True)


@contextmanager
def timed(seconds: dict[str, float], phase: str) -> Iterator[None]:
    """Add the time a block takes to a phase's total, even when the block fails.

    Args: seconds: phase -> seconds; phase: the phase's name.
    Returns: a context manager.
    Raises: whatever the block raises.
    Side effects: updates ``seconds``.
    """
    started = time.monotonic()
    try:
        yield
    finally:
        seconds[phase] = seconds.get(phase, 0.0) + time.monotonic() - started


def load_population(lab_root: Path | str | None = None, only: Sequence[str] | None = None
                    ) -> tuple[tuple[GoldVariable, ...], tuple[Slot, ...], dict[str, int]]:
    """Load the gold and select the scored slots, optionally for a few variables.

    Args: lab_root: iadopt-lab root; only: variable file stems to keep (a pilot).
    Returns: the variables kept, their slots and the population counts, which include
    ``gold_links``: the links as the gold lists them, counted apart from the population loop.
    Raises: UsageError for a stem that matches no variable.
    Side effects: filesystem reads.
    """
    variables = load_gold_links(lab_root)
    if only:
        wanted = set(only)
        variables = tuple(v for v in variables if Path(v.source_path).stem in wanted)
        unknown = wanted - {Path(v.source_path).stem for v in variables}
        if unknown:
            raise UsageError(f"No such variable file stem: {sorted(unknown)}")
    slots, counts = build_population(variables)
    counts["gold_links"] = sum(len(variable.links) for variable in variables)
    return variables, slots, dict(sorted(counts.items()))


def query_keys(slots: Sequence[Slot]) -> list[str]:
    """List the distinct retrieval queries a population needs.

    Args: slots: the population.
    Returns: sorted distinct query keys.
    Raises: None.
    Side effects: none.
    """
    return sorted({slot.query_key for slot in slots})


def run_config(max_description_chars: int | None) -> dict[str, Any]:
    """Describe the configuration that shapes a run's evidence.

    Args: max_description_chars: description cap, or None for no cap.
    Returns: result limit, sources, cap and the reranker's query and document templates.
    Raises: None.
    Side effects: none.
    """
    return {"result_limit_per_source": RESULT_LIMIT, "sources": list(SOURCE_NAMES),
            "max_description_chars": max_description_chars,
            "rerank_query": 'Definition of "{term}" in context: "{definition}"',
            "rerank_document": 'label: "{label}", description: "{description}"'}


def bind_to_run(paths: Paths, *, evidence_sha256: str, only: Sequence[str] | None,
                keys: Sequence[str], counts: Mapping[str, int], config: Mapping[str, Any],
                corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Tie a results folder to the run that starts it, or check that this run continues it.

    Args: paths: results folder; evidence_sha256: hash of the evidence-producing code; only:
    pilot filter or None; keys: query keys; counts: population counts; config: run_config
    output; corpus: corpus_identity output.
    Returns: the plan, as stored in plan.json.
    Raises: RunError when plan.json cannot be read or differs in code, variables, configuration,
    corpus or queries, or when the folder holds evidence but no plan.
    Side effects: writes plan.json for a new folder.
    """
    expected: dict[str, Any] = {
        "evidence_code_sha256": evidence_sha256, "only": sorted(only) if only else None,
        "config": dict(config), "corpus": dict(corpus), "query_keys": list(keys)}
    path = paths.named("plan.json")
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunError(f"{path} cannot be read ({exc}); a results folder without its plan "
                           "cannot be continued. Use a new --out folder.") from exc
        for field, phrase in PLAN_FIELDS:
            if previous.get(field) != expected[field]:
                shown = (f" ({str(previous.get(field))[:12]} against {evidence_sha256[:12]})"
                         if field == "evidence_code_sha256" else "")
                raise RunError(f"{paths.root} was started with {phrase}{shown}; resuming would "
                               "mix evidence. Use a new --out folder.")
        return previous
    if paths.has_evidence():
        raise RunError(f"{paths.root} holds evidence but no plan.json; nothing says what "
                       "produced it. Use a new --out folder.")
    plan = {**expected, "counts": dict(counts), "slots": counts.get("slots", 0)}
    write_json(path, plan)
    return plan


def save_journal(paths: Paths, journal: Mapping[str, Any]) -> None:
    """Write the provenance journal.

    Args: paths: results folder; journal: the journal.
    Returns: None.
    Raises: TypeError or ValueError for a value JSON cannot hold.
    Side effects: replaces provenance.json atomically.
    """
    write_json(paths.named("provenance.json"), journal)


def retrieval_phase(paths: Paths, keys: Sequence[str], *, client: PoliteClient,
                    max_minutes: float, sources: Mapping[str, Any] = SOURCES
                    ) -> tuple[dict[tuple[str, str], Mapping[str, Any]], Counter[str]]:
    """Retrieve every (source, key) and prove none failed.

    Args: paths: results folder; keys: query keys; client: HTTP client; max_minutes: wall-clock
    budget for this call; sources: name -> source.
    Returns: the successful retrieval records keyed by (source, key), and the pass's counters.
    Raises: RetrievalTimeout when the budget runs out; RunError when any request still has no
    successful record. Rerunning resumes and retries only the failures.
    Side effects: network I/O; appends to candidates.jsonl.
    """
    store = JsonlStore(paths.candidates)
    started = time.monotonic()

    def show(done: int, total: int, record: Mapping[str, Any]) -> None:
        """Print retrieval progress.

        Args: done: requests finished; total: requests pending; record: the one just stored.
        Returns: None.
        Raises: None.
        Side effects: writes to stdout.
        """
        failed = not is_ok(record)
        if failed or done % 20 == 0 or done == total:
            elapsed = time.monotonic() - started
            eta = elapsed / done * (total - done)
            size = "FAILED " + str(record["error"] or record["parse_error"]) if failed else (
                f"{len(record['candidates'])} candidates")
            log(f"retrieve {done}/{total} {record['source']} {record['query_key']!r}: {size} "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left]")

    stats = retrieve(keys, sources, client, store, deadline=started + max_minutes * 60,
                     progress=show)
    log(f"retrieval pass done: {dict(sorted(stats.items()))}")
    recorded = latest(store.read(), lambda r: (r["source"], r["query_key"]))
    wanted = [(name, key) for key in keys for name in sources]
    bad = [pair for pair in wanted if not is_ok(recorded.get(pair))]
    if bad:
        raise RunError(f"{len(bad)} retrievals failed, e.g. {bad[:5]}; rerun the same command "
                       "to retry only those")
    return {pair: recorded[pair] for pair in wanted}, stats


class LazyScorer:
    """Load the reranker on first use, so a run that only rebuilds its report never loads it."""

    def __init__(self, factory: Callable[[str | None], Any], device: str | None) -> None:
        """Remember how to build the scorer.

        Args: factory: device -> scorer; device: the device to ask for, or None for the default.
        Returns: None.
        Raises: None.
        Side effects: none until the scorer is first used.
        """
        self._factory, self._device, self._scorer = factory, device, None

    @property
    def loaded(self) -> bool:
        """Say whether the model has been loaded.

        Args: none.
        Returns: True once the scorer exists.
        Raises: None.
        Side effects: none.
        """
        return self._scorer is not None

    def _get(self) -> Any:
        """Build the scorer if it does not exist yet.

        Args: none.
        Returns: the scorer.
        Raises: whatever the factory raises, for example a model missing from the local cache.
        Side effects: may load the model.
        """
        if self._scorer is None:
            self._scorer = self._factory(self._device)
        return self._scorer

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score documents with the underlying scorer.

        Args: query: plain query text; documents: plain document texts.
        Returns: one score per document.
        Raises: whatever the scorer raises.
        Side effects: may load the model; model inference.
        """
        return self._get().score(query, documents)

    def describe(self) -> dict[str, Any]:
        """Identify the model and runtime.

        Args: none.
        Returns: what the scorer reports about itself; empty when it reports nothing.
        Raises: whatever loading raises.
        Side effects: may load the model.
        """
        scorer = self._get()
        return scorer.describe() if hasattr(scorer, "describe") else {}


def default_scorer(device: str | None) -> Scorer:
    """Build the baseline reranker from the local model cache.

    Args: device: "mps", "cpu" or None to prefer mps.
    Returns: the Qwen3 reranker.
    Raises: the loader's error when the weights are not in the local cache.
    Side effects: loads about 2.4 GB of weights.
    """
    from .rerank import Qwen3Reranker
    return Qwen3Reranker(device)


def scoring_phase(paths: Paths, slots: Sequence[Slot], retrieved: Mapping[Any, Any],
                  scorer: Scorer, *, max_description_chars: int | None
                  ) -> tuple[dict[str, dict], Counter[str]]:
    """Score every slot and prove none is missing.

    Args: paths: results folder; slots: the population; retrieved: successful retrievals;
    scorer: the reranker; max_description_chars: description cap.
    Returns: the scores records keyed by slot id, and the pass's counters.
    Raises: RunError when a slot has no record afterwards; PoolError or ScoreError from scoring.
    Side effects: model inference; appends to scores.jsonl.
    """
    store = JsonlStore(paths.scores)
    started = time.monotonic()

    def show(done: int, total: int) -> None:
        """Print scoring progress.

        Args: done: slots handled; total: slots in the population.
        Returns: None.
        Raises: None.
        Side effects: writes to stdout.
        """
        if done % 25 == 0 or done == total:
            elapsed = time.monotonic() - started
            log(f"score {done}/{total} slots [{elapsed:.0f}s elapsed, "
                f"~{elapsed / done * (total - done):.0f}s left]")

    stats = score_all(slots, retrieved, scorer, store, sources=SOURCE_NAMES,
                      max_description_chars=max_description_chars, progress=show)
    log(f"scoring pass done: {dict(sorted(stats.items()))}")
    records = latest(store.read(), lambda r: r["slot_id"])
    missing = [slot.slot_id for slot in slots if slot.slot_id not in records]
    if missing:
        raise RunError(f"{len(missing)} slots have no scores, e.g. {missing[:5]}")
    return {slot.slot_id: records[slot.slot_id] for slot in slots}, stats


def device_phase(paths: Paths, journal: dict[str, Any], records: Mapping[str, Mapping[str, Any]],
                 primary: LazyScorer, make_scorer: Callable[[str | None], Any], *,
                 check_devices: bool, recheck_devices: bool) -> dict[str, Any] | None:
    """Cross-check the scores on the other device, or reuse a passed check of these exact scores.

    Args: paths: results folder; journal: the journal; records: scores records by slot id;
    primary: the scorer that made them; make_scorer: device -> scorer; check_devices: False to
    skip a new check; recheck_devices: True to ignore a stored result.
    Returns: the check result, or None when none exists and none was asked for.
    Raises: ScoreError if a scorer returns an impossible score.
    Side effects: model inference on both devices; updates the journal.
    """
    scores_sha256 = file_sha256(paths.scores)
    stored = journal.get("device_check")
    if (stored and not recheck_devices and stored.get("passed")
            and stored.get("scores_sha256") == scores_sha256):
        log("device check: reusing the stored result for these exact scores")
        return stored
    if not check_devices:
        return None
    primary_device = primary.describe().get("device")
    other = reference_device(primary_device)
    chosen, near_ties = check_slots(records)
    log(f"device check: rescoring {len(chosen)} slots ({near_ties} near-ties) on {other}")
    started = time.monotonic()

    def show(done: int, total: int) -> None:
        """Print device-check progress.

        Args: done: slots rescored; total: slots to rescore.
        Returns: None.
        Raises: None.
        Side effects: writes to stdout.
        """
        if done % 10 == 0 or done == total:
            elapsed = time.monotonic() - started
            log(f"device check {done}/{total} slots [{elapsed:.0f}s elapsed, "
                f"~{elapsed / done * (total - done):.0f}s left]")

    result = device_check(records, primary, LazyScorer(make_scorer, other),
                          primary_device=primary_device, reference_device_name=other,
                          progress=show)
    result["scores_sha256"] = scores_sha256
    result["passed"] = device_passed(result)
    journal["device_check"] = result
    save_journal(paths, journal)
    return result


def report_phase(paths: Paths, slots: Sequence[Slot], counts: Mapping[str, int],
                 keys: Sequence[str], *, plan: Mapping[str, Any], journal: dict[str, Any],
                 run_id: str, device: Mapping[str, Any] | None) -> dict[str, Any]:
    """Evaluate the stored evidence and write summary, outcomes, report and manifest.

    Args: paths: results folder; slots: population; counts: population counts; keys: query keys;
    plan: the stored plan; journal: the provenance journal; run_id: name of the run; device:
    device cross-check or None.
    Returns: the manifest.
    Raises: RunError if any check failed or a required check did not run; KeyError for a
    malformed record.
    Side effects: writes provenance.json, summary.json, outcomes.csv, report.md and, last,
    manifest.json.
    """
    history = JsonlStore(paths.candidates).read()
    scored = latest(JsonlStore(paths.scores).read(), lambda r: r["slot_id"])
    records = {slot.slot_id: scored[slot.slot_id] for slot in slots if slot.slot_id in scored}
    invocations = journal["invocations"]
    releases = merge_releases(invocations)
    cap = plan["config"]["max_description_chars"]
    checks = build_checks(slots=slots, counts=counts, keys=keys, max_description_chars=cap,
                          history=history, releases=releases, records=records, device=device)
    ensure_checks(checks, final=True)
    retrieval = retrieval_summary(history, keys)
    period = evidence_period(history, keys)
    outcomes = [evaluate_slot(records[slot.slot_id]) for slot in slots]
    summary = summarise(outcomes)
    last = invocations[-1]
    manifest: dict[str, Any] = {
        "title": TITLE, "run_id": run_id, "created_at": now_utc(),
        "pilot_variables": plan["only"], "code": last["code"],
        "evidence_code_sha256": plan["evidence_code_sha256"], "corpus": plan["corpus"],
        "model": model_summary(records, invocations), "python": last["python"],
        "platform": last["platform"], "invocations": invocation_summaries(invocations),
        "timings_s": summed_seconds(invocations), "evidence_period": period,
        "config": {**plan["config"], "http": last["http"]},
        "population": {"counts": dict(counts), "slots": len(slots), "query_keys": len(keys)},
        "source_releases": releases, "retrieval": retrieval, "device_check": device,
        "checks": checks,
        "notes": notes_for(counts=counts, retrieval=retrieval, summary=summary, device=device,
                           releases=releases, period=period, max_description_chars=cap)}
    save_journal(paths, journal)
    write_json(paths.named("summary.json"), summary)
    write_text(paths.named("outcomes.csv"), outcomes_csv(outcomes))
    write_text(paths.named("report.md"), render_report(manifest, summary, outcomes))
    manifest["files"] = {name: file_sha256(paths.named(name)) for name in PUBLISHED}
    write_json(paths.named("manifest.json"), manifest)
    return manifest


def _execute(paths: Paths, journal: dict[str, Any], invocation: dict[str, Any], *,
             slots: Sequence[Slot], counts: Mapping[str, int], keys: Sequence[str],
             plan: Mapping[str, Any], client: PoliteClient, primary: LazyScorer,
             make_scorer: Callable[[str | None], Any], max_retrieval_minutes: float,
             check_devices: bool, recheck_devices: bool, run_id: str) -> dict[str, Any]:
    """Run the phases of one invocation, each behind its health checks.

    Args: paths: results folder; journal and invocation: the provenance journal and this
    invocation's entry; slots, counts, keys: the population; plan: the stored plan; client: HTTP
    client; primary: the reranker; make_scorer: device -> scorer; max_retrieval_minutes:
    retrieval budget; check_devices and recheck_devices: device cross-check options; run_id:
    name recorded in the manifest.
    Returns: the manifest.
    Raises: RunError, RetrievalTimeout, PoolError or ScoreError; nothing is reported after one.
    Side effects: network I/O, model inference, files in the results folder.
    """
    seconds = invocation["seconds"]
    cap = plan["config"]["max_description_chars"]
    common: dict[str, Any] = {"slots": slots, "counts": counts, "keys": keys,
                              "max_description_chars": cap}
    with timed(seconds, "retrieval"):
        retrieved, stats = retrieval_phase(paths, keys, client=client, max_minutes=max_retrieval_minutes)
        invocation["retrieval"] = dict(sorted(stats.items()))
        merged = merge_releases(journal["invocations"])
        if stats["requested"] or any(merged[name]["latest"] is None for name in ("om", "qudt")):
            invocation["source_releases"] = probe_releases(client)
    save_journal(paths, journal)
    history = JsonlStore(paths.candidates).read()
    releases = merge_releases(journal["invocations"])
    ensure_checks(build_checks(**common, history=history, releases=releases), final=False)
    with timed(seconds, "scoring"):
        records, stats = scoring_phase(paths, slots, retrieved, primary, max_description_chars=cap)
    invocation["scoring"] = dict(sorted(stats.items()))
    ensure_checks(build_checks(**common, history=history, releases=releases, records=records),
                  final=False)
    with timed(seconds, "device_check"):
        device = device_phase(paths, journal, records, primary, make_scorer,
                              check_devices=check_devices, recheck_devices=recheck_devices)
    invocation["model"] = primary.describe() if primary.loaded else None
    ensure_checks(build_checks(**common, history=history, releases=releases, records=records,
                               device=device), final=False)
    invocation["outcome"], invocation["finished_at"] = "completed", now_utc()
    manifest = report_phase(paths, slots, counts, keys, plan=plan, journal=journal,
                            run_id=run_id, device=device)
    log(f"report written to {paths.named('report.md')}")
    return manifest


def run(out_dir: Path | str, *, lab_root: Path | str | None = None,
        only: Sequence[str] | None = None, max_retrieval_minutes: float = 30.0,
        max_description_chars: int | None = DEFAULT_MAX_DESCRIPTION_CHARS,
        device: str | None = None, check_devices: bool = True, recheck_devices: bool = False,
        client_factory: Callable[[], PoliteClient] | None = None,
        scorer_factory: Callable[[str | None], Any] | None = None,
        run_id: str | None = None) -> dict[str, Any]:
    """Run the whole baseline: retrieve, score, check, report.

    Args: out_dir: results folder; lab_root: iadopt-lab root; only: variable stems for a pilot;
    max_retrieval_minutes: retrieval budget; max_description_chars: description cap; device:
    reranker device; check_devices: False to skip a new cross-device check; recheck_devices:
    True to repeat a stored one; client_factory and scorer_factory: injectable for tests;
    run_id: name recorded in the manifest.
    Returns: the manifest.
    Raises: RunError, RetrievalTimeout, PoolError, ScoreError, StoreError or UsageError; nothing
    is reported after one, and the journal records why the invocation stopped.
    Side effects: network I/O, model inference, files in out_dir; removes an earlier
    manifest.json at once, so an invocation that stops leaves no manifest behind.
    """
    paths = Paths(Path(out_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    root = resolve_lab_root(lab_root)
    _, slots, counts = load_population(root, only)
    keys = query_keys(slots)
    evidence_sha256 = evidence_code_identity()
    plan = bind_to_run(paths, evidence_sha256=evidence_sha256, only=only, keys=keys, counts=counts,
                       config=run_config(max_description_chars), corpus=corpus_identity(root))
    journal = load_journal(paths.named("provenance.json"), evidence_exists=paths.has_evidence())
    # A manifest says "finished and checked"; a folder being worked on must not claim it.
    paths.named("manifest.json").unlink(missing_ok=True)
    log(f"population: {len(slots)} slots from {counts['links']} linked nodes "
        f"({counts.get('excluded_label_only', 0)} label only, "
        f"{counts.get('excluded_non_vocabulary', 0)} non-vocabulary IRIs left out); "
        f"{len(keys)} queries")
    client = client_factory() if client_factory else PoliteClient(USER_AGENT)
    invocation = begin_invocation(
        journal, evidence_sha256=evidence_sha256,
        http_settings=client.settings() if hasattr(client, "settings") else {})
    save_journal(paths, journal)
    make_scorer = scorer_factory or default_scorer
    try:
        return _execute(
            paths, journal, invocation, slots=slots, counts=counts, keys=keys, plan=plan,
            client=client, primary=LazyScorer(make_scorer, device), make_scorer=make_scorer,
            max_retrieval_minutes=max_retrieval_minutes, check_devices=check_devices,
            recheck_devices=recheck_devices, run_id=run_id or paths.root.name)
    except BaseException as exc:
        invocation["outcome"] = f"stopped: {type(exc).__name__}: {exc}"[:300]
        invocation["finished_at"] = now_utc()
        save_journal(paths, journal)
        raise
    finally:
        client.close()
