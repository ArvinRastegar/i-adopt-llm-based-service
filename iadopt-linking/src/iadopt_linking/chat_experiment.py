"""A PSNC chat model as a scorer on the baseline's frozen candidates (DECISIONS L-016 and L-017).

This module is the second ``Experiment`` of the engine in ``jev_experiment``: the same slots, the
same candidates, the same candidate orders and the same evaluator as the Jev run, with a chat model
asked the same question as a numbered list and its answer read from the log-probabilities of its
first token (``chat_choice``). It differs from the Jev experiment only where the service does:

- the service is not billed to the owner, so a run is limited by a number of requests, not money;
- a chat model has no vendor-documented confidence value, so the "confidence" signal is the
  probability of its own choice, and the formula check of Jev's documented value is not made;
- there is no pointwise arm (one yes/no per candidate would need one request per candidate);
- the run can be compared with a finished Jev run on the same slots (``reference``), the second
  pre-specified comparison, from that run's published and hash-verified outcomes.

Requests are sent one at a time by default: answers depend slightly on what else the server is
processing (identical requests sent concurrently gave different probabilities in development,
identical requests sent one after another did not).
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from iadopt_linking_eval.metrics import _count
from iadopt_linking_eval.paired import DEFAULT_RESAMPLES

from .artifacts import file_sha256
from .chat import credentials_from
from .chat_choice import (
    CHAT_PASSES,
    MIN_LABEL_MASS,
    MODEL,
    REASONING_OFF,
    TOP_LOGPROBS,
    build_tasks,
    estimate,
    parse_chat_choice,
    prompt_plan,
    reported_cost,
)
from .chat_redact import RedactingChatClient
from .errors import RunError
from .health import check
from .jev import PRIMARY_PASS
from .jev_analysis import BOOTSTRAP_SEED, POPULATIONS, compare
from .jev_experiment import (
    JEV_EVIDENCE_FILES,
    PLAN_FIELDS,
    Experiment,
    _default_env,
    run_experiment,
)
from .jev_health import (
    CHECK_ANSWERED,
    CHECK_HASHES,
    CHECK_INPUTS,
    CHECK_MODEL,
    CHECK_PARSED,
    CHECK_PRIMARY,
    CHECK_REQUESTS,
    unverified,
)
from .jev_inputs import Baseline, Task
from .jev_report import Voice, _difference, _p, _terms
from .openrouter import FATAL
from .report import _table, ratio

EXPERIMENT_ID = "psnc-entity-linking-v1"
REFERENCE_EXPERIMENT = "jev-entity-linking-v1"
# The Jev evidence files decide the slots, the candidates and their orders, and how a baseline is
# verified; the two chat modules decide what is asked and how it is read. The Jev client
# (`openrouter`) and parser (`jev`) are in the list because the chat modules share their request
# record and ordering code.
PSNC_EVIDENCE_FILES = (*JEV_EVIDENCE_FILES, "iadopt_linking/chat.py", "iadopt_linking/chat_choice.py")
PLAN_FIELDS_WITH_REFERENCE = (*PLAN_FIELDS, ("reference", "a different reference run"))
CHECK_USAGE = "token usage fully accounted"
REQUIRED_CHAT_CHECKS = (CHECK_INPUTS, CHECK_ANSWERED, CHECK_REQUESTS, CHECK_HASHES, CHECK_PARSED,
                        CHECK_MODEL, CHECK_USAGE, CHECK_PRIMARY)
SIGNALS = ["llm_selected_probability", "llm_confidence", "llm_margin", "llm_candidate_mass",
           "llm_pass_agreement"]


def analysis_plan() -> dict[str, Any]:
    """Write the analysis down before the data exist.

    Args: none.
    Returns: the plan stored in plan.json before the first request: the primary result, the
    pre-specified comparison with Jev, the bootstrap, the confidence signals and what is
    exploratory.
    Raises: None.
    Side effects: none.
    """
    return {
        "primary": {
            "pass": PRIMARY_PASS, "population": "reachable: gold in Wikidata, QUDT or OM",
            "decision": "forced choice: the candidate whose option label has the highest "
                        "probability, none_of_these ignored, ties by source order then rank "
                        "(iadopt_linking_eval.choose)",
            "estimand": "top-1 accuracy of the model minus the Qwen3-0.6B baseline, same slots "
                        "and candidates",
            "interval": "percentile bootstrap over terms (query_key)",
            "test": "exact McNemar on slots, descriptive because slots cluster by term"},
        "secondary": {
            "comparison": "the same paired comparison against the primary pass of a finished Jev "
                          "run on the same slots (jev-entity-linking-v1), read from its hash-"
                          "verified outcomes.csv",
            "statistics": "paired counts, exact McNemar, bootstrap over terms and the term-level "
                          "comparison, on each gold vocabulary"},
        "bootstrap": {"resamples": DEFAULT_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": "0.95"},
        "strict_identity_is_the_headline": True,
        "pass_or_fail_criterion": None,
        "probability_source": "the log-probabilities of the first generated token (top "
                              f"{TOP_LOGPROBS}), summed over the tokens that are option labels and "
                              "normalised over those found; labels not reported count as 0",
        "confidence_signals": SIGNALS,
        "exploratory": [
            "every pass other than the primary", "consensus of the passes", "abstention",
            "position bias", "confidence thresholds and escalation", "QUDT/Wikidata equivalence",
            "subpopulations by gold vocabulary"],
        "prompt_changes_after_first_corpus_request": "none"}


# --- the reference run ---

def load_reference(folder: Path | str, baseline: Baseline) -> dict[str, Any]:
    """Read a finished Jev run's per-slot outcomes after proving it is what it says it is.

    Args: folder: the Jev results folder; baseline: the verified baseline of this run.
    Returns: its identity (folder, experiment and the hashes of its responses, plan and outcomes,
    which are its evidence; a rebuilt report does not change them) and its outcomes by slot id,
    each shaped like an evaluate_slot outcome as far as a comparison needs.
    Raises: RunError for a folder without a manifest, a file that no longer matches its hash, a
    run that is not a finished Jev run with its primary pass over this baseline's candidates, one
    that failed a check, or an outcomes file that does not cover exactly this run's slots (so a
    pilot is a valid reference only for a pilot of the same variables).
    Side effects: filesystem reads; hashing.
    """
    root = Path(folder)
    path = root / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("experiment") != REFERENCE_EXPERIMENT
                or PRIMARY_PASS not in [p[0] for p in manifest.get("passes", [])]):
            raise RunError(f"{root.name} is not a finished Jev run with its primary pass")
        files = manifest["files"]
        identity = manifest["baseline"]
    except (OSError, json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise RunError(f"{root} is not a finished run ({type(exc).__name__})") from exc
    for name, expected in files.items():
        if not (root / name).is_file() or file_sha256(root / name) != expected:
            raise RunError(f"{root.name}/{name} changed after its run; it is no longer evidence")
    if not all(c["passed"] is True for c in manifest["checks"]):
        raise RunError(f"{root.name} did not pass every one of its checks")
    mine = baseline.identity
    for field in ("scores_sha256", "candidates_sha256", "plan_sha256", "evidence_code_sha256"):
        if identity.get(field) != mine[field]:
            raise RunError(f"{root.name} was run on different baseline evidence ({field} differs)")
    with (root / "outcomes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    slots = [slot.slot_id for slot in baseline.slots]
    if [row["slot_id"] for row in rows] != slots:
        raise RunError(f"{root.name}/outcomes.csv does not cover exactly this run's slots")
    outcomes = {
        row["slot_id"]: {
            "slot_id": row["slot_id"], "query_key": row["query_key"],
            "reachable": row["reachable"] == "1",
            "gold": (row["gold_vocabulary"], row["gold_local_id"]),
            "correct": row["jev_correct"] == "1",
            "equivalent_correct": row["jev_equivalent_correct"] == "1"}
        for row in rows}
    # The evidence is pinned, not the report: rebuilding the reference's report (a new manifest, new
    # timestamps) changes neither its responses, its plan nor its per-slot outcomes.
    return {"identity": {"folder": root.name, "experiment": manifest["experiment"],
                         "responses_sha256": files["responses.jsonl"],
                         "plan_sha256": files["plan.json"], "outcomes_sha256": files["outcomes.csv"]},
            "outcomes": outcomes}


def plan_extra(reference: Path | str | None, baseline: Baseline) -> dict[str, Any]:
    """Name the reference run in the plan, so a rerun cannot quietly use another.

    Args: reference: the Jev results folder, or None; baseline: this run's verified baseline.
    Returns: {} without a reference, otherwise {"reference": its identity}.
    Raises: RunError as load_reference, and when the reference is not in the baseline's parent
    folder: the plan stores only its name, and the report phase (a rebuild included) looks it up
    there, so a reference anywhere else would pass planning, cost every request and then fail.
    Side effects: filesystem reads.
    """
    if reference is None:
        return {}
    here = Path(os.path.abspath(baseline.root)).parent
    if Path(os.path.abspath(reference)).parent != here:
        raise RunError(f"the reference run {Path(reference).name} must sit in the same folder as the "
                       f"baseline {baseline.root.name}: the plan names it by folder name and the "
                       "report looks it up beside the baseline. Copy or move it there.")
    return {"reference": load_reference(reference, baseline)["identity"]}


def extra_analysis(plan: Mapping[str, Any], baseline: Baseline, results: Mapping[str, Any],
                   analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the primary pass with the reference run on the same slots.

    Args: plan: the stored plan; baseline: the verified baseline (the reference folder is looked up
    beside it); results: pass_results by pass name; analysis: the analysis so far.
    Returns: {} for a run without a reference, otherwise {"reference": ...} with the reference's
    top-1 on each population and the paired, term-level and equivalence comparisons.
    Raises: RunError if the reference folder changed since the plan named it.
    Side effects: filesystem reads; hashing.
    """
    if "reference" not in plan:
        return {}
    named = plan["reference"]
    loaded = load_reference(baseline.root.parent / named["folder"], baseline)
    if loaded["identity"] != named:
        raise RunError(f"the reference run {named['folder']} changed since this run was planned")
    outcomes = [loaded["outcomes"][slot.slot_id] for slot in baseline.slots]
    own = [r["outcome"] for r in results[PRIMARY_PASS]]
    resamples = plan["analysis"]["bootstrap"]["resamples"]
    seed = plan["analysis"]["bootstrap"]["seed"]
    return {"reference": {
        "folder": named["folder"], "experiment": named["experiment"],
        "top1": {name: _count(sum(o["correct"] for o in outcomes if population(o)),
                              sum(1 for o in outcomes if population(o)))
                 for name, population in POPULATIONS},
        "equivalent_top1": _count(sum(o["equivalent_correct"] for o in outcomes if o["reachable"]),
                                  sum(1 for o in outcomes if o["reachable"])),
        "primary_vs_reference": {
            name: compare(own, outcomes, population, resamples=resamples, seed=seed)
            for name, population in POPULATIONS},
        "primary_vs_reference_equivalence": compare(
            own, outcomes, POPULATIONS[0][1], "equivalent_correct", resamples=resamples,
            seed=seed)}}


# --- checks, notes, accounting ---

def tokens_of(text: str | None) -> tuple[int, int] | None:
    """Read the tokens a stored response says it used, whether or not it was a valid answer.

    Args: text: a response body, or None.
    Returns: (prompt tokens, completion tokens), or None when the text is not JSON or reports no
    usable counts.
    Raises: None.
    Side effects: none.
    """
    if text is None:
        return None
    try:
        usage = json.loads(text)["usage"]
        counts = (usage["prompt_tokens"], usage["completion_tokens"])
    except (ValueError, KeyError, TypeError, RecursionError):
        return None
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts):
        return None
    return counts


def chat_checks(tasks: Sequence[Task], recorded: Mapping[tuple[str, str], Mapping[str, Any]],
                parsed: Mapping[tuple[str, str], Mapping[str, Any]],
                problems: Mapping[str, Sequence[Any]], *, history: Sequence[Mapping[str, Any]],
                inputs: Mapping[str, int], cap_usd: Decimal | None,
                max_requests: int | None, model: str = MODEL) -> list[dict[str, Any]]:
    """Run every check on the stored evidence of a chat-model run.

    Args: tasks: the requests; recorded: latest stored exchange per task key; parsed and problems:
    read_answers output; history: every stored exchange, including superseded failures; inputs:
    counts from the baseline verification; cap_usd and max_requests: unused (the service is not
    billed, and its request limit was enforced while asking); model: the model asked.
    Returns: the checks, in the order REQUIRED_CHAT_CHECKS lists them.
    Raises: None.
    Side effects: none.
    """
    total = len(tasks)
    done = [recorded[t.key] for t in tasks if t.key in parsed]
    retried = sum(r["attempts"] > 1 for r in done)
    unanswered = len(problems["missing"]) + len(problems["failed"])
    models = sorted({a["model"] for a in parsed.values()})
    builds = sorted({str(a["system_fingerprint"]) for a in parsed.values()})
    masses = [a["label_mass"] for a in parsed.values()]
    mismatched = [t.key for t in tasks
                  if t.key in recorded and (recorded[t.key].get("request_sha256") != t.sha256
                                            or recorded[t.key].get("shown") != list(t.shown))]
    primary = [t for t in tasks if t.spec.name == PRIMARY_PASS]
    primary_done = sum(t.key in parsed for t in primary)
    altered = unverified(history)
    baseline_slots = inputs.get("baseline_slots")
    inputs_consistent = baseline_slots is None or (
        inputs["slots"] == baseline_slots == inputs.get("baseline_records"))
    rejected = sum(r.get("parse_error") is not None for r in history)
    usage = [tokens_of(r["response_text"]) for r in history
             if r.get("status") == 200 and r.get("response_text") is not None]
    unreadable = sum(u is None for u in usage)
    tokens = sum(sum(u) for u in usage if u is not None)
    return [
        check(CHECK_INPUTS, inputs_consistent,
              f"{inputs['slots']} slots and {inputs['candidates']} candidate documents rebuilt "
              f"and equal to the baseline's; {inputs['baseline_files']} baseline files re-hashed"
              + (f"; the baseline scored {baseline_slots} slots in {inputs['baseline_records']} "
                 "records" if baseline_slots is not None else "")
              if inputs_consistent else
              f"this tree builds {inputs['slots']} slots but the baseline scored {baseline_slots} "
              f"in {inputs.get('baseline_records')} records"),
        check(CHECK_ANSWERED, len(parsed) == total,
              f"{len(parsed)}/{total} requests answered ({retried} needed an HTTP retry; "
              f"{rejected} stored responses were rejected as invalid answers and asked again)"
              if unanswered == 0 else
              f"{unanswered} of {total} requests have no usable response "
              f"({len(problems['missing'])} never sent)"),
        check(CHECK_REQUESTS, not mismatched,
              f"{total - len(mismatched)}/{total} stored requests equal the ones this tree builds"),
        check(CHECK_HASHES, not altered,
              f"{sum(r.get('response_text') is not None for r in history)} stored responses "
              "re-hashed" if not altered else
              f"{len(altered)} stored responses do not match their hash"),
        check(CHECK_PARSED, not problems["unparseable"],
              f"{len(parsed)} responses read strictly; at least {min(masses, default=1.0):.4f} "
              f"of every first token's probability sits on option labels (required: "
              f"{MIN_LABEL_MASS}), no response held reasoning text" if not problems["unparseable"]
              else f"{len(problems['unparseable'])} responses are not valid answers, e.g. "
                   f"{problems['unparseable'][0][1]}"),
        check(CHECK_MODEL, models == [model] and len(builds) == 1,
              f"served as {', '.join(models) or 'nothing'} by build "
              f"{', '.join(builds) or 'nobody'} (requested {model})"),
        check(CHECK_USAGE, unreadable == 0,
              f"{tokens:,} tokens used over {len(parsed)} answers and {rejected} rejected "
              "responses, every stored response included; not billed to the owner"
              + (f"; {unreadable} stored responses report no token usage" if unreadable else "")),
        check(CHECK_PRIMARY, bool(primary) and primary_done == len(primary),
              f"{primary_done}/{len(primary)} requests of {PRIMARY_PASS} answered"),
    ]


def chat_spend(parsed: Mapping[Any, Mapping[str, Any]], recorded: Mapping[Any, Mapping[str, Any]],
               history: Sequence[Mapping[str, Any]], cap_usd: Decimal | None,
               plan: Mapping[str, Any], max_requests: int | None = None) -> dict[str, Any]:
    """Account for what the run used.

    Args: parsed: parsed answers; recorded: the latest stored exchange per task; history: every
    stored exchange, rejected ones included; cap_usd: unused; plan: the stored plan; max_requests:
    the request limit of the invocation.
    Returns: answered requests with their tokens, the service's latency, the request limit, and the
    fact that the service is not billed to the owner.
    Raises: KeyError for a malformed answer.
    Side effects: none.
    """
    latencies = [r["request_s"] for r in history if r.get("request_s") is not None]
    return {"requests": len(parsed),
            "responses_stored": sum(r.get("response_text") is not None for r in history),
            "rejected_responses": sum(r.get("parse_error") is not None for r in history),
            "requests_sent": sum(r.get("status") is not None for r in history),
            "input_tokens": sum(a["input_tokens"] for a in parsed.values()),
            "output_tokens": sum(a["output_tokens"] for a in parsed.values()),
            "cost_usd": "0", "billing": "owner-reported free access; no price applies",
            "max_requests": max_requests,
            "answered_after_retry": sum(recorded[k]["attempts"] > 1 for k in parsed),
            "failed_attempts_repeated": sum(len(recorded[k]["retries"]) for k in parsed),
            "request_seconds": {"total": round(sum(latencies), 1),
                                "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
                                "max": round(max(latencies, default=0.0), 3)}}


def chat_notes(summary: Mapping[str, Any], counts: Mapping[str, int], asked: Mapping[str, int],
               margin_ties: tuple[int, int], rejected: Mapping[str, int], cap: int | None,
               has_repeat: bool, parsed: Mapping[Any, Mapping[str, Any]], model: str = MODEL
               ) -> list[str]:
    """Write the report's notes from this run's numbers.

    Args: summary: the run summary; counts: population counts; asked: slots and empty pools;
    margin_ties: (primary selections that tie with their runner-up, primary selections that have
    a runner-up); rejected: reason -> number of stored responses that failed the validity rule
    and were asked again; cap: the description cap the model saw; has_repeat: whether an identical
    repeat pass was run; parsed: parsed answers; model: the model asked.
    Returns: the notes.
    Raises: KeyError for a malformed summary.
    Side effects: none.
    """
    pops = summary["primary"]["populations"]
    unreachable = pops.get("unreachable", {"n": 0})["n"]
    ties, with_runner_up = margin_ties
    shown = "uncapped" if cap is None else f"capped at {cap} characters"
    masses = sorted(a["label_mass"] for a in parsed.values())
    outside = [(1 - a["top_mass"]) / a["label_mass"] for a in parsed.values()]
    reported = Counter(a["labels_reported"] for a in parsed.values())
    reasoned = sum(count for reason, count in rejected.items() if "reasoned" in reason)
    repeat = summary["analysis"].get("repeatability")
    control = ", ".join(f"{outer}.{inner}={str(value).lower()}"
                        for outer, nested in REASONING_OFF[model].items()
                        for inner, value in nested.items())
    notes = [
        f"The model saw, for every candidate, the label and the description ({shown}): the "
        "information the Qwen3 reranker saw, with no vocabulary, identifier, rank or score. The "
        "term and the variable's definition are the baseline's, and the candidates were shown in "
        "the same seeded orders the Jev run used.",
        f"{asked['empty_pools']} slots have no candidate; they were not asked and count as "
        "misses, as in the baseline.",
        f"{counts.get('excluded_label_only', 0)} label-only nodes and "
        f"{counts.get('excluded_non_vocabulary', 0)} non-vocabulary IRIs are outside the "
        "population, as in the baseline.",
        f"{unreachable} slots have their gold in a vocabulary these sources cannot reach; they "
        "count as misses in the \"All controlled-vocabulary slots\" row and are left out of the "
        "primary population (the reachable slots).",
        f"Reasoning is switched off with `{control}`, the control "
        f"the lab measured for {model} on this gateway; the parser refuses any response that "
        "carries reasoning text, and "
        + ("none did." if not reasoned else f"{reasoned} did and were asked again."),
        "A probability is the model's probability of that label given that its first token is a "
        f"label, estimated from the {TOP_LOGPROBS} most probable tokens the gateway reports: the "
        "tokens that spell a label are summed and divided by the total over the labels found, so "
        "the options sum to 1, and a label outside the reported tokens counts as 0. The share of "
        f"the first token's probability on option labels was at least {masses[0]:.4f} in every "
        f"response (median {masses[len(masses) // 2]:.4f}), the rest sitting on other tokens, "
        "which the division leaves out; between "
        f"{min(reported)} and {max(reported)} labels were reported per response, and what lay "
        "outside the reported tokens bounds the error of any probability at "
        f"{max(outside):.4f}." if masses else "No response was read.",
        f"{ties} of {with_runner_up} primary selections that have a runner-up tie with it and were "
        "decided by the baseline's rule (source order, then rank).",
        "Gold terms are the corpus labels of the gold nodes; for QUDT slots they are often QUDT's "
        "own labels, so QUDT accuracy on these terms may overstate accuracy on LLM-predicted "
        "wording.",
        "The service is shared and its answers can differ slightly with what else its server is "
        "processing: in development, identical requests sent concurrently gave different "
        "probabilities and identical requests sent one after another gave identical ones, so the "
        "experiment sends one request at a time unless told otherwise (the header says how this "
        "run was sent)."
        + (f" The repeated pass (an identical request, sent later in the same run) selected the "
           f"same candidate for {ratio(repeat['same_selection'])} and gave identical "
           f"probabilities for {ratio(repeat['identical_scores'])}." if has_repeat and repeat
           else "")
        + " A later run, or one made while the server is busier, may differ, and a later build of "
          "the model server is a different system."]
    if rejected:
        reasons = "; ".join(f"{count} x {reason}" for reason, count in sorted(rejected.items()))
        notes.append(
            f"{sum(rejected.values())} stored response(s) were rejected by the validity rule and "
            f"the request was asked again, as the protocol says ({reasons}); rejected responses "
            "stay in responses.jsonl.")
    return notes


def chat_model_block(parsed: Mapping[Any, Mapping[str, Any]], plan: Mapping[str, Any]
                     ) -> dict[str, Any]:
    """Describe the model asked and the build that answered.

    Args: parsed: parsed answers; plan: the stored plan.
    Returns: requested and served model, the provider, the server builds seen, and the request
    parameters.
    Raises: KeyError for a malformed answer or plan.
    Side effects: none.
    """
    return {"requested": plan["model"]["requested"],
            "served": sorted({a["model"] for a in parsed.values()}),
            "providers": sorted({str(a["provider"]) for a in parsed.values()}),
            "builds": sorted({str(a["system_fingerprint"]) for a in parsed.values()}),
            "gateway": plan["model"]["gateway"], "billing": plan["model"]["billing"],
            "request_parameters": plan["prompt"]["request_parameters"]}


# --- the voice ---

def _cost_sentence(manifest: Mapping[str, Any]) -> str:
    """Say how many requests were answered and what they used.

    Args: manifest: run manifest.
    Returns: the sentence that follows the period of the requests in the header.
    Raises: KeyError if the manifest lacks a field.
    Side effects: none.
    """
    spend = manifest["spend"]
    seconds = spend["request_seconds"]
    workers = [i["workers"] for i in manifest["invocations"] if i["asking"]]
    sent = ("" if not workers else ", sent one request at a time" if max(workers) == 1
            else f", sent by up to {max(workers)} workers")
    return (f"{spend['requests']} requests answered ({spend['input_tokens']:,} input tokens, "
            f"{spend['output_tokens']:,} output tokens; {seconds['total']:g} s of request time, "
            f"{seconds['mean']:g} s on average, {seconds['max']:g} s at most) over "
            f"{spend['responses_stored']} stored responses{sent}. The service is not billed to the "
            "owner.")


def _intro(manifest: Mapping[str, Any]) -> str:
    """Describe how the model was asked and what the primary result is.

    Args: manifest: run manifest.
    Returns: the paragraph that follows the identity paragraph.
    Raises: KeyError if the manifest lacks a field.
    Side effects: none.
    """
    cap = manifest["prompt"]["max_description_chars"]
    name = manifest["model"]["requested"]
    phrases = {"shuffle-a": "an identical repeat of the primary request",
               "reverse-a": "the reverse order"}
    others = [p for p in manifest["passes"] if p[0] != PRIMARY_PASS]
    again = ("The same pool was also asked, to see how much the result depends on order and on "
             "chance: " + "; ".join(phrases.get(p[2], f"another candidate order ({p[2]})")
                                    for p in others) + ". ") if others else ""
    params = manifest["model"]["request_parameters"]
    scope = ("The comparison with Jev on the same slots, also by gold vocabulary, is pre-specified "
             "too; everything else, the rows by gold vocabulary against the baseline and every "
             "reading that counts a QUDT kind and its equivalent items as one entity included, is "
             "exploratory." if manifest.get("reference") else
             "No reference run was named, so the pre-specified comparison with Jev was not made; "
             "everything else is exploratory.")
    return (f"{name} is a chat model served by PSNC; reasoning is switched off, the temperature is "
            f"{params['temperature']} and one token is generated. Each slot's pooled candidates, "
            "exactly as the Qwen3 baseline saw them (label and description "
            f"{'uncapped' if cap is None else f'capped at {cap} characters'}, no source tag), "
            "were put to it as a numbered list with an extra option 0, `none of these`, and it "
            "was asked to reply with one label. Its probabilities for the options are read from "
            "the log-probabilities of that first token. The primary result, fixed in the run's plan "
            "and in DECISIONS L-016 and L-017 before the first request (no commit predates the "
            "run), is the "
            "first pass (a seeded candidate order) with the forced choice: the candidate with the "
            "highest probability, `none_of_these` ignored, ties decided as in the baseline. "
            + again + scope)


def _confidence_notes(analysis: Mapping[str, Any]) -> list[str]:
    """Say what the own-choice probability measures when the model refuses.

    Args: analysis: the analysis block.
    Returns: the paragraphs that follow the signals table, as Markdown lines.
    Raises: KeyError for a malformed analysis.
    Side effects: none.
    """
    a = analysis["abstention"]
    return ["", "The model has no confidence value of its own; its own-choice probability is the "
            "probability of the option it answered with, `none_of_these` included. For "
            f"{a['forced_choice_correct_when_none_chosen']['denominator']} reachable slots that "
            f"choice was `none_of_these` ({a['refusals_at_confidence_0.9']} of them at 0.9 or "
            "more), so for those slots it measures confidence in the refusal, not in the forced "
            "candidate whose correctness is scored here."]


def _reference_sections(summary: Mapping[str, Any], voice_name: str) -> list[str]:
    """Write the comparison with the reference (Jev) run.

    Args: summary: the run summary; voice_name: the model's name.
    Returns: Markdown lines; none for a run without a reference.
    Raises: KeyError for a malformed summary.
    Side effects: none.
    """
    ref = summary["analysis"].get("reference")
    if ref is None:
        return []
    top1, versus = summary["primary"]["populations"], ref["primary_vs_reference"]
    base = summary["baseline"]["populations"]
    labels = (("reachable", "Gold in Wikidata, QUDT or OM (primary)"),
              ("all", "All controlled-vocabulary slots"), ("gold:wikidata", "Gold in Wikidata"),
              ("gold:qudt", "Gold in QUDT"), ("gold:om", "Gold in OM"))
    rows = [[label, str(top1[name]["n"]), ratio(top1[name]["top1"]), ratio(base[name]["top1"]),
             ratio(ref["top1"][name]), _difference(versus[name]), _p(versus[name]["mcnemar_p"])]
            for name, label in labels if name in top1]
    c, eq = versus["reachable"], ref["primary_vs_reference_equivalence"]
    out = ["", f"## {voice_name} against Jev on the same slots (pre-specified)", "",
           f"Jev is the primary pass of the finished run `{ref['folder']}` (read from its published, "
           "hash-verified outcomes), asked about the same candidates. Differences are in top-1 "
           f"correct slots ({voice_name} minus Jev); the interval resamples whole terms.", ""]
    out += _table(["Population", "Slots", f"{voice_name} top-1", "Qwen3 top-1", "Jev top-1",
                   f"{voice_name} minus Jev", "McNemar p"], rows)
    out += ["", _paired_sentence(c, voice_name),
            f"Strict identity on the reachable slots: {voice_name} minus Jev {_difference(c)}; "
            f"exact McNemar p {_p(c['mcnemar_p'])}.",
            f"Counted once per term: {_terms(c['terms'], voice_name, 'Jev')}.",
            "Exploratory, because whether such twins count is an open question of the project: if "
            f"a QUDT kind and its `wikidataMatch` item counted as one entity, {voice_name} minus "
            f"Jev {_difference(eq)}; {_terms(eq['terms'], voice_name, 'Jev')}.", ""]
    return out


def _paired_sentence(block: Mapping[str, Any], voice_name: str) -> str:
    """Show the paired counts of two systems in one sentence.

    Args: block: compare output; voice_name: the first system.
    Returns: a sentence naming the slots both got right, only one got right, and neither.
    Raises: KeyError for a malformed block.
    Side effects: none.
    """
    return (f"On the {block['n']} reachable slots both are right on {block['both']}, only "
            f"{voice_name} on {block['only_first']}, only Jev on {block['only_second']} and "
            f"neither on {block['neither']}.")


def voice_for(model: str) -> Voice:
    """Name a chat model for the published files.

    Args: model: the PSNC model id.
    Returns: the voice that describes it.
    Raises: None.
    Side effects: none.
    """
    return Voice(
        key="llm", name=model, own_confidence_word="an own-choice probability",
        own_confidence_short="own-choice probability",
        own_confidence_label=f"{model}: probability of its own choice (`none_of_these` included)",
        pointwise=False, cost_sentence=_cost_sentence, intro=_intro,
        confidence_notes=_confidence_notes,
        reliability_note="Probabilities are read from the first token's log-probabilities, so "
                         "they are continuous, not rounded; they are estimated from the "
                         f"{TOP_LOGPROBS} most probable tokens, and the notes say how much of "
                         "the first token those left out.",
        attribution=f"answers from {model} served by PSNC", post_hoc="exploratory",
        term_level_note="exploratory: slots cluster by term", term_heading="exploratory",
        extra=lambda summary: _reference_sections(summary, model))


# --- the experiment ---

def _client(env_file: str | None, root: Path) -> RedactingChatClient:
    """Build the client for a real run.

    Args: env_file: credential file or None; root: iadopt-lab root, whose parent holds ``.env``.
    Returns: a chat client that holds the key privately and redacts the forms of failure text that
    ``chat.scrub`` leaves (``chat_redact``).
    Raises: RunError when no key can be read or the base URL is not allowed.
    Side effects: reads the credential file.
    """
    key, base_url = credentials_from(env_file or _default_env(root), os.environ)
    return RedactingChatClient(key, base_url=base_url)


def _model_plan(model: str) -> dict[str, Any]:
    """Describe the model and the service for the plan.

    Args: model: the PSNC model id.
    Returns: the model requested, the gateway host and the billing basis.
    Raises: None.
    Side effects: none.
    """
    return {"requested": model, "gateway": "llm.hpc.psnc.pl",
            "billing": "not billed to the owner (owner-reported free access, recorded in the lab's "
                       "parameters.yml); the run is limited by a number of requests"}


def chat_experiment(model: str = MODEL) -> Experiment:
    """Describe the experiment of one PSNC chat model.

    Args: model: the PSNC model id; it must have a measured way to turn reasoning off.
    Returns: the experiment the engine runs.
    Raises: KeyError for a model without one.
    Side effects: none.
    """
    if model not in REASONING_OFF:
        raise KeyError(f"{model!r} has no measured control for turning reasoning off; known: "
                       f"{sorted(REASONING_OFF)}")
    return Experiment(
        voice=voice_for(model), experiment_id=EXPERIMENT_ID,
        title=f"{model} (reasoning off, PSNC) over the baseline's pooled Wikidata, QUDT and OM "
              "candidates",
        evidence_files=PSNC_EVIDENCE_FILES, passes=CHAT_PASSES,
        analysis_plan=analysis_plan(), plan_fields=PLAN_FIELDS_WITH_REFERENCE, billed=False,
        formula_check=False, fatal=FATAL,
        prompt_plan=lambda cap: prompt_plan(cap, model), model_plan=lambda: _model_plan(model),
        build_tasks=lambda baseline, specs, session: build_tasks(
            baseline, specs, model=model, session_id=session),
        estimate=estimate, parse_answer=lambda task, text: parse_chat_choice(text, task.offered),
        cost_of=reported_cost,
        checks=lambda *a, **k: chat_checks(*a, model=model, **k),
        manifest_model=chat_model_block, spend=chat_spend,
        notes=lambda *a: chat_notes(*a, model=model), make_client=_client,
        plan_extra=plan_extra, extra_analysis=extra_analysis)


PSNC = chat_experiment()


def run_chat(out_dir: Path | str, *, baseline_dir: Path | str, max_requests: int | None,
             model: str = MODEL, env_file: str | None = None, lab_root: Path | str | None = None,
             only: Sequence[str] | None = None, passes: Sequence[str] | None = None,
             workers: int = 1, run_id: str | None = None, resamples: int = DEFAULT_RESAMPLES,
             client_factory: Any = None, reference: Path | str | None = None) -> dict[str, Any]:
    """Run the whole experiment of a PSNC chat model: verify, plan, ask, check, analyse, report.

    Args: out_dir: results folder; baseline_dir: the finished baseline to compare against and take
    candidates from; max_requests: the request limit (required while requests are pending);
    model: the PSNC model id; env_file: credential file defining PSNC_API_KEY, used only when
    requests are pending; lab_root: iadopt-lab root; only: variable stems for a pilot; passes: pass
    names (default all); workers: parallel requests (one by default, see the module docstring);
    run_id: name recorded in the manifest; resamples: bootstrap replicates; client_factory:
    injectable for tests; reference: a finished Jev run on the same candidates to compare with.
    Returns: the manifest.
    Raises: as run_experiment; KeyError for a model without a measured reasoning control.
    Side effects: network I/O, files in out_dir.
    """
    return run_experiment(chat_experiment(model), out_dir, baseline_dir=baseline_dir,
                          max_requests=max_requests, env_file=env_file, lab_root=lab_root,
                          only=only, passes=passes, workers=workers, run_id=run_id,
                          resamples=resamples, client_factory=client_factory, reference=reference)


__all__ = ["PSNC", "PSNC_EVIDENCE_FILES", "REQUIRED_CHAT_CHECKS", "chat_experiment",
           "load_reference", "run_chat", "voice_for"]
