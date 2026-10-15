"""Run health: the checks a run must pass, the retrieval summary and the report's notes.

Everything here is a function of stored evidence; nothing reads a file or calls a service. A check
has three states: True (passed), False (failed, which stops the run) and None (not run, shown as
such and refused for every check except the device cross-check, which may be skipped).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from iadopt_linking_eval import REACHABLE

from .devices import describe_device_check, device_passed
from .errors import RunError
from .population import Slot
from .retrieve import is_ok, response_hash_matches
from .sources import RESULT_LIMIT, SOURCES

SOURCE_NAMES = tuple(SOURCES)
CORE_QUDT_VOCABULARY = "http://qudt.org/vocab/quantitykind"
# A source that answers for fewer keys than this is broken or misused, whatever its status codes.
MIN_ANSWER_SHARE = Fraction(1, 20)

CHECK_ACCOUNTED = "every linked node accounted for"
CHECK_RETRIEVED = "every key retrieved from every source"
CHECK_ANSWERED = "every source answered for some keys"
CHECK_HASHES = "stored response hashes verify"
CHECK_RELEASES = "source releases recorded"
CHECK_SCORED = "every slot scored"
CHECK_RANGE = "scores finite and in [0, 1]"
CHECK_LIMIT = f"at most {RESULT_LIMIT} candidates per source"
CHECK_CONFIG = "one scoring configuration"
CHECK_DEVICE = "device consistency"
REQUIRED_CHECKS = (CHECK_ACCOUNTED, CHECK_RETRIEVED, CHECK_ANSWERED, CHECK_HASHES, CHECK_RELEASES,
                   CHECK_SCORED, CHECK_RANGE, CHECK_LIMIT, CHECK_CONFIG, CHECK_DEVICE)
# May be "not run" in a report (a pilot, --skip-device-check); every other check may not.
OPTIONAL_CHECKS = (CHECK_DEVICE,)


def check(name: str, passed: bool | None, detail: str) -> dict[str, Any]:
    """Form one check result.

    Args: name: the check's name; passed: True, False, or None when it was not run; detail: the
    numbers behind the result.
    Returns: {"name", "passed", "detail"}.
    Raises: None.
    Side effects: none.
    """
    return {"name": name, "passed": passed, "detail": detail}


def population_check(slots: Sequence[Slot], counts: Mapping[str, int]) -> dict[str, Any]:
    """Check that every linked node is either scored or left out for a stated reason.

    Args: slots: the population; counts: population counts, with ``gold_links`` counted from the
    gold's own link lists rather than by the population loop.
    Returns: the check; it passes when the gold's links, the population loop's links and the sum
    of scored, label-only and non-vocabulary nodes agree, and the slots match the scored count.
    Raises: None.
    Side effects: none.
    """
    kept = counts.get("slots", 0)
    label_only = counts.get("excluded_label_only", 0)
    other = counts.get("excluded_non_vocabulary", 0)
    gold = counts.get("gold_links", -1)
    ok = len(slots) == kept and gold == counts.get("links", -2) == kept + label_only + other
    return check(CHECK_ACCOUNTED, ok, f"{gold} linked nodes in the gold: {len(slots)} scored, "
                 f"{label_only} label only, {other} non-vocabulary IRI")


def retrieval_checks(keys: Sequence[str], history: Sequence[Mapping[str, Any]] | None,
                     releases: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Check the retrieval evidence.

    Args: keys: query keys; history: every stored retrieval record in file order, or None when
    retrieval has not run; releases: merge_releases output, or None.
    Returns: the four retrieval checks in REQUIRED_CHECKS order.
    Raises: None.
    Side effects: none.
    """
    names = (CHECK_RETRIEVED, CHECK_ANSWERED, CHECK_HASHES, CHECK_RELEASES)
    if history is None:
        return [check(name, None, "not run yet") for name in names]
    wanted = set(keys)
    mine = [r for r in history if r["query_key"] in wanted]
    pairs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in mine:
        pairs[(record["source"], record["query_key"])].append(record)
    final = {pair: records[-1] for pair, records in pairs.items()}
    unresolved = sum(1 for name in SOURCE_NAMES for key in keys if not is_ok(final.get((name, key))))
    repeated = sum(1 for records in pairs.values() for r in records[:-1] if not is_ok(r))
    retried = sum(1 for r in final.values() if is_ok(r) and r["attempts"] > 1)
    answered = {name: sum(1 for key in keys if is_ok(final.get((name, key)))
                          and final[(name, key)]["candidates"]) for name in SOURCE_NAMES}
    thin = [name for name in SOURCE_NAMES
            if not keys or Fraction(answered[name], len(keys)) < MIN_ANSWER_SHARE]
    stored = [r for r in mine if r.get("response_text") is not None]
    unverified = [r for r in mine if not response_hash_matches(r)]
    if unresolved == 0:
        retrieved = (f"{len(keys)} keys x {len(SOURCE_NAMES)} sources, all answered; {repeated} "
                     f"failed requests were repeated successfully and {retried} needed an HTTP "
                     "retry")
    else:
        retrieved = (f"{unresolved} of {len(keys) * len(SOURCE_NAMES)} (source, key) pairs have "
                     "no successful retrieval")
    result = [
        check(CHECK_RETRIEVED, unresolved == 0, retrieved),
        check(CHECK_ANSWERED, not thin,
              ", ".join(f"{name} {answered[name]}/{len(keys)}" for name in SOURCE_NAMES)
              + f" keys returned candidates (floor {float(MIN_ANSWER_SHARE):.0%})"),
        check(CHECK_HASHES, not unverified,
              f"{len(stored)} stored responses re-hashed" if not unverified else
              f"{len(unverified)} stored responses do not match their hash"),
    ]
    if releases is None:
        result.append(check(CHECK_RELEASES, None, "not run yet"))
    else:
        missing = [name for name in ("om", "qudt") if releases[name]["latest"] is None]
        result.append(check(
            CHECK_RELEASES, not missing,
            "OM and QUDT release data probed" + (
                f" ({releases['om']['probes']} + {releases['qudt']['probes']} probes)" if not missing
                else f"; no usable probe of {', '.join(missing)}")))
    return result


def scoring_checks(slots: Sequence[Slot], records: Mapping[str, Mapping[str, Any]] | None,
                   max_description_chars: int | None) -> list[dict[str, Any]]:
    """Check the scoring evidence.

    Args: slots: the population; records: scores records by slot id, or None when scoring has
    not run; max_description_chars: the description cap the run asked for.
    Returns: the four scoring checks in REQUIRED_CHECKS order.
    Raises: None.
    Side effects: none.
    """
    names = (CHECK_SCORED, CHECK_RANGE, CHECK_LIMIT, CHECK_CONFIG)
    if records is None:
        return [check(name, None, "not run yet") for name in names]
    pools = [record["pool"] for record in records.values()]
    scores = [candidate["score"] for pool in pools for candidate in pool]
    in_range = all(isinstance(s, int | float) and math.isfinite(s) and 0.0 <= s <= 1.0
                   for s in scores)
    sizes = [len(pool) for pool in pools] or [0]
    over = any(sum(1 for c in pool if c["source"] == name) > RESULT_LIMIT
               for pool in pools for name in SOURCE_NAMES)
    configs = Counter(
        (s.get("model"), s.get("revision"), s.get("batch_size"), s.get("max_description_chars"))
        for s in ((record.get("scoring") or {}) for record in records.values()))
    devices = Counter((record.get("scoring") or {}).get("device") for record in records.values())
    model, revision, _, cap = configs.most_common(1)[0][0] if configs else (None,) * 4
    config_ok = (len(configs) == 1 and model is not None and revision is not None
                 and cap == max_description_chars)
    where = ", ".join(f"{device} {count} slots"
                      for device, count in sorted(devices.items(), key=lambda i: str(i[0])))
    return [
        check(CHECK_SCORED, len(records) == len(slots), f"{len(records)} of {len(slots)} slots"),
        check(CHECK_RANGE, in_range, f"{len(scores)} scored candidates"),
        check(CHECK_LIMIT, not over, f"pool size min {min(sizes)}, max {max(sizes)}"),
        check(CHECK_CONFIG, config_ok,
              f"{model} revision {str(revision)[:10]}, description cap {cap}; {where}"
              if len(configs) == 1 else f"{len(configs)} different scoring configurations found"),
    ]


def device_consistency_check(device: Mapping[str, Any] | None) -> dict[str, Any]:
    """Turn a device cross-check into its check result.

    Args: device: devices.device_check output, or None when it was not run.
    Returns: the check; None when the cross-check was skipped.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    if device is None:
        return check(CHECK_DEVICE, None, "the cross-device check was not run")
    return check(CHECK_DEVICE, device_passed(device), describe_device_check(device))


def build_checks(*, slots: Sequence[Slot], counts: Mapping[str, int], keys: Sequence[str],
                 max_description_chars: int | None,
                 history: Sequence[Mapping[str, Any]] | None = None,
                 releases: Mapping[str, Any] | None = None,
                 records: Mapping[str, Mapping[str, Any]] | None = None,
                 device: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """List every required check with its current state.

    Args: slots, counts, keys: the population; max_description_chars: the asked-for cap; history:
    stored retrieval records or None; releases: merged release probes or None; records: scores
    records by slot id or None; device: device cross-check or None.
    Returns: one result per REQUIRED_CHECKS name, in that order; a phase that has not run
    contributes "not run" results.
    Raises: None.
    Side effects: none.
    """
    return [population_check(slots, counts), *retrieval_checks(keys, history, releases),
            *scoring_checks(slots, records, max_description_chars),
            device_consistency_check(device)]


def ensure_checks(checks: Sequence[Mapping[str, Any]], *, final: bool) -> None:
    """Stop the run if a check failed, or, for a final report, if one was not run.

    Args: checks: build_checks output; final: True before a report is written.
    Returns: None.
    Raises: RunError naming each failed check and its numbers; with final, also each check that
    was not run, except those in OPTIONAL_CHECKS.
    Side effects: none.
    """
    failed = [c for c in checks if c["passed"] is False]
    if failed:
        raise RunError("health check failed: " + "; ".join(
            f"{c['name']} ({c['detail']})" for c in failed))
    if final:
        unrun = [c["name"] for c in checks if c["passed"] is None
                 and c["name"] not in OPTIONAL_CHECKS]
        if unrun:
            raise RunError("health checks did not run: " + ", ".join(unrun))


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    """Describe a list of durations.

    Args: values: seconds.
    Returns: median and maximum to milliseconds; None for both when there are none.
    Raises: None.
    Side effects: none.
    """
    if not values:
        return {"median": None, "max": None}
    return {"median": round(statistics.median(values), 3), "max": round(max(values), 3)}


def retrieval_summary(history: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """Summarise retrieval health per source, counting failures that were later repeated.

    Args: history: every stored retrieval record in file order; keys: query keys.
    Returns: per source: keys answered, empty results, candidate-count distribution, candidates
    and how many lack a description, the longest description, requests that needed an HTTP retry
    with the causes, failed requests that were repeated with the causes, and service and wall
    time; for QUDT also the distinct kinds, the vocabularies defining them and the number of
    kinds outside the core vocabulary.
    Raises: None.
    Side effects: none.
    """
    wanted = set(keys)
    mine = [r for r in history if r["query_key"] in wanted]
    final: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in mine:
        final[(record["source"], record["query_key"])] = record
    out: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        done = [r for (source, _), r in final.items() if source == name and is_ok(r)]
        past = [r for r in mine if r["source"] == name]
        failures = [r for r in past if not is_ok(r)]
        sizes = Counter(len(r["candidates"]) for r in done)
        candidates = [c for r in done for c in r["candidates"]]
        asked = [r for r in done if r["request"] is not None]
        out[name] = {
            "keys": len(done), "empty": sizes.get(0, 0), "candidate_counts": dict(sorted(sizes.items())),
            "limit": RESULT_LIMIT, "candidates": len(candidates),
            "empty_descriptions": sum(1 for c in candidates if not c["description"]),
            "longest_description_chars": max((len(c["description"]) for c in candidates), default=0),
            "retried": sum(1 for r in done if r["attempts"] > 1),
            "retry_causes": dict(sorted(Counter(
                cause.split(";")[0] for r in past for cause in r.get("attempt_errors", [])).items())),
            "failed_requests": len(failures),
            "failure_causes": dict(sorted(Counter(
                str(r["error"] or r["parse_error"])[:120] for r in failures).items())),
            "service_seconds": _spread([r["request_s"] for r in asked]),
            "wall_seconds": _spread([r["latency_s"] for r in asked])}
    kinds = {c["iri"]: tuple(c["extras"].get("defined_by", []))
             for (source, _), r in final.items() if source == "qudt" and is_ok(r)
             for c in r["candidates"]}
    out["qudt"]["distinct_kinds"] = len(kinds)
    out["qudt"]["defined_by"] = sorted({iri for defined in kinds.values() for iri in defined})
    out["qudt"]["extension_kinds"] = sum(
        1 for defined in kinds.values() if defined != (CORE_QUDT_VOCABULARY,))
    return out


def evidence_period(history: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """Say when the retrieved evidence was gathered.

    Args: history: stored retrieval records in file order; keys: query keys.
    Returns: the earliest and latest retrieval time of the answers in use; None for both when
    there are none.
    Raises: None.
    Side effects: none.
    """
    wanted = set(keys)
    final: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in history:
        if record["query_key"] in wanted:
            final[(record["source"], record["query_key"])] = record
    times = [r["retrieved_at"] for r in final.values() if is_ok(r)]
    return {"first_retrieval": min(times, default=None), "last_retrieval": max(times, default=None)}


def notes_for(*, counts: Mapping[str, int], retrieval: Mapping[str, Any],
              summary: Mapping[str, Any], device: Mapping[str, Any] | None,
              releases: Mapping[str, Any], period: Mapping[str, Any],
              max_description_chars: int | None) -> list[str]:
    """Write the caveats a reader of the report needs, from this run's own numbers.

    Args: counts: population counts; retrieval: retrieval_summary output; summary: metrics
    summary; device: device cross-check or None; releases: merged release probes; period:
    evidence_period output; max_description_chars: the description cap.
    Returns: one sentence per caveat.
    Raises: KeyError for a malformed input.
    Side effects: none.
    """
    pops = summary["populations"]
    om = retrieval["om"]
    cap = (f"cut to {max_description_chars} characters when a document is built"
           if max_description_chars is not None else "not cut")
    notes = [
        f"Retrieval is live: the candidates in candidates.jsonl are what the services returned "
        f"between {period['first_retrieval']} and {period['last_retrieval']}. They will drift, "
        "so a later run need not match; rescoring and reporting on the stored candidates is "
        "exactly repeatable.",
        f"Descriptions are {cap}; the longest retrieved description is "
        f"{retrieval['qudt']['longest_description_chars']} characters for QUDT and "
        f"{retrieval['wikidata']['longest_description_chars']} for Wikidata.",
        (f"OLS returned no description for any of the {om['candidates']} OM candidates, so the "
         "reranker scores them from their label alone." if om["empty_descriptions"] == om["candidates"]
         else f"{om['empty_descriptions']} of {om['candidates']} OM candidates have no "
              "description, and the reranker scores those from their label alone."),
        "Gold terms are the corpus labels of the gold nodes; for QUDT slots they are often QUDT's "
        "own labels, so QUDT accuracy on these terms may overstate accuracy on LLM-predicted "
        "wording.",
    ]
    other = sorted(name.split(":", 1)[1] for name in pops
                   if name.startswith("gold:") and name.split(":", 1)[1] not in REACHABLE)
    if other:
        missed = pops["all"]["n"] - pops["reachable"]["n"]
        notes.append(f"{missed} slots have gold in a vocabulary these sources cannot reach "
                     f"({', '.join(other)}); they count as misses in the first row of the result "
                     "table and are left out of the others.")
    notes.append(
        "Gold IRIs are compared by vocabulary and local identifier, never as strings. Read by "
        f"pattern: {counts.get('gold_form_repaired', 0)} malformed IRIs, "
        f"{counts.get('gold_form_variant', 0)} in a variant path form.")
    qudt = retrieval["qudt"]
    notes.append(f"QUDT's endpoint also serves community-extension kinds: {qudt['extension_kinds']} "
                 f"of the {qudt['distinct_kinds']} distinct kinds retrieved are defined outside "
                 "the core quantity-kind vocabulary.")
    changing = [name for name in ("om", "qudt") if releases[name]["distinct_releases"] > 1]
    if changing:
        notes.append("Retrieval spanned more than one release of " + " and ".join(changing)
                     + "; the manifest lists each probe.")
    near = summary["margins"]["below_1e-3"]
    if device is None:
        notes.append(f"{near['numerator']} of {near['denominator']} selections were decided by a "
                     "top-two margin below 1e-3 and could change with the device or the batch; "
                     "the cross-device check was not run for this report.")
    else:
        notes.append(
            f"{near['numerator']} of {near['denominator']} selections were decided by a top-two "
            "margin below 1e-3 and could change with the device or the batch. The cross-device "
            f"check rescored {device['near_ties']} near-ties and {device['spread']} other slots "
            f"on {device['reference_device']}: {len(device['selection_changes'])} selections "
            f"changed; over those {device['slots_checked']} slots the correct count was "
            f"{device['correct_on_primary']} on {device['primary_device']} and "
            f"{device['correct_on_reference']} on {device['reference_device']}.")
    return notes
