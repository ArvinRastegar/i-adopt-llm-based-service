"""Compare two runs of the baseline: how far the live services and the scores drifted apart.

Live retrieval changes with the services, so a second run is a replication check as much as a
repeat. The comparison reads only the stored evidence of both runs; it calls nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from iadopt_linking_eval import evaluate_slot

from .errors import UsageError
from .recorder import JsonlStore, latest
from .retrieve import is_ok
from .sources import SOURCES

EXAMPLES = 12
EVIDENCE = ("candidates.jsonl", "scores.jsonl")


def _require_run(root: Path) -> None:
    """Refuse a folder that does not hold a run's evidence.

    Args: root: a results folder.
    Returns: None.
    Raises: UsageError naming the folder and the missing file.
    Side effects: filesystem checks.
    """
    for name in EVIDENCE:
        if not (root / name).is_file():
            raise UsageError(f"{root} is not a results folder: {name} is missing")


def _retrievals(root: Path) -> dict[tuple[str, str], list[str] | None]:
    """Load the ordered candidate IRIs of every retrieval.

    Args: root: a results folder.
    Returns: (source, query key) -> candidate IRIs in rank order, or None for a retrieval that
    failed, so a failure is never mistaken for an empty result.
    Raises: StoreError for a corrupt candidates file.
    Side effects: filesystem read.
    """
    records = latest(JsonlStore(root / "candidates.jsonl").read(),
                     lambda r: (r["source"], r["query_key"]))
    return {key: [c["iri"] for c in r["candidates"]] if is_ok(r) else None
            for key, r in records.items()}


def _selection(outcome: Mapping[str, Any]) -> tuple[str, str] | None:
    """Name what a slot's selected candidate was.

    Args: outcome: evaluate_slot result.
    Returns: (source, local identifier), or None when nothing was selected.
    Raises: None.
    Side effects: none.
    """
    chosen = outcome["selected"]
    return None if chosen is None else (chosen["source"], chosen["local_id"])


def compare_runs(first: Path | str, second: Path | str) -> dict[str, Any]:
    """Measure how much two runs agree.

    Args: first, second: results folders with candidates.jsonl and scores.jsonl.
    Returns: per-source retrieval agreement, the headline correct counts of each run, the slots
    whose selected candidate changed (with examples), and the score differences of candidates
    present in both runs.
    Raises: UsageError for a folder that is not a results folder; StoreError for a corrupt
    evidence file.
    Side effects: filesystem reads.
    """
    a, b = Path(first), Path(second)
    _require_run(a)
    _require_run(b)
    ra, rb = _retrievals(a), _retrievals(b)
    shared = sorted(set(ra) & set(rb))
    retrieval = {}
    for source in SOURCES:
        keys = [k for k in shared if k[0] == source]
        unavailable = sum(ra[k] is None or rb[k] is None for k in keys)
        usable = [k for k in keys if ra[k] is not None and rb[k] is not None]
        identical = sum(ra[k] == rb[k] for k in usable)
        reordered = sum(ra[k] != rb[k] and sorted(ra[k]) == sorted(rb[k]) for k in usable)
        retrieval[source] = {"keys": len(keys), "identical": identical, "reordered": reordered,
                             "different": len(usable) - identical - reordered,
                             "unavailable": unavailable}
    sa = latest(JsonlStore(a / "scores.jsonl").read(), lambda r: r["slot_id"])
    sb = latest(JsonlStore(b / "scores.jsonl").read(), lambda r: r["slot_id"])
    slots = sorted(set(sa) & set(sb))
    correct = [0, 0]
    changed: list[dict[str, Any]] = []
    deltas: list[float] = []
    for slot in slots:
        outcomes = (evaluate_slot(sa[slot]), evaluate_slot(sb[slot]))
        correct[0] += outcomes[0]["correct"]
        correct[1] += outcomes[1]["correct"]
        if _selection(outcomes[0]) != _selection(outcomes[1]):
            changed.append({"slot_id": slot, "term": outcomes[0]["term"],
                            "gold": outcomes[0]["gold"], "first": _selection(outcomes[0]),
                            "second": _selection(outcomes[1]),
                            "first_correct": outcomes[0]["correct"],
                            "second_correct": outcomes[1]["correct"]})
        later = {c["iri"]: c["score"] for c in sb[slot]["pool"]}
        deltas += [abs(c["score"] - later[c["iri"]]) for c in sa[slot]["pool"] if c["iri"] in later]
    return {
        "slots_compared": len(slots), "only_in_first": len(set(sa) - set(sb)),
        "only_in_second": len(set(sb) - set(sa)), "retrieval": retrieval,
        "correct_first": correct[0], "correct_second": correct[1],
        "selection_changed": len(changed), "changed_examples": changed[:EXAMPLES],
        "common_candidates": len(deltas), "max_score_delta": max(deltas, default=0.0),
        "mean_score_delta": sum(deltas) / len(deltas) if deltas else 0.0,
    }


def render_comparison(result: Mapping[str, Any], names: Sequence[str] = ("first", "second")) -> str:
    """Write the comparison as Markdown.

    Args: result: compare_runs output; names: labels for the two runs.
    Returns: Markdown text.
    Raises: KeyError for a malformed result.
    Side effects: none.
    """
    n = result["slots_compared"]
    lines = [f"# Comparison of {names[0]} and {names[1]}", "",
             f"{n} slots compared ({result['only_in_first']} only in {names[0]}, "
             f"{result['only_in_second']} only in {names[1]}).", "",
             f"Correct: {names[0]} {result['correct_first']}/{n}, {names[1]} "
             f"{result['correct_second']}/{n}. The selected candidate changed for "
             f"{result['selection_changed']} slots.", "",
             "| Source | Queries | Identical lists | Same set, reordered | Different | Unavailable |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {s} | {v['keys']} | {v['identical']} | {v['reordered']} | {v['different']} | "
              f"{v['unavailable']} |" for s, v in result["retrieval"].items()]
    lines += ["", f"Candidates present in both runs: {result['common_candidates']}; score "
              f"difference mean {result['mean_score_delta']:.2e}, max {result['max_score_delta']:.2e}.",
              ""]
    if result["changed_examples"]:
        lines += ["Slots whose selection changed:", ""]
        lines += [f"- `{c['term']}`: {c['first']} -> {c['second']}; gold {c['gold'][0]}:{c['gold'][1]}"
                  f" ({'correct' if c['first_correct'] else 'wrong'} -> "
                  f"{'correct' if c['second_correct'] else 'wrong'})" for c in result["changed_examples"]]
    return "\n".join(lines) + "\n"
