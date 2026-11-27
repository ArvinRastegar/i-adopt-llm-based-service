"""Cross-check the reranker's scores on a second device."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from iadopt_linking_eval import evaluate_slot

from .rerank import Scorer, validate_scores

# Scores are sigmoids of fp32 logits: a gap beyond this between devices is a different
# computation, not rounding (a 2026-10-14 sample of ten slots differed by 9.4e-06 at most).
DEVICE_TOLERANCE = 1e-4
REPRODUCIBLE_TOLERANCE = 1e-6
# Selections whose top-two margin is below this are rescored on the other device.
NEAR_TIE_MARGIN = 1e-3
SPREAD_SAMPLE = 10


def reference_device(primary_device: str | None) -> str:
    """Name the device to cross-check against."""
    return "mps" if primary_device == "cpu" else "cpu"


def top_two_margin(record: Mapping[str, Any]) -> float | None:
    """Measure how decisively a slot's best candidate beats the runner-up."""
    ranked = sorted((candidate["score"] for candidate in record["pool"]), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) >= 2 else None


def check_slots(records: Mapping[str, Mapping[str, Any]],
                spread: int = SPREAD_SAMPLE) -> tuple[list[str], int]:
    """Choose the slots to rescore on the other device."""
    multi = {slot_id: record for slot_id, record in records.items() if len(record["pool"]) >= 2}
    near = [slot_id for slot_id, record in multi.items()
            if top_two_margin(record) < NEAR_TIE_MARGIN]
    taken = set(near)
    others = [slot_id for slot_id in multi if slot_id not in taken]
    step = max(1, len(others) // spread) if spread else 1
    return near + (others[::step][:spread] if spread else []), len(near)


def _rescored(record: Mapping[str, Any], scores: list[float]) -> dict[str, Any]:
    """Copy a scores record with other scores in its pool."""
    return {**record, "pool": [{**candidate, "score": score} for candidate, score
                               in zip(record["pool"], scores, strict=True)]}


def device_check(records: Mapping[str, Mapping[str, Any]], primary: Scorer, reference: Scorer, *,
                 primary_device: str | None, reference_device_name: str,
                 spread: int = SPREAD_SAMPLE,
                 progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Rescore a sample on the other device and on the primary device again."""
    chosen, near_ties = check_slots(records, spread)
    largest = reproducibility = 0.0
    changes: list[dict[str, Any]] = []
    correct = [0, 0]
    for done, slot_id in enumerate(chosen, 1):
        record = records[slot_id]
        documents = [candidate["document"] for candidate in record["pool"]]
        stored = [candidate["score"] for candidate in record["pool"]]
        other = validate_scores(reference.score(record["query"], documents), len(documents))
        again = validate_scores(primary.score(record["query"], documents), len(documents))
        largest = max([largest, *(abs(a - b) for a, b in zip(stored, other, strict=True))])
        reproducibility = max([reproducibility,
                               *(abs(a - b) for a, b in zip(stored, again, strict=True))])
        mine, theirs = evaluate_slot(record), evaluate_slot(_rescored(record, other))
        correct[0] += mine["correct"]
        correct[1] += theirs["correct"]
        picked = [(o["selected"]["source"], o["selected"]["local_id"]) for o in (mine, theirs)]
        if picked[0] != picked[1]:
            changes.append({
                "slot_id": slot_id, "term": record["term"], "margin": top_two_margin(record),
                "primary": list(picked[0]), "reference": list(picked[1]),
                "primary_correct": mine["correct"], "reference_correct": theirs["correct"]})
        if progress is not None:
            progress(done, len(chosen))
    return {
        "primary_device": primary_device, "reference_device": reference_device_name,
        "slots_checked": len(chosen), "near_ties": near_ties, "spread": len(chosen) - near_ties,
        "near_tie_margin": NEAR_TIE_MARGIN, "tolerance": DEVICE_TOLERANCE,
        "reproducibility_tolerance": REPRODUCIBLE_TOLERANCE,
        "max_abs_difference": largest, "max_reproducibility_difference": reproducibility,
        "selection_changes": changes, "correct_on_primary": correct[0],
        "correct_on_reference": correct[1]}


def device_passed(result: Mapping[str, Any]) -> bool:
    """Say whether a device check is within its tolerances."""
    return bool(result["slots_checked"] > 0
                and result["max_abs_difference"] <= result["tolerance"]
                and result["max_reproducibility_difference"] <= result["reproducibility_tolerance"])


def describe_device_check(result: Mapping[str, Any]) -> str:
    """Summarise a device check in one sentence."""
    changed = len(result["selection_changes"])
    return (f"{result['slots_checked']} slots rescored on {result['reference_device']} "
            f"({result['near_ties']} with a top-two margin below {result['near_tie_margin']:g}, "
            f"{result['spread']} others): max |difference| {result['max_abs_difference']:.2e} "
            f"(tolerance {result['tolerance']:g}); rescoring on {result['primary_device']} "
            f"differed by at most {result['max_reproducibility_difference']:.1e}; "
            f"{changed} selections change on {result['reference_device']} (correct among the "
            f"checked: {result['correct_on_primary']} on {result['primary_device']}, "
            f"{result['correct_on_reference']} on {result['reference_device']})")
