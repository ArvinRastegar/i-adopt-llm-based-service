"""What Jev is asked: the baseline's frozen slots and candidates, turned into requests."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .artifacts import file_sha256
from .errors import RunError
from .experiment import load_population
from .jev import (
    ARM_CHOICE,
    NONE_KEY,
    PassSpec,
    candidate_key,
    choice_request,
    encode,
    noul_request,
    option_id,
    presentation_order,
    request_sha256,
)
from .population import Slot, normalise_term
from .provenance import RESULTS_CODE, RESULTS_EVIDENCE, evidence_code_identity
from .recorder import JsonlStore, latest
from .rerank import build_document
from .score import slot_query

# Jev's tokenizer is not published and the tutorial's example needs about 1.9 characters per token;
# the estimate assumes fewer so that it errs high.
ASSUMED_CHARS_PER_TOKEN = 1.5


@dataclass(frozen=True)
class Baseline:
    """A finished baseline run, verified, with the slots it scored."""

    root: Path
    manifest: dict[str, Any]
    records: dict[str, dict[str, Any]]
    slots: tuple[Slot, ...]
    counts: dict[str, int]
    max_description_chars: int | None

    @property
    def identity(self) -> dict[str, Any]:
        """Identify the baseline for a plan: which run, which bytes, which code."""
        files = self.manifest["files"]
        return {"folder": self.root.name, "run_id": self.manifest["run_id"],
                "scores_sha256": files["scores.jsonl"], "candidates_sha256": files["candidates.jsonl"],
                "plan_sha256": files["plan.json"],
                "evidence_code_sha256": self.manifest["evidence_code_sha256"],
                "max_description_chars": self.max_description_chars}


@dataclass(frozen=True)
class Task:
    """One request: a slot asked in one pass."""

    slot: Slot
    spec: PassSpec
    order: tuple[int, ...]
    shown: tuple[str, ...]
    offered: tuple[str, ...]
    body: dict[str, Any]
    sha256: str
    chars: int

    @property
    def key(self) -> tuple[str, str]:
        """Name the task for resuming."""
        return (self.slot.slot_id, self.spec.name)


def _verify_manifest(root: Path) -> dict[str, Any]:
    """Prove a baseline is finished, unaltered, and made by this tree's or RESULTS_CODE's evidence code."""
    path = root / "manifest.json"
    if not path.is_file():
        raise RunError(f"{root} is not a finished baseline: manifest.json is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RunError(f"{path} cannot be read as a baseline manifest ({exc})") from exc
    for name, expected in files.items():
        if not (root / name).is_file() or file_sha256(root / name) != expected:
            raise RunError(f"{root.name}/{name} changed after its run; it is no longer evidence")
    if manifest.get("pilot_variables") is not None:
        raise RunError(f"{root.name} is a pilot, not a baseline")
    if not all(check["passed"] is True for check in manifest["checks"]):
        raise RunError(f"{root.name} did not pass every one of its checks")
    made_by = manifest["evidence_code_sha256"]
    if made_by not in (evidence_code_identity(), RESULTS_EVIDENCE["baseline"]):
        raise RunError(f"{root.name} was made neither by this tree's retrieval, scoring and gold code "
                       f"nor by that code at {RESULTS_CODE[:7]}, so its inputs cannot be proven")
    return manifest


def _verify_inputs(slots: Sequence[Slot], records: Mapping[str, Mapping[str, Any]],
                   cap: int | None) -> None:
    """Prove that what Jev will be given is exactly what the baseline scored."""
    for slot in slots:
        record = records.get(slot.slot_id)
        if record is None:
            raise RunError(f"the baseline has no scores for {slot.slot_id}")
        gold = record["gold"]
        if (record["term"] != slot.term or gold["vocabulary"] != slot.gold_vocabulary
                or gold["local_id"] != slot.gold_local_id):
            raise RunError(f"{slot.slot_id}: the baseline's term or gold differs from the corpus")
        if record["query"] != slot_query(slot):
            raise RunError(f"{slot.slot_id}: the baseline's query differs from the one this tree "
                           "builds")
        keys = [candidate_key(c) for c in record["pool"]]
        if len(set(keys)) != len(keys):
            raise RunError(f"{slot.slot_id}: two candidates share one identity")
        for candidate in record["pool"]:
            if build_document(candidate["label"], candidate["description"], cap) != (
                    candidate["document"]):
                raise RunError(f"{slot.slot_id}: the document of {candidate_key(candidate)} "
                               "differs from the one the baseline scored")


def load_baseline(root: Path | str, *, lab_root: Path | str | None = None,
                  only: Sequence[str] | None = None) -> Baseline:
    """Load a baseline run and verify it end to end."""
    folder = Path(root)
    manifest = _verify_manifest(folder)
    records = dict(latest(JsonlStore(folder / "scores.jsonl").read(), lambda r: r["slot_id"]))
    _, slots, counts = load_population(lab_root, only)
    cap = manifest["config"]["max_description_chars"]
    _verify_inputs(slots, records, cap)
    return Baseline(folder, manifest, records, slots, counts, cap)


def build_tasks(baseline: Baseline, specs: Sequence[PassSpec], *, session_id: str | None
                ) -> tuple[list[Task], dict[str, int]]:
    """Build every request of a run."""
    tasks: list[Task] = []
    empty = sum(not baseline.records[slot.slot_id]["pool"] for slot in baseline.slots)
    for spec in specs:
        build = choice_request if spec.arm == ARM_CHOICE else noul_request
        for slot in baseline.slots:
            pool = baseline.records[slot.slot_id]["pool"]
            if not pool:
                continue
            order = presentation_order(pool, slot.slot_id, spec.order)
            body = build(normalise_term(slot.term), slot.definition, pool, order,
                         max_description_chars=baseline.max_description_chars,
                         session_id=session_id)
            offered = tuple(option_id(place) for place in range(1, len(pool) + 1)) + (
                (NONE_KEY,) if spec.arm == ARM_CHOICE else ())
            shown = tuple(candidate_key(pool[index]) for index in order)
            tasks.append(Task(slot, spec, tuple(order), shown, offered, body,
                              request_sha256(body), len(encode(body))))
    return tasks, {"slots": len(baseline.slots), "empty_pools": empty}


def estimate(tasks: Sequence[Task], price_per_token_usd: str) -> dict[str, Any]:
    """Estimate a run's cost before any billed call, erring high."""
    chars = sum(task.chars for task in tasks)
    tokens = int(chars / ASSUMED_CHARS_PER_TOKEN) + 1
    cost = Decimal(price_per_token_usd) * tokens
    by_pass = Counter(task.spec.name for task in tasks)
    return {"requests": len(tasks), "requests_by_pass": dict(sorted(by_pass.items())),
            "request_chars": chars, "assumed_chars_per_token": ASSUMED_CHARS_PER_TOKEN,
            "input_tokens_upper_estimate": tokens, "price_per_input_token_usd": price_per_token_usd,
            "estimated_cost_usd": format(cost, "f")}
