"""What Jev is asked: the baseline's frozen slots and candidates, turned into requests.

A Jev run never retrieves anything. It reads a finished, hash-verified baseline folder, proves that
the corpus and the code in this tree still reproduce that baseline's queries and candidate documents
exactly, and builds one request per (slot, pass). Whatever differs between the two experiments is
therefore the scorer, not the information it was given.
"""

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
from .provenance import evidence_code_identity
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
        """Identify the baseline for a plan: which run, which bytes, which code.

        Args: none.
        Returns: folder name, the manifest's file hashes of the evidence, and its code hashes.
        Raises: KeyError if the manifest lacks a field.
        Side effects: none.
        """
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
        """Name the task for resuming.

        Args: none.
        Returns: (slot id, pass name).
        Raises: None.
        Side effects: none.
        """
        return (self.slot.slot_id, self.spec.name)


def _verify_manifest(root: Path) -> dict[str, Any]:
    """Prove that a baseline folder is finished, unaltered and made by the code in this tree.

    Args: root: the baseline's results folder.
    Returns: its manifest.
    Raises: RunError for a missing or unreadable manifest, a file that no longer matches its hash,
    a baseline that did not pass every check or was a pilot, or evidence code that has changed.
    Side effects: hashes every published file.
    """
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
    if manifest["evidence_code_sha256"] != evidence_code_identity():
        raise RunError(f"the retrieval, scoring and gold code changed since {root.name} was made, "
                       "so this tree can no longer prove it reproduces the baseline's inputs")
    return manifest


def _verify_inputs(slots: Sequence[Slot], records: Mapping[str, Mapping[str, Any]],
                   cap: int | None) -> None:
    """Prove that what Jev will be given is exactly what the baseline scored.

    Args: slots: the population from this tree; records: the baseline's scores by slot id; cap:
    the baseline's description cap.
    Returns: None.
    Raises: RunError naming the first difference: a slot with no record, a different term, gold,
    query or candidate document, or two candidates with the same identity in one pool.
    Side effects: none.
    """
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
    """Load a baseline run and verify it end to end.

    Args: root: the baseline's results folder; lab_root: iadopt-lab root; only: variable stems to
    keep (a pilot).
    Returns: the baseline with its slots and scores records.
    Raises: RunError if the folder is not a finished, unaltered, current baseline or if what Jev
    would be given differs from what the baseline scored; UsageError for an unknown stem.
    Side effects: filesystem reads; hashing.
    """
    folder = Path(root)
    manifest = _verify_manifest(folder)
    records = dict(latest(JsonlStore(folder / "scores.jsonl").read(), lambda r: r["slot_id"]))
    _, slots, counts = load_population(lab_root, only)
    cap = manifest["config"]["max_description_chars"]
    _verify_inputs(slots, records, cap)
    return Baseline(folder, manifest, records, slots, counts, cap)


def build_tasks(baseline: Baseline, specs: Sequence[PassSpec], *, session_id: str | None
                ) -> tuple[list[Task], dict[str, int]]:
    """Build every request of a run.

    Args: baseline: the verified baseline; specs: the passes; session_id: OpenRouter log grouping
    label.
    Returns: the tasks, pass by pass in population order, and counts: slots and empty pools. A
    slot with no candidates is not asked about: there is nothing to choose.
    Raises: ValueError for a pool too large for the options Jev accepts.
    Side effects: none; deterministic.
    """
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
    """Estimate a run's cost before any billed call, erring high.

    Args: tasks: the requests; price_per_token_usd: the model's input price per token as a decimal
    string.
    Returns: requests, request characters, the assumed characters per token, the token estimate,
    the price and the estimated cost in USD as a fixed-point string.
    Raises: decimal.InvalidOperation for a malformed price.
    Side effects: none.
    """
    chars = sum(task.chars for task in tasks)
    tokens = int(chars / ASSUMED_CHARS_PER_TOKEN) + 1
    cost = Decimal(price_per_token_usd) * tokens
    by_pass = Counter(task.spec.name for task in tasks)
    return {"requests": len(tasks), "requests_by_pass": dict(sorted(by_pass.items())),
            "request_chars": chars, "assumed_chars_per_token": ASSUMED_CHARS_PER_TOKEN,
            "input_tokens_upper_estimate": tokens, "price_per_input_token_usd": price_per_token_usd,
            "estimated_cost_usd": format(cost, "f")}
