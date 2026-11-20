#!/usr/bin/env python
"""Stage 0 of the pattern-retrieval work: one scoreboard, fixed folds, and the fresh variables.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Steps 1 to 5 showed that the experts' patterns lift the decomposition and the patterns the model
finds do not. The stages that follow try to find the right patterns more often. This module is
what every one of them is measured with. It calls no model.

- The scoreboard marks any set of found patterns against the experts' tags on all 102 variables:
  each of the six decisions, and "complete" (all six right), by group and overall.
- The folds split the 102 variables into six fixed parts. A method that learns from tagged
  variables may, for a target, use only variables of the other five parts.
- The fresh list names the tagged variables of the examples repository that are not among the
  102. It reads the stored snapshot only; nothing is downloaded here.

    ./scoreboard.py --board     # the stored walks of steps 2 to 5 on the scoreboard
    ./scoreboard.py --folds     # the fold sizes and the fingerprint of the split
    ./scoreboard.py --fresh     # the tagged variables outside the 102
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import groups  # noqa: E402
import route_key  # noqa: E402
import rulebook  # noqa: E402
import settings  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

GROUPS = ("examples", "practice", "held-out")
# The six decisions of a pattern set, and the name `walk.mark` gives each one's mark.
DECISIONS = {"property": "property", "object": "object", "matrix": "matrix", "context": "context",
             "modifier": "modifier", "constraint": "constraint_set"}
WALKS = ("direct", "steps", "direct-all", "mixed", "mixed-all")
FOLDS = 6
FOLD_SEED = 20261123
BOARD = HERE.parent / "output" / "retrieval" / "scoreboard.json"


def ids_by_group(ctx: Mapping[str, Any]) -> dict[str, list[str]]:
    """The 102 variables by group.

    Args: ctx: from `groups.load`.
    Returns: {"examples": 25 ids, "practice": 53 ids, "held-out": 24 ids}, each sorted.
    Raises: ValueError when the groups overlap or do not hold 102 variables.
    Side Effects: none. Deterministic.
    """
    out = {group: groups.targets(ctx, group) for group in GROUPS}
    everyone = [v for ids in out.values() for v in ids]
    if len(everyone) != 102 or len(set(everyone)) != 102:
        raise ValueError("the three groups are not 102 distinct variables")
    return out


def stored_walk(ctx: Mapping[str, Any], name: str) -> dict[str, dict[str, Any] | None]:
    """The patterns one stored walk of steps 2 to 5 found, for all 102 variables.

    Args: ctx: from `groups.load`; name: one of `WALKS`.
    Returns: {variable id: facts, or None where the walk gave none}.
    Raises: KeyError when a variable has no stored walk.
    Side Effects: reads the stored walk files.
    """
    out: dict[str, dict[str, Any] | None] = {}
    for group, ids in ids_by_group(ctx).items():
        rows = (settings.walks_named(ctx, name, "practice") if group == "practice"
                else groups.walks_named(ctx, name, group))
        out.update({v: rows[v].get("facts") for v in ids})
    return out


def marks(ctx: Mapping[str, Any], found: Mapping[str, Mapping[str, Any] | None]) -> dict[str, dict[str, bool]]:
    """Mark found patterns against the experts' tags, variable by variable.

    Args: ctx: from `groups.load`; found: {variable id: facts or None}.
    Returns: {variable id: {each decision: right?, "complete": all six right?}}.
    Raises: KeyError when a variable has no experts' tags.
    Side Effects: none. Deterministic.
    """
    out = {}
    for variable_id, facts in found.items():
        mark = walk.mark(ctx, facts, ctx["key"][variable_id])
        row = {decision: bool(mark[name]) for decision, name in DECISIONS.items()}
        row["complete"] = all(row.values())
        out[variable_id] = row
    return out


def wilson(right: int, of: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95% Wilson interval of a share.

    Args: right: successes; of: trials; z: the normal quantile.
    Returns: (low, high); (0.0, 1.0) for no trials.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if not of:
        return 0.0, 1.0
    share = right / of
    centre = share + z * z / (2 * of)
    spread = z * math.sqrt(share * (1 - share) / of + z * z / (4 * of * of))
    return (centre - spread) / (1 + z * z / of), (centre + spread) / (1 + z * z / of)


def board(ctx: Mapping[str, Any], found: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """The scoreboard of one set of found patterns.

    Args: ctx: from `groups.load`; found: {variable id: facts or None} for any of the 102.
    Returns: {"all" and each group: {each decision and "complete": {"right", "of", "share",
        "low", "high"}}}; a group without a marked variable is left out.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    marked = marks(ctx, found)
    rows = {"all": sorted(marked)}
    rows.update({group: [v for v in ids if v in marked] for group, ids in ids_by_group(ctx).items()})
    out: dict[str, Any] = {}
    for name, ids in rows.items():
        if not ids:
            continue
        cells = {}
        for column in (*DECISIONS, "complete"):
            right = sum(marked[v][column] for v in ids)
            low, high = wilson(right, len(ids))
            cells[column] = {"right": right, "of": len(ids), "share": right / len(ids), "low": low, "high": high}
        out[name] = cells
    return out


def folds(ctx: Mapping[str, Any], count: int = FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    """Split the 102 variables into fixed parts of equal size.

    Each part gets its share of every group and of the variables with and without a matrix, the
    decision the walks get wrong most often.

    Args: ctx: from `groups.load`; count: number of parts; seed: of the shuffles.
    Returns: {variable id: part number from 0}.
    Raises: nothing.
    Side Effects: none. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    out: dict[str, int] = {}
    position = 0
    for group, ids in ids_by_group(ctx).items():
        for has_matrix in (True, False):
            stratum = sorted(v for v in ids if bool(ctx["key"][v]["matrix"]) == has_matrix)
            rng.shuffle(stratum)
            for variable_id in stratum:
                out[variable_id] = position % count
                position += 1
    return out


def usable_for(target: str, fold_of: Mapping[str, int]) -> list[str]:
    """The tagged variables a learning method may use for one target.

    Args: target: the variable being classified; fold_of: from `folds`.
    Returns: the sorted ids of every variable in another part than the target's.
    Raises: KeyError for an unknown target.
    Side Effects: none. Deterministic.
    """
    return sorted(v for v, fold in fold_of.items() if fold != fold_of[target])


def fresh(snapshot: Sequence[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The tagged entries of the examples repository that are none of the 102 variables.

    Args: snapshot: the stored issue snapshot; key: the experts' tags of the 102.
    Returns: rows {"issue", "title", "state", "property", "flags", "corpus_path"} of every entry
        with a property tag whose issue is not in the key, by issue number.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    used = {entry["issue"] for entry in key.values()}
    rows = []
    for issue in sorted(snapshot, key=lambda item: item["number"]):
        parsed = route_key.parse_labels(list(issue["labels"]))
        if issue["number"] in used or not parsed["property"]:
            continue
        rows.append({"issue": issue["number"], "title": issue["title"], "state": issue["state"],
                     "property": parsed["property"], "flags": parsed["flags"],
                     "corpus_path": issue.get("corpus_path")})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode: board, folds or fresh.

    Args: argv: command-line arguments; `sys.argv` when None.
    Returns: the exit status, 0 on success.
    Raises: SystemExit on bad arguments.
    Side Effects: --board writes the scoreboard file; no network.
    """
    parser = argparse.ArgumentParser(description="Scoreboard, folds and fresh variables (stage 0).")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--board", "--folds", "--fresh"):
        mode.add_argument(name, action="store_true")
    args = parser.parse_args(argv)
    ctx = groups.load()
    if args.board:
        boards = {name: board(ctx, stored_walk(ctx, name)) for name in WALKS}
        BOARD.parent.mkdir(parents=True, exist_ok=True)
        BOARD.write_text(json.dumps(boards, indent=2) + "\n", encoding="utf-8")
        for name, rows in boards.items():
            for row, cells in rows.items():
                print(f"{name:11} {row:9} " + "  ".join(
                    f"{column} {cell['right']}/{cell['of']}" for column, cell in cells.items()))
        return 0
    if args.folds:
        fold_of = folds(ctx)
        group_of = {v: group for group, ids in ids_by_group(ctx).items() for v in ids}
        for fold in range(FOLDS):
            members = [v for v, f in fold_of.items() if f == fold]
            print(fold, len(members), dict(Counter(group_of[v] for v in members)),
                  "with matrix", sum(bool(ctx["key"][v]["matrix"]) for v in members))
        print("sha256", ft.sha256("\n".join(f"{v} {fold_of[v]}" for v in sorted(fold_of))))
        return 0
    rows = fresh(rulebook.load("issues_snapshot")["issues"], ctx["key"])
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(len(rows), "entries;", sum(bool(r["flags"]) for r in rows), "flagged;",
          sum(r["corpus_path"] is not None for r in rows), "link a corpus file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
