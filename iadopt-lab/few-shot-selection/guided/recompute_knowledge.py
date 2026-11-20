#!/usr/bin/env python
"""Independent recomputation of stage 3 (knowledge for the pattern-finding step).

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `paper/tracking/PATTERN_RETRIEVAL_20261119.md` (private repository).

Standard library only, and nothing imported from the lab or from the other modules of this
folder. Every stored answer is read and marked again with this file's own reading of the answer
and of the experts' tags (`data/route_key.json`, `data/patterns.json`,
`data/constraint_types.json`); the folds are dealt again from the groups stored by steps 1 and 5;
and every figure of `knowledge-summary.json` is worked out a second way: each cell's shares, the
effect of each kind of knowledge with both intervals, the setting chosen fold by fold and its
figures, the agreement of the repetitions and the cost. The list of cells and of what is compared
is held here, not taken from the summary. It also fails when an answer came from another host or
model than the pinned one or carries reasoning, when the prompts stored with the answers are not
the ones fingerprinted before the classifier run, and when the six rule texts are not the ones
fingerprinted in the record (written from the fingerprinted prompts, by the pinned host, ended
normally, without reasoning).

    ./recompute_knowledge.py            # prints the comparison; exit status 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "output"
CALLS = OUTPUT / "retrieval" / "knowledge-calls.jsonl"
RULES = OUTPUT / "retrieval" / "knowledge-rules.jsonl"
SUMMARY = OUTPUT / "retrieval" / "knowledge-summary.json"
GROUP_FOLDER = OUTPUT / "guided"
GROUP_FILES = {"examples": "groups-examples-calls.jsonl", "practice": "ceiling-calls.jsonl",
               "held-out": "groups-held-out-calls.jsonl"}
DATA = HERE / "data"
HOST, MODEL = "Near AI", "qwen/qwen3.8-27b"
FACTORS = ("G", "D", "S", "I")
DECISIONS = ("property", "object", "matrix", "context", "modifier", "constraint")
COLUMNS = (*DECISIONS, "complete", "matrix_pattern", "both")
PRIMARY = ("matrix", "constraint")
EFFECT_COLUMNS = ("matrix", "constraint", "property", "complete", "both")
CHOSEN_COLUMNS = ("matrix", "constraint", "both", "complete")
REPETITIONS, DRAWS, SEED = 2, 10000, 20261126
FOLDS, FOLD_SEED, FOLDS_DIGEST = 6, 20261123, "80f37c48d2cf004c"
STRICT = 0.05 / 10
STRONG_SHAPES = ("ratio", "symmetric", "source_target")
TOLERANCE = 1e-9
# The digest of each cell's 102 prompts as the record holds it from before the classifier run:
# the cells without `I` in the plan, the cells with `I` in the run log once the rules existed.
PLAN_DIGESTS = {
    "base": "6356edf04f4a8753", "G": "2161ce02395214ea", "D": "f2da2ec19cdbc2ff", "G+D": "29f11c26c6d99bfe",
    "S": "3b9f070c4e672fda", "G+S": "a7180c3624086079", "D+S": "cb050d884a91cfbe", "G+D+S": "7a52080756970618",
    "I": "742e2756302739e2", "G+I": "fbfefed62a55ff93", "D+I": "e2d192e4e3640fe4", "G+D+I": "2eefbe04203909d9",
    "S+I": "2bd0690b92a21126", "G+S+I": "bd5c0a76d7a06870", "D+S+I": "fd3ddef769cd83e3",
    "G+D+S+I": "c7d8bd580c5a35af"}
# The prompts that asked for the rules (plan) and the rule texts they gave (run log), by fold.
RULE_PROMPT_DIGESTS = {0: "94226f96c6551942", 1: "c0fc8c443f23a033", 2: "716df497d5dd3d36",
                       3: "c258f3b4c8692c9c", 4: "3cf0ab2fdde4ea3d", 5: "d4dbe0ec70358299"}
RULE_TEXT_DIGESTS = {0: "33676525c942150a", 1: "938d5f2f73f6b703", 2: "9158aa5c59cefd03",
                     3: "0cd276208540db67", 4: "fc18703d7a69e784", 5: "3526bbaaa0536b46"}
ANSWER_KEYS = {"property", "matrix", "context", "modifier", "constraints"}


def cells() -> list[str]:
    """The 16 cells, written out here a second time.

    Args: none.
    Returns: "base" and every other combination of the four kinds of knowledge.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out = []
    for mask in range(16):
        on = [factor for bit, factor in enumerate(FACTORS) if mask & (1 << bit)]
        out.append("+".join(on) if on else "base")
    return out


def rows(path: Path) -> list[dict]:
    """Read a JSON-lines file.

    Args: path: the file.
    Returns: its rows; none when the file is missing.
    Raises: ValueError for a line that is not JSON.
    Side Effects: reads the file.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_hash(text: str) -> str:
    """SHA-256 of a text.

    Args: text: any text.
    Returns: the hex digest.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def kept_calls() -> dict[tuple, dict]:
    """One call per (cell, repetition, variable): the first that was answered, else the last failure.

    Args: none.
    Returns: {key: record}.
    Raises: OSError when the calls file cannot be read.
    Side Effects: reads the calls file.
    """
    out: dict[tuple, dict] = {}
    for record in rows(CALLS):
        key = (record["arm"], record["repetition"], record["variable_id"])
        if key not in out or not out[key]["ok"]:
            out[key] = record
    return out


def groups() -> dict[str, list[str]]:
    """The three groups of the corpus, from the variables stored by steps 1 and 5.

    Args: none.
    Returns: {group: sorted variable ids}.
    Raises: ValueError when the groups overlap or are not 25, 53 and 24 variables.
    Side Effects: reads three stored calls files.
    """
    out = {name: sorted({row["variable_id"] for row in rows(GROUP_FOLDER / file)})
           for name, file in GROUP_FILES.items()}
    sizes = [len(out[name]) for name in ("examples", "practice", "held-out")]
    if sizes != [25, 53, 24] or len({v for ids in out.values() for v in ids}) != 102:
        raise ValueError(f"the stored groups are not 25, 53 and 24 distinct variables: {sizes}")
    return out


def deal_folds(by_group: dict[str, list[str]], tags: dict[str, dict]) -> dict[str, int]:
    """Deal the 102 variables into the six folds again, as stage 0 defines them.

    Args: by_group: from `groups`; tags: the experts' tags by variable.
    Returns: {variable id: fold}: group by group, the variables with a matrix before those
        without, each such set sorted, shuffled by one generator and dealt round the folds.
    Raises: KeyError when a variable has no tags.
    Side Effects: none. Deterministic for the fixed seed.
    """
    rng = random.Random(FOLD_SEED)
    out: dict[str, int] = {}
    for group in ("examples", "practice", "held-out"):
        for has_matrix in (True, False):
            stratum = sorted(v for v in by_group[group] if bool(tags[v]["matrix"]) == has_matrix)
            rng.shuffle(stratum)
            for variable_id in stratum:
                out[variable_id] = len(out) % FOLDS
    return out


def read_answer(text: str | None) -> dict | None:
    """The JSON object in a model answer, if it is one.

    Args: text: the answer.
    Returns: the object; None when the answer holds no JSON object.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    text = (text or "").strip()
    candidates = [text]
    if "```" in text:
        candidates += [part.removeprefix("json").strip() for part in text.split("```")[1::2]]
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"):text.rindex("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class Rulebook:
    """The experts' patterns and constraint types as this file reads them."""

    def __init__(self) -> None:
        """Read the two data files.

        Args: none.
        Returns: nothing.
        Raises: OSError when a data file is missing.
        Side Effects: reads `data/patterns.json` and `data/constraint_types.json`.
        """
        self.patterns = json.loads((DATA / "patterns.json").read_text(encoding="utf-8"))["patterns"]
        types = json.loads((DATA / "constraint_types.json").read_text(encoding="utf-8"))["types"]
        self.name_of = {row["number"]: row["name"].lower() for row in types}
        self.number_of: dict[str, int] = {}
        for row in types:                       # two rows that share a name answer to the first
            self.number_of.setdefault(row["name"].lower(), row["number"])

    def found(self, answer: dict | None) -> dict | None:
        """The pattern set an answer states, read strictly.

        Args: answer: from `read_answer`.
        Returns: {"property", "object", "matrix", "context", "constraint": numbers, "modifier",
            "names", "invented": how many constraint names are no constraint type}; None when
            the answer is not the object the prompt asks for: the five keys, a property pattern,
            a matrix that is null or a matrix pattern, true or false for context and modifier,
            and a list of names for the constraints.
        Raises: nothing.
        Side Effects: none. Deterministic.
        """
        if answer is None or not ANSWER_KEYS <= set(answer):
            return None
        pattern = self.patterns.get(str(answer["property"]))
        if pattern is None or pattern["kind"] != "property":
            return None
        matrix, names = answer["matrix"], answer["constraints"]
        if not (matrix is None or (type(matrix) is int and 40 <= matrix <= 45)):
            return None
        if type(answer["context"]) is not bool or type(answer["modifier"]) is not bool:
            return None
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            return None
        given = {name.lower() for name in names}
        return {"property": [int(answer["property"])], "object": [], "matrix": [] if matrix is None else [matrix],
                "context": [50] if answer["context"] else [],
                "constraint": sorted({self.number_of[name] for name in given if name in self.number_of}),
                "modifier": answer["modifier"], "names": {}, "invented": len(given - set(self.number_of))}

    def form(self, facts: dict) -> dict:
        """What a pattern set says about each decision.

        Args: facts: a pattern set, found or tagged.
        Returns: {"object": shape or None, "matrix": shape, "context", "modifier", "types"}.
        Raises: nothing.
        Side Effects: none. Deterministic.
        """
        shapes = [self.patterns.get(str(n), {}).get("shape", "single") for n in facts["object"]]
        shape = next((strong for strong in STRONG_SHAPES if strong in shapes), "single" if shapes else None)
        if not shapes:
            for number in facts["property"]:
                implied = self.patterns.get(str(number), {}).get("object_shape")
                if implied:
                    shape = "single" if implied == "variable" else implied
                    break
        matrix_shapes = [self.patterns.get(str(n), {}).get("shape", "single") for n in facts["matrix"]]
        matrix = "empty" if not matrix_shapes else ("source_target" if "source_target" in matrix_shapes else "single")
        types = {self.name_of.get(n) or facts["names"].get(str(n), str(n)) for n in facts["constraint"]}
        return {"object": shape, "matrix": matrix, "context": bool(facts["context"]),
                "modifier": bool(facts["modifier"]), "types": types}

    def mark(self, answer_text: str | None, entry: dict) -> dict:
        """Mark one answer against the experts' tags of its variable.

        Args: answer_text: the model's answer, or None for a failed call; entry: the tags.
        Returns: the nine columns as booleans, "tp", "fp", "fn" for the constraint types, whether
            the answer could be read and whether it states a matrix; an unreadable answer is wrong
            in everything.
        Raises: KeyError when the tags lack a field.
        Side Effects: none. Deterministic.
        """
        tagged = {"property": list(entry["property"]), "object": list(entry["object"]),
                  "matrix": list(entry["matrix"]), "context": list(entry["context"]),
                  "constraint": list(entry["constraint"]), "modifier": bool(entry["statistical_modifier"]),
                  "names": dict(entry.get("names", {}))}
        want = self.form(tagged)
        facts = self.found(read_answer(answer_text))
        if facts is None:
            return {**{column: False for column in COLUMNS}, "tp": 0, "fp": 0, "fn": len(want["types"]),
                    "parsed": False, "matrix_stated": False}
        got = self.form(facts)
        out = {"property": facts["property"] == tagged["property"], "object": got["object"] == want["object"],
               "matrix": got["matrix"] == want["matrix"], "context": got["context"] == want["context"],
               "modifier": got["modifier"] == want["modifier"],
               "constraint": got["types"] == want["types"] and not facts["invented"]}
        out["complete"] = all(out.values())
        out.update(matrix_pattern=facts["matrix"] == tagged["matrix"], both=out["matrix"] and out["constraint"],
                   tp=len(got["types"] & want["types"]), fp=len(got["types"] - want["types"]) + facts["invented"],
                   fn=len(want["types"] - got["types"]), parsed=True, matrix_stated=bool(facts["matrix"]))
        return out


def spread(values: list[float]) -> dict[str, float]:
    """The mean of per-variable figures with its two intervals over resampled variables.

    Args: values: one figure per variable, in the order of the sorted variable ids.
    Returns: {"mean", "low", "high", "strict_low", "strict_high"}.
    Raises: ZeroDivisionError for no values.
    Side Effects: none. Deterministic for the fixed seed.
    """
    rng = random.Random(SEED)
    drawn = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(DRAWS))
    return {"mean": sum(values) / len(values), "low": drawn[int(0.025 * DRAWS)],
            "high": drawn[int(0.975 * DRAWS) - 1], "strict_low": drawn[int(STRICT / 2 * DRAWS)],
            "strict_high": drawn[int((1 - STRICT / 2) * DRAWS) - 1]}


def main() -> int:
    """Recompute and compare.

    Args: none.
    Returns: 0 when every figure matches the stored summary, 1 otherwise.
    Raises: OSError when a stored file is missing.
    Side Effects: reads stored files; prints the comparison.
    """
    calls = kept_calls()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    tags = json.loads((DATA / "route_key.json").read_text(encoding="utf-8"))["variables"]
    by_group = groups()
    ids = sorted(v for members in by_group.values() for v in members)
    names = cells()
    book = Rulebook()
    problems: list[str] = []

    def check(name: str, mine: float | None, theirs: float | None) -> None:
        """Note a mismatch between a recomputed figure and the stored one.

        Args: name: what is compared; mine: recomputed; theirs: stored.
        Returns: nothing.
        Raises: nothing.
        Side Effects: appends to `problems`.
        """
        if mine is None or theirs is None:
            if mine is not theirs:
                problems.append(f"{name}: recomputed {mine}, stored {theirs}")
        elif not abs(mine - theirs) <= TOLERANCE:
            problems.append(f"{name}: recomputed {mine}, stored {theirs}")

    # What was sent and what came back.
    answered = [record for record in calls.values() if record["ok"]]
    print(f"calls kept {len(calls)}; answered {len(answered)}; cells {len({k[0] for k in calls})}; "
          f"variables {len({k[2] for k in calls})}; repetitions {sorted({k[1] for k in calls})}")
    wanted_keys = {(cell, r, v) for cell in names for r in range(1, REPETITIONS + 1) for v in ids}
    if set(calls) != wanted_keys:
        problems.append("the stored calls are not 16 cells x 102 variables x 2 repetitions")
    if len(answered) != len(calls):
        problems.append(f"{len(calls) - len(answered)} calls failed")
    hosts = Counter(record["host"] for record in answered)
    models = Counter(record["served_model"] for record in answered)
    thinking = sum(1 for record in answered if record.get("reasoning_chars") or record.get("reasoning_tokens"))
    print("hosts", dict(hosts), "models", dict(models), "answers with reasoning", thinking,
          "finish reasons", dict(Counter(record["finish_reason"] for record in answered)))
    if set(hosts) - {HOST} or set(models) - {MODEL}:
        problems.append(f"answers came from another host or model: {dict(hosts)} {dict(models)}")
    if thinking:
        problems.append(f"{thinking} answers carry reasoning")
    matched = 0
    for cell in names:
        hashes = defaultdict(set)
        for (name, _, variable_id), record in calls.items():
            if name == cell:
                hashes[variable_id].add(record["prompt_sha256"])
        if any(len(found) != 1 for found in hashes.values()) or set(hashes) != set(ids):
            problems.append(f"{cell}: a variable was sent different prompts in different repetitions")
            continue
        digest = text_hash("\n".join(next(iter(hashes[v])) for v in ids))[:16]
        if digest != PLAN_DIGESTS[cell]:
            problems.append(f"{cell}: the prompts sent ({digest}) are not the fingerprinted ones ({PLAN_DIGESTS[cell]})")
        else:
            matched += 1
    print("prompt digests match the record for", matched, "cells")
    written: dict[int, dict] = {}
    for row in rows(RULES):                 # the first usable text of each fold is the one the cells used
        usable = (row.get("ok") and row.get("finish_reason") == "stop" and (row.get("rules") or "").strip()
                  and not row.get("reasoning_chars") and not row.get("reasoning_tokens"))
        if usable and row["fold"] not in written:
            written[row["fold"]] = row
    if sorted(written) != list(range(FOLDS)):
        problems.append(f"rule texts: stored for folds {sorted(written)}, not for all six")
    for fold, row in sorted(written.items()):
        if row.get("host") != HOST or row.get("served_model") != MODEL:
            problems.append(f"rule text of fold {fold}: written by {row.get('host')} / {row.get('served_model')}")
        if str(row.get("prompt_sha256"))[:16] != RULE_PROMPT_DIGESTS.get(fold):
            problems.append(f"rule text of fold {fold}: not written from the fingerprinted prompt")
        if text_hash(row["rules"])[:16] != RULE_TEXT_DIGESTS.get(fold):
            problems.append(f"rule text of fold {fold}: not the fingerprinted text")
        if row.get("examples") != 85:
            problems.append(f"rule text of fold {fold}: written from {row.get('examples')} variables, not 85")
    print("rule texts match the record for", sum(1 for fold in written if not any(
        p.startswith(f"rule text of fold {fold}:") for p in problems)), "folds")
    fold_of = deal_folds(by_group, tags)
    digest = text_hash("\n".join(f"{v} {fold_of[v]}" for v in sorted(fold_of)))[:16]
    if digest != FOLDS_DIGEST:
        problems.append(f"the folds dealt again ({digest}) are not those of stage 0 ({FOLDS_DIGEST})")
    if summary.get("complete") is not True or summary.get("repetitions_used") != [1, 2]:
        problems.append(f"the summary is not of a complete run: complete {summary.get('complete')}, "
                        f"repetitions used {summary.get('repetitions_used')}")
    if list(summary["cells"]) != names:
        problems.append("the summary's cells are not the 16 of the plan, in their order")
    if summary.get("variables") != len(ids):
        problems.append(f"the summary counts {summary.get('variables')} variables, not {len(ids)}")
    if problems and set(calls) != wanted_keys:
        for problem in problems:
            print("MISMATCH", problem)
        print(f"{len(problems)} mismatches")
        return 1

    # Marks, and each cell's scoreboard.
    marked = {key: book.mark(record["answer"] if record["ok"] else None, tags[key[2]]) for key, record in calls.items()}
    reps = list(range(1, REPETITIONS + 1))

    def share(cell: str, column: str, members: list[str]) -> float:
        """How often one cell is right in one column over some variables, both repetitions.

        Args: cell: a cell; column: an outcome column; members: variables.
        Returns: the share.
        Raises: KeyError when a call is missing.
        Side Effects: none.
        """
        values = [bool(marked[(cell, r, v)][column]) for r in reps for v in members]
        return sum(values) / len(values) if values else 0.0

    def per_variable(group: list[str], column: str, variable_id: str) -> float:
        """One variable's share of right answers over some cells, both repetitions.

        Args: group: cell names; column: an outcome column; variable_id: the variable.
        Returns: the share.
        Raises: KeyError when a call is missing.
        Side Effects: none.
        """
        return sum(bool(marked[(cell, r, variable_id)][column]) for cell in group for r in reps) / (len(group) * len(reps))

    with_matrix = [v for v in ids if tags[v]["matrix"]]
    without_matrix = [v for v in ids if not tags[v]["matrix"]]
    spent = 0.0
    for cell in names:
        stored = summary["cells"].get(cell)
        if stored is None:
            continue
        for column in (*COLUMNS, "parsed"):
            check(f"{cell} {column}", share(cell, column, ids), stored[column])
        tp, fp, fn = (sum(marked[(cell, r, v)][name] for r in reps for v in ids) for name in ("tp", "fp", "fn"))
        precision, recall = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        check(f"{cell} constraint_precision", precision, stored["constraint_precision"])
        check(f"{cell} constraint_recall", recall, stored["constraint_recall"])
        check(f"{cell} constraint_f1", 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
              stored["constraint_f1"])
        check(f"{cell} matrix found where there is one", share(cell, "matrix", with_matrix),
              stored["matrix_found_where_there_is_one"])
        check(f"{cell} no matrix where there is none", share(cell, "matrix", without_matrix),
              stored["no_matrix_where_there_is_none"])
        for group, members in by_group.items():
            for column in PRIMARY:
                check(f"{cell} {group} {column}", share(cell, column, members), stored["by_group"][group][column])
        for column in PRIMARY:
            for field, value in spread([per_variable([cell], column, v) for v in ids]).items():
                check(f"{cell} interval {column} {field}", value, stored["intervals"][column][field])
        cost = sum(record.get("cost_usd") or 0.0 for key, record in calls.items() if key[0] == cell)
        spent += cost
        check(f"{cell} cost", cost, stored["cost_usd"])
        tokens = [record["prompt_tokens"] for key, record in calls.items() if key[0] == cell]
        check(f"{cell} prompt tokens", sum(tokens) / len(tokens), stored["prompt_tokens_mean"])
        print(f"{cell:10} matrix {share(cell, 'matrix', ids):.3f}  constraints {share(cell, 'constraint', ids):.3f}  "
              f"property {share(cell, 'property', ids):.3f}  complete {share(cell, 'complete', ids):.3f}  ${cost:.2f}")
    check("spent", spent, summary["spent_usd"])

    # What each kind of knowledge changes.
    if set(summary["effects"]) != set(EFFECT_COLUMNS):
        problems.append("the summary's effects are not for matrix, constraint, property, complete and both")
    for column in EFFECT_COLUMNS:
        if set(summary["effects"].get(column, {})) != set(FACTORS):
            problems.append(f"effects on {column}: not the four kinds of knowledge")
            continue
        for factor in FACTORS:
            on = [cell for cell in names if factor in cell.split("+")]
            off = [cell for cell in names if factor not in cell.split("+")]
            mine = spread([per_variable(on, column, v) - per_variable(off, column, v) for v in ids])
            for field, value in mine.items():
                check(f"effect of {factor} on {column} {field}", value, summary["effects"][column][factor][field])

    # The setting chosen fold by fold, scored on the fold itself.
    choice = {}
    for fold in range(FOLDS):
        others = [v for v in ids if fold_of[v] != fold]
        right = [sum(bool(marked[(cell, r, v)][column]) for column in PRIMARY for r in reps for v in others)
                 for cell in names]          # whole counts: equal cells stay equal
        choice[str(fold)] = names[right.index(max(right))]
    if choice != summary["chosen"].get("by_fold"):
        problems.append(f"chosen by fold: recomputed {choice}, stored {summary['chosen'].get('by_fold')}")
    if set(summary["chosen"].get("columns", {})) != set(CHOSEN_COLUMNS):
        problems.append("the summary's chosen setting is not scored on matrix, constraint, both and complete")
    for column in CHOSEN_COLUMNS:
        stored = summary["chosen"].get("columns", {}).get(column)
        if stored is None:
            continue
        own = [per_variable([choice[str(fold_of[v])]], column, v) for v in ids]
        base = [per_variable(["base"], column, v) for v in ids]
        check(f"chosen {column} share", sum(own) / len(own), stored["share"])
        check(f"chosen {column} base", sum(base) / len(base), stored["base"])
        for field, value in spread([a - b for a, b in zip(own, base, strict=True)]).items():
            check(f"chosen {column} against base {field}", value, stored["against_base"][field])
    same = [marked[(cell, 1, v)] == marked[(cell, 2, v)] for cell in names for v in ids]
    check("agreement", sum(same) / len(same), summary["agreement"])
    print("chosen by fold", choice, "| agreement", round(sum(same) / len(same), 4), f"| cost ${spent:.4f}")
    for problem in problems:
        print("MISMATCH", problem)
    print("all figures match" if not problems else f"{len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
