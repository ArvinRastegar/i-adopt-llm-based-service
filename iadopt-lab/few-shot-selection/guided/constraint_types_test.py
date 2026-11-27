#!/usr/bin/env python
"""Constraint types: state the ones the finder is sure of, and simulate a person for the rest.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/CONSTRAINT_TYPES_TEST_20261127.md` (private repository).

Arvin, 2026-11-27: "Yes, plan it" (a test aimed at the constraint types), then in the question
dialog "Yes, include it" (the simulated person) and "Up to $2".

Step 1 is free: it reads the ten named-first answers stored per variable today (the conventions
test's `plain` and the answer-form check's `named-first`, one prompt) and works out how far the
finder's agreement can be trusted (S1a), whether its restricting phrases cover the experts'
constraint values (S1b), how well a phrase typed by its most similar expert value would do (S1c), and
which signal should decide when to ask a person (S1d). It also derives Step 2's settings by the rules
fixed in the plan.

Step 2 is paid: today's best pipeline (stage 5's `rag-25` with the finder's matrix stated) as the
control, then with the confident types ("at least"), with a simulated person (the experts' set where
the signal flags), and with the experts' set everywhere. Ten runs.

    ./constraint_types_test.py --step1         # free; loads the scorer's local embedding model
    ./constraint_types_test.py --plan-only
    ./constraint_types_test.py --run-step2
    ./constraint_types_test.py --score-step2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import answer_form_check as afc  # noqa: E402
import ceiling  # noqa: E402
import conventions_test as ct  # noqa: E402
import draft_rag as dr  # noqa: E402
import engine  # noqa: E402
import hints  # noqa: E402
import knowledge as kn  # noqa: E402
import orclient  # noqa: E402
import rag_tuning as rt  # noqa: E402
import scoreboard  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

ARMS = ("best", "best+confident", "best+person", "best+experts")
SOURCES = ((ct.CALLS1, "plain"), (afc.CALLS_A, "named-first"))   # five runs each, one prompt
RUN_ONE = 5                          # position of the answer-form check's named-first run 1 among the ten
REPETITIONS, SEED = 10, 20261221
PURPOSE, ALLOWANCE_USD = "constraint-types", 2.0                  # Arvin, 2026-11-27: "Up to $2"
CONCURRENCY = 16
PRECISION, COVER, HALF = 0.90, 0.80, 0.5
SIGNALS = ("R", "N", "R or N")
STEP1 = orclient.OUTPUT / "constraint-types-step1.json"
CALLS = orclient.OUTPUT / "constraint-types-calls.jsonl"
SCORES = orclient.OUTPUT / "constraint-types-scores.jsonl"
SUMMARY = orclient.OUTPUT / "constraint-types-summary.json"


# --- reading the stored answers -------------------------------------------------------------------

def stored_answers(ctx: Mapping[str, Any]) -> dict[str, list[str | None]]:
    """The ten named-first answers of every corpus variable, in a fixed order.

    Args: ctx: from `answer_form_check.load`.
    Returns: {variable: [plain runs 1 to 5, then named-first runs 1 to 5]}; None for a failed call.
    Raises: KeyError when an answer is missing.
    Side Effects: reads the two calls files.
    """
    out: dict[str, list[str | None]] = {v: [] for v in ctx["ids"]}
    for path, arm in SOURCES:
        calls = engine.latest(path)
        for v in ctx["ids"]:
            for run in range(1, 6):
                record = calls[(arm, run, v)]
                out[v].append(record.get("answer") if record.get("ok") else None)
    return out


def read(ctx: Mapping[str, Any], answer: str | None) -> tuple[frozenset[str] | None, list[str]]:
    """One answer's constraint types and restricting phrases.

    Args: ctx: from `answer_form_check.load`; answer: the reply or None.
    Returns: (the set of type names, or None for an answer that is not the requested JSON; the
        "restrictions" texts, [] when missing).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    from iadopt_lab.generation.extractor import extract_json

    proper, _ = kn.requested(ctx, answer)
    facts = walk.facts_from_direct(ctx, answer or "") if proper else None
    types = frozenset(hints.expected_form(facts, ctx["patterns"], ctx["types"])["constraint_types"]) if facts else None
    extracted = extract_json(answer or "")
    data = extracted.candidate if extracted.success and isinstance(extracted.candidate, dict) else {}
    phrases = data.get("restrictions")
    return types, [p for p in phrases if isinstance(p, str) and p.strip()] if isinstance(phrases, list) else []


def gold_types(ctx: Mapping[str, Any], v: str) -> frozenset[str]:
    """The experts' set of constraint types of a variable, by the completed table's names.

    Args: ctx: from `answer_form_check.load`; v: the variable.
    Returns: the set.
    Raises: KeyError for an unknown variable.
    Side Effects: none. Deterministic.
    """
    return frozenset(hints.expected_form(hints.facts_from_key(ctx["key"][v]), ctx["patterns"], ctx["types"])["constraint_types"])


def pool(ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The experts' constraint values with their types, as S1c's pool.

    Args: ctx: from `answer_form_check.load`.
    Returns: [{"variable", "position", "value", "type"}] in variable then position order: a label's
        type is the word before its colon when that is one of its variable's types, else its
        variable's only type no other of its labels starts with; labels with neither are left out.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out = []
    for v in ctx["ids"]:
        types = gold_types(ctx, v)
        labels = [c["label"] for c in ctx["records"][v]["gold"]["hasConstraint"]]
        prefixes = [label.split(":", 1)[0].strip().lower() if ":" in label else "" for label in labels]
        remaining = sorted(types - set(prefixes))
        for position, (label, prefix) in enumerate(zip(labels, prefixes, strict=True)):
            kind = prefix if prefix in types else remaining[0] if len(remaining) == 1 else None
            if kind is not None:
                out.append({"variable": v, "position": position,
                            "value": (label.split(":", 1)[1] if ":" in label else label).strip(), "type": kind})
    return out


def cosine(similar: Callable[[str, str], float]) -> Callable[[str, str], float]:
    """The scorer's policy for two texts: 1.0 when equal after lower case and trimming, else the cosine.

    Args: similar: the embedding backend.
    Returns: a similarity of two texts.
    Raises: nothing itself.
    Side Effects: none beyond the backend's cache.
    """
    def score(first: str, second: str) -> float:
        """Similarity of two texts.

        Args: first, second: texts.
        Returns: the value.
        Raises: as the backend.
        Side Effects: the backend's cache.
        """
        a, b = first.lower().strip(), second.lower().strip()
        return 1.0 if a == b else float(similar(a, b))

    return score


def nearest(phrase: str, candidates: Sequence[Mapping[str, Any]], score: Callable[[str, str], float]) -> dict[str, Any]:
    """The most similar candidate value to a phrase; ties go to the earliest candidate.

    Args: phrase: a restricting phrase; candidates: pool entries, in pool order; score: a similarity.
    Returns: {"variable", "position", "type", "similarity"}.
    Raises: ValueError for no candidates.
    Side Effects: calls `score`.
    """
    best, best_value = None, None
    for entry in candidates:
        value = score(entry["value"], phrase)
        if best is None or value > best_value:
            best, best_value = entry, value
    if best is None:
        raise ValueError("no candidates")
    return {"variable": best["variable"], "position": best["position"], "type": best["type"], "similarity": best_value}


# --- Step 1 ---------------------------------------------------------------------------------------

def majority(sets: Sequence[frozenset[str] | None]) -> frozenset[str] | None:
    """The most frequent set; ties go to the earliest.

    Args: sets: one set (or None) per answer.
    Returns: the set.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    tally = Counter(sets)
    top = max(tally.values())
    return next(s for s in sets if tally[s] == top)


def precision_recall(k: Mapping[tuple[str, str], int], gold: Mapping[str, frozenset[str]], ids: Sequence[str],
                     threshold: int) -> dict[str, Any]:
    """S1a at one threshold.

    Args: k: {(variable, type): answers naming it}; gold: experts' sets; ids: the variables counted;
        threshold: K.
    Returns: {"named", "right", "precision", "recall"}; precision None without a named pair.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    members = set(ids)
    named = [(v, t) for (v, t), count in k.items() if v in members and count >= threshold]
    right = sum(1 for v, t in named if t in gold[v])
    total = sum(len(gold[v]) for v in members)
    return {"named": len(named), "right": right, "precision": right / len(named) if named else None,
            "recall": right / total if total else None}


def choose_signal(rows: Mapping[str, Mapping[str, float]]) -> str:
    """The plan's rule for the signal of Step 2.

    Args: rows: {signal: {"flagged" share, "accuracy"}} for `SIGNALS`.
    Returns: the signal with the highest accuracy among those flagging at most half of the variables;
        without one, the signal flagging fewest. Ties go to the order of `SIGNALS`.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    allowed = [s for s in SIGNALS if rows[s]["flagged"] <= HALF]
    if allowed:
        return max(allowed, key=lambda s: (rows[s]["accuracy"], -SIGNALS.index(s)))
    return min(SIGNALS, key=lambda s: (rows[s]["flagged"], SIGNALS.index(s)))


def thresholds(ctx: Mapping[str, Any], k: Mapping[tuple[str, str], int], gold: Mapping[str, frozenset[str]]) -> dict[int, int | None]:
    """The plan's confident-type threshold of each fold, chosen on the other five folds.

    Args: ctx: from `answer_form_check.load`; k: as in `precision_recall`; gold: experts' sets.
    Returns: {fold: the smallest K from 2 to 10 with precision at least 0.90, or None}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    out: dict[int, int | None] = {}
    for fold in range(scoreboard.FOLDS):
        others = [v for v in ctx["ids"] if ctx["fold_of"][v] != fold]
        out[fold] = next((K for K in range(2, 11) if (precision_recall(k, gold, others, K)["precision"] or 0) >= PRECISION),
                         None)
    return out


def step1(ctx: Mapping[str, Any], score: Callable[[str, str], float]) -> dict[str, Any]:
    """Step 1 and the settings it fixes for Step 2.

    Args: ctx: from `answer_form_check.load`; score: a similarity of two texts (`cosine`).
    Returns: the Step 1 summary: S1a to S1d, the folds' thresholds, the confident types, the chosen
        signal and its flagged variables, and every phrase's nearest value (for the recomputation).
    Raises: KeyError when a stored answer is missing.
    Side Effects: calls `score`.
    """
    ids = ctx["ids"]
    answers = stored_answers(ctx)
    gold = {v: gold_types(ctx, v) for v in ids}
    read_all = {v: [read(ctx, a) for a in answers[v]] for v in ids}
    entries = pool(ctx)
    k: dict[tuple[str, str], int] = Counter()
    for v in ids:
        for types, _ in read_all[v]:
            for t in types or ():
                k[(v, t)] += 1
    s1a = {str(K): precision_recall(k, gold, ids, K) for K in range(1, 11)}
    agree = [v for v in ids if len({types for types, _ in read_all[v]}) == 1 and read_all[v][0][0] is not None]
    others = [v for v in ids if v not in agree]
    agreement = {"agreeing": len(agree), "agreeing_right": sum(read_all[v][0][0] == gold[v] for v in agree),
                 "other_answers": 10 * len(others),
                 "other_answers_right": sum(types == gold[v] for v in others for types, _ in read_all[v])}
    covered = total = 0
    for v in ids:
        values = [(c["label"].split(":", 1)[1] if ":" in c["label"] else c["label"]).strip()
                  for c in ctx["records"][v]["gold"]["hasConstraint"]]
        for _, phrases in read_all[v]:
            for value in values:
                total += 1
                covered += any(score(value, p) >= COVER for p in phrases)
    s1b = {"values": total, "covered": covered, "share": covered / total if total else None}
    nearest_rows: dict[str, list[list[dict[str, Any]]]] = {}
    model_sets: dict[str, list[frozenset[str]]] = {}
    for v in ids:
        candidates = [e for e in entries if ctx["fold_of"][e["variable"]] != ctx["fold_of"][v]]
        nearest_rows[v] = [[dict(nearest(p, candidates, score), phrase=p) for p in phrases] for _, phrases in read_all[v]]
        model_sets[v] = [frozenset(row["type"] for row in rows) for rows in nearest_rows[v]]
    s1c: dict[str, Any] = {}
    for name, sets in (("finder", {v: [types for types, _ in read_all[v]] for v in ids}), ("no model", model_sets)):
        right = sum(s == gold[v] for v in ids for s in sets[v])
        tp = sum(len((s or frozenset()) & gold[v]) for v in ids for s in sets[v])
        fp = sum(len((s or frozenset()) - gold[v]) for v in ids for s in sets[v])
        fn = sum(len(gold[v] - (s or frozenset())) for v in ids for s in sets[v])
        s1c[name] = {"answers": 10 * len(ids), "sets_right": right, "share_right": right / (10 * len(ids)),
                     "type_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    flags = {"R": {v for v in ids if len({types for types, _ in read_all[v]}) > 1},
             "N": {v for v in ids if majority([types for types, _ in read_all[v]]) != majority(model_sets[v])}}
    flags["R or N"] = flags["R"] | flags["N"]
    s1d = {}
    for signal in SIGNALS:
        flagged = flags[signal]
        right = sum(1 for v in ids if v in flagged or read_all[v][RUN_ONE][0] == gold[v])
        s1d[signal] = {"flagged": len(flagged) / len(ids), "flagged_count": len(flagged), "accuracy": right / len(ids)}
    chosen = choose_signal(s1d)
    by_fold = thresholds(ctx, k, gold)
    confident = {v: sorted(t for t in {t for (u, t) in k if u == v}
                           if by_fold[ctx["fold_of"][v]] is not None and k[(v, t)] >= by_fold[ctx["fold_of"][v]])
                 for v in ids}
    return {"generated_at": ft.now(), "answers_per_variable": 10, "s1a": s1a, "agreement": agreement, "s1b": s1b,
            "s1c": s1c, "s1d": s1d, "signal": chosen, "flagged": sorted(flags[chosen]),
            "thresholds": {str(f): K for f, K in by_fold.items()}, "confident": confident,
            "confident_arm_runs": any(K is not None for K in by_fold.values()),
            "pool_size": len(entries), "nearest": nearest_rows}


# --- Step 2 ---------------------------------------------------------------------------------------

def with_types(ctx: Mapping[str, Any], facts: Mapping[str, Any], names: Sequence[str] | None = None,
               numbers: Sequence[int] | None = None) -> dict[str, Any]:
    """The finder's facts with another set of constraint types.

    Args: ctx: from `answer_form_check.load`; facts: the finder's facts; names: type names, or
        numbers: type numbers (the experts' tags).
    Returns: a copy whose "constraint" holds the given types.
    Raises: KeyError for a name the completed table lacks.
    Side Effects: none. Deterministic.
    """
    by_name = {q["name"]: q["number"] for q in ctx["constraint_questions"]}
    chosen = list(numbers) if numbers is not None else [by_name[n] for n in names or ()]
    return dict(facts, constraint=sorted(chosen))


def prompt2(ctx: Mapping[str, Any], world: Mapping[str, Any], job: engine.Job, found: Mapping[str, Any],
            settings: Mapping[str, Any]) -> str:
    """The prompt of one Step 2 call.

    Args: ctx: from `answer_form_check.load`; world: from `rag_tuning.corpus_world`; job: (arm, run,
        variable); found: the finder's facts (run 1); settings: from `step1`.
    Returns: today's best pipeline's prompt (`best`), with the confident types as "at least",
        the experts' exact set for flagged variables, or the experts' exact set for all. Without
        finder facts, or a variable without confident types in `best+confident`, the control prompt.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    arm, run, v = job
    control = afc.prompt_b(ctx, world, ("rag-25+finder-matrix", run, v), found)
    facts = found.get(v)
    if arm == "best" or facts is None:
        if arm not in ARMS:
            raise KeyError(arm)
        return control
    experts = with_types(ctx, facts, numbers=ctx["key"][v]["constraint"])
    confident = settings["confident"][v]
    if arm == "best+experts" or (arm == "best+person" and v in set(settings["flagged"])):
        stmt = dr.statement(experts, {"matrix", "constraint"})
    elif arm in ("best+confident", "best+person"):
        if not confident:
            return control
        stmt = dr.statement(with_types(ctx, facts, names=confident), {"matrix", "constraint"}, at_least=True)
    else:
        raise KeyError(arm)
    ids = rt.examples(world, "rag-25", v)
    inputs = dict(world["base_inputs"](v), examples=[world["record"](u) for u in ids])
    return dr.fill_prompt(inputs, world["definition"](v), dr.render(ctx, stmt))


def arms_to_run(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """The Step 2 arms by the plan: the confident-types arm only when some fold has a threshold.

    Args: settings: from `step1`.
    Returns: the arms.
    Raises: KeyError when the settings lack the flag.
    Side Effects: none. Deterministic.
    """
    return ARMS if settings["confident_arm_runs"] else tuple(a for a in ARMS if a != "best+confident")


def summarise2(ctx2: Mapping[str, Any], scores: Mapping[engine.Job, Mapping[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    """Step 2: each arm's Close F1 per run and by part, and each arm against `best`.

    Args: ctx2: from `rag_tuning.load`; scores: from `engine.score`; arms: the arms run.
    Returns: the summary.
    Raises: KeyError when a score is missing.
    Side Effects: none. Deterministic.
    """
    renamed = {(("rag-25" if k[0] == "best" else k[0]), k[1], k[2]): s for k, s in scores.items()}
    out = ct.summarise2(ctx2, renamed, ["rag-25" if a == "best" else a for a in arms])
    out["arms"] = {("best" if a == "rag-25" else a): row for a, row in out.get("arms", {}).items()}
    out["by_part"] = {("best" if a == "rag-25" else a): row for a, row in out.get("by_part", {}).items()}
    out["against_best"] = out.pop("against_rag25", {})
    return out


def fingerprints(ctx: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the code, the Step 1 summary and every Step 2 prompt.

    Args: ctx: from `answer_form_check.load`; settings: from `step1`.
    Returns: hex digests (16 characters) and mean prompt lengths.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files and stage 5's similarity table.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "constraint_types_test.py", "answer_form_check.py", "conventions_test.py", "draft_rag.py", "rag_tuning.py")}
    out["step1"] = ceiling.file_sha256(STEP1)[:16]
    world = rt.corpus_world(rt.load(), None)
    found = afc.finder_facts(ctx, engine.latest(afc.CALLS_A))
    out["prompts"] = {}
    for arm in arms_to_run(settings):
        prompts = [prompt2(ctx, world, (arm, 1, v), found, settings) for v in ctx["ids"]]
        out["prompts"][arm] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                               "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: arguments; `sys.argv` when None.
    Returns: 0 on success; 1 when a stop rule ended the run or the summary is not complete.
    Raises: SystemExit on bad arguments or a step run out of order.
    Side Effects: --step1 loads the local model and writes `STEP1`; --run-step2 calls the provider.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--step1", "--plan-only", "--run-step2", "--score-step2"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args(argv)
    ctx = afc.load()
    if args.step1:
        similar, _ = dr.local_similarity()
        out = step1(ctx, cosine(similar))
        STEP1.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        brief = {key: out[key] for key in ("agreement", "s1b", "s1c", "s1d", "signal", "thresholds", "confident_arm_runs")}
        print(json.dumps(brief, indent=1))
        print("S1a precision/recall by K:", {K: (round(r["precision"], 3) if r["precision"] is not None else None,
                                                 round(r["recall"], 3)) for K, r in out["s1a"].items()})
        print("flagged:", len(out["flagged"]), "| variables with confident types:", sum(1 for t in out["confident"].values() if t))
        return 0
    if not STEP1.exists():
        raise SystemExit("run --step1 first")
    settings = json.loads(STEP1.read_text(encoding="utf-8"))
    if args.plan_only:
        print(json.dumps({"arms": arms_to_run(settings), "repetitions": REPETITIONS,
                          "calls": len(ctx["ids"]) * REPETITIONS * len(arms_to_run(settings)),
                          "allowance_usd": ALLOWANCE_USD, "model": orclient.MODEL, "host": orclient.HOST["name"],
                          "questions": len(settings["flagged"]), **fingerprints(ctx, settings)}, indent=2))
        return 0
    if args.score_step2:
        ctx2 = rt.load()
        scores = engine.score(CALLS, SCORES, ctx2["records"], ctx2["inputs"]["schema"])
        summary = summarise2(ctx2, scores, arms_to_run(settings))
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for arm, row in summary.get("arms", {}).items():
            print(f"{arm:16} F1 {row['f1_mean']:.4f} sd {row['f1_sd']:.4f} invalid {row['invalid']}")
        for arm, row in summary.get("against_best", {}).items():
            print(f"{arm} - best: {row['gain']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open the transport, run Step 2, always close the client.

        Args: none.
        Returns: 1 when a stop rule ended the run, else 0.
        Raises: SystemExit when the key is missing or the ledger is held.
        Side Effects: provider calls, the ledger and the calls file.
        """
        spend = orclient.Spend(purpose=PURPOSE, allowance=ALLOWANCE_USD)
        guard = orclient.Guard(spend)
        transport, client = orclient.open_transport(spend, PURPOSE, concurrency=args.concurrency)
        try:
            world = rt.corpus_world(rt.load(), None)
            found = afc.finder_facts(ctx, engine.latest(afc.CALLS_A))
            jobs = engine.schedule(arms_to_run(settings), ctx["ids"], REPETITIONS, SEED)
            print(json.dumps(await orclient.run_jobs(jobs, lambda job: prompt2(ctx, world, job, found, settings), transport,
                                                     CALLS, guard, concurrency=args.concurrency, request=dict(engine.REQUEST),
                                                     max_calls=args.max_calls), indent=2))
            if guard.reason:
                print("stopped:", guard.reason)
            return 1 if guard.reason else 0
        finally:
            await orclient.close_transport(transport, client)

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
