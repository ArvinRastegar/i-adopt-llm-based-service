#!/usr/bin/env python
"""A model stands in for the user who answers the draft's plain category questions.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.
Plan and results: `docs/tracking/STAND_IN_USER_TEST_20261127.md` (private repository).

Arvin, 2026-11-27: "Yes, plan it" (a cheap stand-in test of the question to the user), then in the
question dialog "Same Qwen3.8-27B" and "Up to $1.50"; then "I also updated the psnc api key so you can
use it again instead of openrouter". Every call goes to PSNC (`walk.chat_transport`, model
`Qwen3.8-27B`, reasoning off).

1. The stand-in: for each corpus variable, the model gets the description, the phrases the finder
   listed (the answer-form check's `named-first` run 1) and the draft's first two screens with the
   fifteen plain categories (`data/question_categories_v1.json`); no I-ADOPT examples and no
   proposed categories. One answer per variable, temperature 0.
2. The typing: a restriction whose category has one type gets it; otherwise a short prompt picks one
   of the category's types.
3. The decomposition: today's best pipeline (stage 5's `rag-25` with the finder's matrix) as `best`,
   `best+stand-in` (the stand-in's types from both screens stated exactly for every variable),
   `best+screen-1` (only the types of the phrases the finder listed, screen 1) and `best+experts` (the
   experts' types for every variable). Ten runs. The screen-1 arm was added after the dev check, where
   screen 2 added only restrictions the experts do not use; Arvin, 2026-11-27: "Add screen-1-only arm".

    ./stand_in_test.py --plan-only
    ./stand_in_test.py --dev             # both new prompts on invented variables only
    ./stand_in_test.py --run-answers     # the stand-in, then the typing
    ./stand_in_test.py --score-answers
    ./stand_in_test.py --run-decomposition
    ./stand_in_test.py --score-decomposition
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "example_selection", HERE.parent, HERE.parents[1] / "src"):
    sys.path.insert(0, str(_path))

import analysis  # noqa: E402
import answer_form_check as afc  # noqa: E402
import ceiling  # noqa: E402
import constraint_types_test as cy  # noqa: E402
import conventions_test as ct  # noqa: E402
import draft_rag as dr  # noqa: E402
import engine  # noqa: E402
import knowledge as kn  # noqa: E402
import rag_tuning as rt  # noqa: E402
import rulebook  # noqa: E402
import walk  # noqa: E402
from engine import ft  # noqa: E402

CATEGORIES = rulebook.DATA / "question_categories_v1.json"
NOT_A_RESTRICTION = "not a restriction"
ARMS = ("best", "best+stand-in", "best+screen-1", "best+experts")
STAND_IN_ARMS = {"best+stand-in": ("listed", "added"), "best+screen-1": ("listed",)}   # the screens each arm uses
REPETITIONS, SEED = 10, 20261241
ANSWER_REQUEST = {"stream": False, "temperature": 0.0, "top_p": 1.0, "max_tokens": 600,
                  "chat_template_kwargs": {"enable_thinking": False}}
TYPE_REQUEST = dict(ANSWER_REQUEST, max_tokens=20)
CONCURRENCY_ANSWERS, CONCURRENCY = 4, 16
FAILURES_IN_A_ROW = 10                    # a budget block (HTTP 429, 2026-11-19) fails every later call
OUT = HERE.parent / "output" / "retrieval"
ANSWERS = OUT / "stand-in-answers-calls.jsonl"
TYPING = OUT / "stand-in-typing-calls.jsonl"
DEV = OUT / "stand-in-dev-calls.jsonl"
SUMMARY_A = OUT / "stand-in-answers-summary.json"
CALLS = OUT / "stand-in-decomposition-calls.jsonl"
SCORES = OUT / "stand-in-decomposition-scores.jsonl"
SUMMARY = OUT / "stand-in-decomposition-summary.json"

# The draft's screens 1 and 2 (`docs/tracking/USER_QUESTION_DRAFT_20261127.md`), without proposed kinds.
SCREEN_1 = ("Your description says what is measured. Some of its words narrow that down, for example dissolved, "
            "at 25 °C, per hour or near the coast. We found these in your description:")
SCREEN_1_ASK = ('For each, choose what kind of restriction it is. If a phrase does not narrow what is measured, '
                'choose "not a restriction".')
SCREEN_1_NONE = "Your description says what is measured. We found no words in it that narrow that down."
SCREEN_2 = ("Is what you measure narrowed in any other way that we did not list? For example a depth, a time of day, "
            "a method or standard, a cause, or who did the measurement. Write it in your own words and pick what "
            "kind it is.")


class Guard(engine.Guard):
    """The study's stop rules, plus a stop after `FAILURES_IN_A_ROW` failed calls in a row."""

    def __init__(self) -> None:
        """Start with no stop reason and no counts.

        Args: none.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        super().__init__()
        self.in_a_row = 0

    def record(self, result: Mapping[str, Any]) -> None:
        """Apply the stop rules to one finished call.

        Args: result: the stored call record.
        Returns: nothing.
        Raises: nothing.
        Side Effects: updates counts; may set the stop reason.
        """
        super().record(result)
        self.in_a_row = 0 if result.get("ok") else self.in_a_row + 1
        if self.in_a_row >= FAILURES_IN_A_ROW:
            self.reason = self.reason or f"{self.in_a_row} calls failed in a row (last: HTTP {result.get('status')})"


def categories() -> list[dict[str, Any]]:
    """The fifteen plain categories.

    Args: none.
    Returns: [{"name", "help", "types"}] in the file's order.
    Raises: OSError when the file is missing.
    Side Effects: reads the data file.
    """
    return json.loads(CATEGORIES.read_text(encoding="utf-8"))["categories"]


def stand_in_prompt(definition: str, phrases: Sequence[str], cats: Sequence[Mapping[str, Any]]) -> str:
    """The question to the user, as the stand-in sees it: the draft's first two screens.

    Args: definition: the variable's description; phrases: the restrictions the finder listed;
        cats: from `categories`.
    Returns: the prompt: the description, the two screens (no proposed kinds) and the kinds with
        their help texts.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    lines = ["You wrote this description of a variable that you measure or collect:", f'"{definition}"', "",
             "A data service asks you two short questions so that it can record your variable correctly.", ""]
    if phrases:
        lines += [f"Question 1. {SCREEN_1}", *[f'- "{p}"' for p in phrases], SCREEN_1_ASK]
    else:
        lines.append(f"Question 1. {SCREEN_1_NONE}")
    lines += ["", f"Question 2. {SCREEN_2} If there is nothing else, give nothing.", "", "Kinds of restriction:"]
    lines += [f"- {c['name']}: {c['help']}" for c in cats]
    lines += [f"- {NOT_A_RESTRICTION}: this phrase does not narrow what is measured.", "",
              'Answer with one JSON object only: {"phrases": [{"phrase": "...", "kind": "..."}], '
              '"added": [{"phrase": "...", "kind": "..."}]}. "phrases" answers question 1, one entry per phrase '
              'we found; "added" answers question 2. Write each kind exactly as in the list above.']
    return "\n".join(lines)


def read_answer(answer: str | None) -> list[dict[str, str]] | None:
    """Every item of a stand-in answer.

    Args: answer: the reply or None.
    Returns: [{"phrase", "kind" (lower case, trimmed), "source" ("listed" or "added")}] for every
        item with a phrase, whatever its kind; None when the reply is not a JSON object with
        "phrases" or "added" as lists (null counts as empty).
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    from iadopt_lab.generation.extractor import extract_json

    extracted = extract_json(answer or "")
    data = extracted.candidate if extracted.success and isinstance(extracted.candidate, dict) else None
    if data is None or not {"phrases", "added"} & set(data):
        return None
    lists = {source: data.get(source) or [] for source in ("phrases", "added")}
    if not all(isinstance(items, list) for items in lists.values()):
        return None
    out = []
    for source, items in lists.items():
        for item in items:
            phrase = item.get("phrase") if isinstance(item, dict) else None
            if isinstance(phrase, str) and phrase.strip():
                out.append({"phrase": phrase.strip(), "kind": str(item.get("kind") or "").strip().lower(),
                            "source": "listed" if source == "phrases" else "added"})
    return out


def type_prompt(definition: str, phrase: str, category: Mapping[str, Any], definitions: Mapping[str, str]) -> str:
    """The short prompt that picks one type within a category.

    Args: definition: the variable's description; phrase: the restriction; category: its category;
        definitions: {type name: definition} of the completed table.
    Returns: the prompt.
    Raises: KeyError for a type without a definition.
    Side Effects: none. Deterministic.
    """
    lines = [f'A variable is described as: "{definition}"',
             f'One phrase in it, "{phrase}", is a restriction of this kind: {category["name"]} ({category["help"]})',
             "Which of these more precise types is it?"]
    lines += [f"- {t}: {definitions[t]}" for t in category["types"]]
    lines.append("Answer with the type name only.")
    return "\n".join(lines)


def read_type(answer: str | None, category: Mapping[str, Any]) -> str | None:
    """The type a typing answer names.

    Args: answer: the reply or None; category: the category asked about.
    Returns: the category's type the answer names (the whole answer, lower case, without quotes,
        asterisks or full stops; else the longest type name it contains as words); None otherwise.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    text = re.sub(r"[\"'`*.]", "", (answer or "").strip().lower()).strip()
    if text in category["types"]:
        return text
    inside = [t for t in category["types"] if re.search(rf"\b{re.escape(t)}\b", text)]
    return max(inside, key=len) if inside else None


def load() -> dict[str, Any]:
    """The context: the answer-form check's, the categories, the type definitions and the finder's phrases.

    Args: none.
    Returns: the context with "categories", "definitions" ({type: definition}) and "phrases"
        ({variable: the restrictions of the finder's named-first run-1 answer}).
    Raises: as the loaders.
    Side Effects: reads local files.
    """
    ctx = afc.load()
    ctx["categories"] = categories()
    ctx["definitions"] = {q["name"]: q["definition"] for q in ctx["constraint_questions"]}
    calls = engine.latest(afc.CALLS_A)
    ctx["phrases"] = {v: cy.read(ctx, calls[("named-first", 1, v)].get("answer"))[1] for v in ctx["ids"]}
    return ctx


def typing_jobs(cats: Sequence[Mapping[str, Any]], answers: Mapping[str, list[dict[str, str]] | None]) -> dict[str, dict[str, Any]]:
    """The typing calls the stand-in's answers need: one per restriction whose category has several types.

    Args: cats: from `categories`; answers: {variable: from `read_answer`}.
    Returns: {target id ("<variable>#<position in the answer>"): {"variable", "phrase", "category"}}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    by_name = {c["name"]: c for c in cats}
    out = {}
    for v, items in sorted(answers.items()):
        for position, item in enumerate(items or []):
            category = by_name.get(item["kind"])
            if category and len(category["types"]) > 1:
                out[f"{v}#{position}"] = {"variable": v, "phrase": item["phrase"], "category": category}
    return out


def stand_in_types(cats: Sequence[Mapping[str, Any]], answers: Mapping[str, list[dict[str, str]] | None],
                   typed: Mapping[str, str | None], sources: Sequence[str] = ("listed", "added")) -> dict[str, list[str] | None]:
    """Each variable's constraint types from the stand-in's answers.

    Args: cats: from `categories`; answers: {variable: from `read_answer`}; typed: {target id: the
        type its typing call named, or None}; sources: the screens to use ("listed" is screen 1,
        "added" screen 2).
    Returns: {variable: sorted types, or None when its stand-in answer was unusable}. An item of
        another screen, an item marked "not a restriction" or with an unknown kind, and a
        restriction whose typing gave no type add nothing.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    by_name = {c["name"]: c for c in cats}
    out: dict[str, list[str] | None] = {}
    for v, items in answers.items():
        if items is None:
            out[v] = None
            continue
        types = set()
        for position, item in enumerate(items):
            category = by_name.get(item["kind"])
            if category is None or item["source"] not in sources:
                continue
            chosen = category["types"][0] if len(category["types"]) == 1 else typed.get(f"{v}#{position}")
            if chosen:
                types.add(chosen)
        out[v] = sorted(types)
    return out


def prompt2(ctx: Mapping[str, Any], world: Mapping[str, Any], job: engine.Job, found: Mapping[str, Any],
            types: Mapping[str, Mapping[str, list[str] | None]]) -> str:
    """The prompt of one decomposition call.

    Args: ctx: from `load`; world: from `rag_tuning.corpus_world`; job: (arm, run, variable); found: the
        finder's facts (run 1); types: {stand-in arm: from `stand_in_types` with the arm's screens}.
    Returns: today's best pipeline's prompt (the constraint-types test's `best`); with the stand-in's
        types or the experts' types stated exactly with the finder's matrix. A variable without
        finder facts, or (in a stand-in arm) with an unusable stand-in answer, gets the control.
    Raises: KeyError for an unknown arm.
    Side Effects: none. Deterministic.
    """
    arm, run, v = job
    if arm not in ARMS:
        raise KeyError(arm)
    control = afc.prompt_b(ctx, world, ("rag-25+finder-matrix", run, v), found)
    facts = found.get(v)
    if arm == "best" or facts is None or (arm in STAND_IN_ARMS and types[arm].get(v) is None):
        return control
    if arm == "best+experts":
        chosen = cy.with_types(ctx, facts, numbers=ctx["key"][v]["constraint"])
    else:
        chosen = cy.with_types(ctx, facts, names=types[arm][v])
    ids = rt.examples(world, "rag-25", v)
    inputs = dict(world["base_inputs"](v), examples=[world["record"](u) for u in ids])
    return dr.fill_prompt(inputs, world["definition"](v), dr.render(ctx, dr.statement(chosen, {"matrix", "constraint"})))


def stored_answers(ctx: Mapping[str, Any], path: Path = ANSWERS) -> dict[str, list[dict[str, str]] | None]:
    """The stand-in's stored answers, read.

    Args: ctx: from `load`; path: the calls file.
    Returns: {variable: from `read_answer`}; None for a variable without an answered call.
    Raises: nothing.
    Side Effects: reads the file.
    """
    calls = engine.latest(path)
    return {v: read_answer(calls[("stand-in", 1, v)].get("answer")) if calls.get(("stand-in", 1, v), {}).get("ok")
            else None for v in ctx["ids"]}


def stored_types(ctx: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str | None]]:
    """The stand-in's answers, the typing they needed and the type each typing call named.

    Args: ctx: from `load`.
    Returns: (answers, typing jobs, {target id: type or None}).
    Raises: nothing.
    Side Effects: reads the two calls files.
    """
    answers = stored_answers(ctx)
    jobs = typing_jobs(ctx["categories"], answers)
    calls = engine.latest(TYPING)
    typed = {t: read_type(calls[("typing", 1, t)].get("answer") if calls.get(("typing", 1, t), {}).get("ok") else None,
                          job["category"]) for t, job in jobs.items()}
    return answers, jobs, typed


def summarise_answers(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """The stand-in (both screens, screen 1 alone) and the finder against the experts.

    Args: ctx: from `load`.
    Returns: counts of items and typing calls; for each of the two stand-in arms and the finder (its
        named-first run 1) the share of the variables whose category set and whose type set equal
        the experts', and the pooled type F1; and "types" ({stand-in arm: {variable: its types}}).
    Raises: KeyError when a finder answer is missing.
    Side Effects: reads the stored calls.
    """
    answers, jobs, typed = stored_types(ctx)
    types = {arm: stand_in_types(ctx["categories"], answers, typed, sources) for arm, sources in STAND_IN_ARMS.items()}
    names = {c["name"] for c in ctx["categories"]}
    of = {t: c["name"] for c in ctx["categories"] for t in c["types"]}
    items = [i for a in answers.values() for i in a or []]
    out: dict[str, Any] = {
        "generated_at": ft.now(), "variables": len(ctx["ids"]), "usable": sum(a is not None for a in answers.values()),
        "phrases_shown": sum(len(ctx["phrases"][v]) for v in ctx["ids"]),
        "listed": sum(i["source"] == "listed" for i in items),
        "listed_not_a_restriction": sum(i["source"] == "listed" and i["kind"] == NOT_A_RESTRICTION for i in items),
        "added": sum(i["source"] == "added" for i in items),
        "added_not_a_restriction": sum(i["source"] == "added" and i["kind"] == NOT_A_RESTRICTION for i in items),
        "unknown_kind": sum(i["kind"] not in names and i["kind"] != NOT_A_RESTRICTION for i in items),
        "typing_calls": len(jobs), "not_typed": sum(t is None for t in typed.values()), "by": {}}
    finder = engine.latest(afc.CALLS_A)
    for who in (*STAND_IN_ARMS, "finder"):
        sets_right = cats_right = tp = fp = fn = 0
        for v in ctx["ids"]:
            gold = cy.gold_types(ctx, v)
            if who in STAND_IN_ARMS:
                got = frozenset(types[who][v] or ())
                kinds = {i["kind"] for i in answers[v] or [] if i["kind"] in names and i["source"] in STAND_IN_ARMS[who]}
            else:
                got = cy.read(ctx, finder[("named-first", 1, v)].get("answer"))[0] or frozenset()
                kinds = {of[t] for t in got}
            sets_right += got == gold
            cats_right += kinds == {of[t] for t in gold}
            tp, fp, fn = tp + len(got & gold), fp + len(got - gold), fn + len(gold - got)
        out["by"][who] = {"types_right": sets_right / len(ctx["ids"]), "categories_right": cats_right / len(ctx["ids"]),
                          "type_tp": tp, "type_fp": fp, "type_fn": fn,
                          "type_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    out["types"] = types
    return out


async def dev(ctx: Mapping[str, Any], transport: Any, guard: Guard) -> list[dict[str, Any]]:
    """Both new prompts on stage 3's invented variables (their gold constraint values as the found phrases).

    Args: ctx: from `load`; transport: PSNC's; guard: the stop rules.
    Returns: one row per call: its key, whether it was read and the answer.
    Raises: nothing from the provider.
    Side Effects: PSNC calls for invented definitions only; appends to `DEV`.
    """
    items = ct.invented(ctx)
    phrases = {name: [c["label"].split(":", 1)[-1].strip() for c in kn.INVENTED_GOLD[name].get("hasConstraint", [])]
               for name in items}
    await engine.run_jobs([("stand-in", 1, name) for name in sorted(items)],
                          lambda job: stand_in_prompt(items[job[2]]["definition"], phrases[job[2]], ctx["categories"]),
                          transport, DEV, concurrency=CONCURRENCY_ANSWERS, guard=guard, request=ANSWER_REQUEST)
    calls = engine.latest(DEV)
    answers = {name: read_answer(calls[("stand-in", 1, name)].get("answer")) for name in items}
    needed = typing_jobs(ctx["categories"], answers)
    await engine.run_jobs([("typing", 1, t) for t in sorted(needed)],
                          lambda job: type_prompt(items[needed[job[2]]["variable"]]["definition"], needed[job[2]]["phrase"],
                                                  needed[job[2]]["category"], ctx["definitions"]),
                          transport, DEV, concurrency=CONCURRENCY_ANSWERS, guard=guard, request=TYPE_REQUEST)
    calls = engine.latest(DEV)
    rows = [{"key": ("stand-in", name), "read": answers[name] is not None,
             "answer": calls[("stand-in", 1, name)].get("answer") or ""} for name in sorted(items)]
    rows += [{"key": ("typing", t), "read": read_type(calls.get(("typing", 1, t), {}).get("answer"),
                                                      needed[t]["category"]) is not None,
              "answer": calls.get(("typing", 1, t), {}).get("answer") or ""} for t in sorted(needed)]
    return rows


def fingerprints(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the code, the data, the stand-in's prompts and the two fixed decomposition arms.

    Args: ctx: from `load`.
    Returns: hex digests (16 characters) and mean prompt lengths; `best+stand-in` and the typing
        prompts depend on the answers and are hashed in the calls files.
    Raises: OSError when a file cannot be read.
    Side Effects: reads files and stage 5's similarity table.
    """
    out: dict[str, Any] = {name: ceiling.file_sha256(HERE / name)[:16] for name in (
        "stand_in_test.py", "constraint_types_test.py", "answer_form_check.py", "draft_rag.py", "rag_tuning.py",
        "walk.py", "engine.py")}
    out["data/question_categories_v1.json"] = ceiling.file_sha256(CATEGORIES)[:16]
    prompts = [stand_in_prompt(ctx["records"][v]["definition"], ctx["phrases"][v], ctx["categories"]) for v in ctx["ids"]]
    out["prompts"] = {"stand-in": {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in prompts))[:16],
                                   "chars_mean": round(sum(len(p) for p in prompts) / len(prompts), 1)}}
    world = rt.corpus_world(rt.load(), None)
    found = afc.finder_facts(ctx, engine.latest(afc.CALLS_A))
    for arm in ("best", "best+experts"):
        ps = [prompt2(ctx, world, (arm, 1, v), found, {a: {} for a in STAND_IN_ARMS}) for v in ctx["ids"]]
        out["prompts"][arm] = {"sha256": ft.sha256("\n".join(ft.sha256(p) for p in ps))[:16],
                               "chars_mean": round(sum(len(p) for p in ps) / len(ps), 1)}
    return out


COMPARISONS = (("best+screen-1", "best+stand-in"), ("best+stand-in", "best+experts"), ("best+screen-1", "best+experts"))


def summarise(ctx2: Mapping[str, Any], scores: Mapping[engine.Job, Mapping[str, Any]]) -> dict[str, Any]:
    """The decomposition: each arm's Close F1, each against `best`, and the plan's other comparisons.

    Args: ctx2: from `rag_tuning.load`; scores: from `engine.score`.
    Returns: the summary of `constraint_types_test.summarise2`, with "comparisons" ({"first - second":
        stage 5's interval} for `COMPARISONS`).
    Raises: KeyError when a score is missing.
    Side Effects: none. Deterministic.
    """
    out = cy.summarise2(ctx2, scores, ARMS)
    if out["complete"]:
        counts = analysis.variable_counts({k: s for k, s in scores.items() if k[0] in ARMS})
        out["comparisons"] = {f"{a} - {b}": rt.gain(counts, a, b, ctx2["ids"]) for a, b in COMPARISONS}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line mode.

    Args: argv: arguments; `sys.argv` when None.
    Returns: 0 on success; 1 when a stop rule ended a run or a summary is not complete.
    Raises: SystemExit on bad arguments or a step run out of order.
    Side Effects: --dev and --run-* call PSNC; the others read and write local files.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("--plan-only", "--dev", "--run-answers", "--score-answers", "--run-decomposition", "--score-decomposition"):
        mode.add_argument(name, action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args(argv)
    ctx = load()
    if args.plan_only:
        print(json.dumps({"provider": "PSNC", "model": engine.MODEL, "variables": len(ctx["ids"]),
                          "phrases_shown": sum(len(p) for p in ctx["phrases"].values()),
                          "variables_without_phrases": sum(not p for p in ctx["phrases"].values()), "arms": ARMS,
                          "decomposition_calls": len(ctx["ids"]) * REPETITIONS * len(ARMS),
                          "answer_request": ANSWER_REQUEST, "type_request": TYPE_REQUEST,
                          "fill_request": engine.REQUEST, **fingerprints(ctx)}, indent=2))
        return 0
    if args.score_answers:
        out = summarise_answers(ctx)
        SUMMARY_A.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "types"}, indent=1))
        return 0 if out["usable"] == out["variables"] else 1
    if args.score_decomposition:
        ctx2 = rt.load()
        summary = summarise(ctx2, engine.score(CALLS, SCORES, ctx2["records"], ctx2["inputs"]["schema"]))
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for arm, row in summary.get("arms", {}).items():
            print(f"{arm:14} F1 {row['f1_mean']:.4f} sd {row['f1_sd']:.4f} invalid {row['invalid']}")
        for arm, row in summary.get("against_best", {}).items():
            print(f"{arm} - best: {row['gain']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        for name, row in summary.get("comparisons", {}).items():
            print(f"{name}: {row['gain']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
        return 0 if summary["complete"] else 1

    async def go() -> int:
        """Open PSNC's transport, run the asked part, always close the client.

        Args: none.
        Returns: 1 when a stop rule ended the work, else 0.
        Raises: SystemExit when the key is missing or a step is run out of order.
        Side Effects: PSNC calls and the calls files.
        """
        guard = Guard()
        workers = args.concurrency or (CONCURRENCY if args.run_decomposition else CONCURRENCY_ANSWERS)
        transport, client = walk.chat_transport(concurrency=workers)
        try:
            if args.dev:
                for row in await dev(ctx, transport, guard):
                    print(row["key"], "read" if row["read"] else "NOT READ", "|", row["answer"].replace("\n", " ")[:300])
            elif args.run_answers:
                print(json.dumps(await engine.run_jobs(
                    [("stand-in", 1, v) for v in ctx["ids"]],
                    lambda job: stand_in_prompt(ctx["records"][job[2]]["definition"], ctx["phrases"][job[2]], ctx["categories"]),
                    transport, ANSWERS, concurrency=workers, guard=guard, request=ANSWER_REQUEST)))
                if not guard.reason:
                    needed = typing_jobs(ctx["categories"], stored_answers(ctx))
                    print(json.dumps(await engine.run_jobs(
                        [("typing", 1, t) for t in sorted(needed)],
                        lambda job: type_prompt(ctx["records"][needed[job[2]]["variable"]]["definition"],
                                                needed[job[2]]["phrase"], needed[job[2]]["category"], ctx["definitions"]),
                        transport, TYPING, concurrency=workers, guard=guard, request=TYPE_REQUEST)))
            else:
                if not SUMMARY_A.exists():
                    raise SystemExit("run --score-answers first")
                types = json.loads(SUMMARY_A.read_text(encoding="utf-8"))["types"]
                if set(types) != set(STAND_IN_ARMS):
                    raise SystemExit("the stored stand-in types do not match the arms; run --score-answers again")
                world = rt.corpus_world(rt.load(), None)
                found = afc.finder_facts(ctx, engine.latest(afc.CALLS_A))
                jobs = engine.schedule(ARMS, ctx["ids"], REPETITIONS, SEED)
                print(json.dumps(await engine.run_jobs(jobs, lambda job: prompt2(ctx, world, job, found, types), transport,
                                                       CALLS, concurrency=workers, guard=guard, request=dict(engine.REQUEST))))
            if guard.reason:
                print("stopped:", guard.reason)
            return 1 if guard.reason else 0
        finally:
            await client.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    raise SystemExit(main())
