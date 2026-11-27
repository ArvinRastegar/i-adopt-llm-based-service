"""Stage orchestration and reporting — see contracts/pipeline.md.

The stage ordering is a scientific guard, not a convenience: the confirmation set must never
inform which candidates are selected, so the pipeline refuses to evaluate it before the final
stage and refuses to let its observations reach finalist selection."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import design
from attribution import fit
from harness import candidate_hash, evaluate

OBSERVATIONS = "example-selection-observations.jsonl"
REFINE_NEIGHBOURS = 12

DEFAULT_SUBSETS = 600
DEFAULT_SEARCH_REPS = 5
DEFAULT_CONFIRM_REPS = 15
CALL_CEILING = 120_000


def planned_calls(n_subsets: int, search_reps: int, confirm_reps: int,
                  n_candidates: int, n_eval: int, n_confirm: int) -> int:
    """Compute the provider calls a full run would issue, for the budget gate."""
    attribution = n_subsets * n_eval
    construct = n_candidates * search_reps * n_eval
    refine = REFINE_NEIGHBOURS * search_reps * n_eval
    confirm = n_candidates * confirm_reps * n_confirm
    return attribution + construct + refine + confirm


async def stage_attribution(context: Any, split: Mapping[str, list[str]],
                            n_subsets: int = DEFAULT_SUBSETS) -> dict:
    """Evaluate random subsets on E and fit the contribution model."""
    pool, targets = split["P"], split["E"]
    _check_budget(n_subsets, len(targets), len(split["C"]),
                  search_reps=getattr(context, "search_reps", DEFAULT_SEARCH_REPS),
                  confirm_reps=getattr(context, "confirm_reps", DEFAULT_CONFIRM_REPS))
    subsets = design.sample_subsets(pool, n_subsets, seed=17)
    observations = []
    for position, subset in enumerate(subsets, 1):
        result = await _observe(context, subset, targets, 1, "attribution", f"subset-{position}", "E")
        observations.append((subset, result["close_f1_official"]))
    return {"fit": fit(pool, observations), "observations": observations}


async def stage_construct(context: Any, split: Mapping[str, list[str]],
                          coefficients: Mapping[str, float], reps: int = DEFAULT_SEARCH_REPS) -> dict:
    """Build finalists and baselines and evaluate each on E with repetitions."""
    pool, targets = split["P"], split["E"]
    candidates = {
        "top25": design.top_k_candidate(pool, coefficients),
        "bottom25": design.top_k_candidate(pool, coefficients, invert=True),
        "stratified": design.stratified_baseline(list(context.records.values()), pool),
    }
    for position, subset in enumerate(design.random_baselines(pool, 3, seed=29), 1):
        candidates[f"random-{position}"] = subset
    scores: dict[str, list[float]] = {}
    for name, subset in candidates.items():
        scores[name] = [
            (await _observe(context, subset, targets, rep, "construct", name, "E"))["close_f1_official"]
            for rep in range(1, reps + 1)]
    return {"candidates": candidates, "scores": scores}


async def stage_refine(context: Any, split: Mapping[str, list[str]],
                       candidates: Mapping[str, Sequence[str]], reps: int = DEFAULT_SEARCH_REPS) -> dict:
    """Run replicated swap search around the best constructed candidate."""
    pool, targets = split["P"], split["E"]
    start = max(candidates, key=lambda name: statistics.mean(candidates[name]["scores"])) \
        if candidates and isinstance(next(iter(candidates.values())), dict) else None
    best_name = start or next(iter(candidates))
    best = tuple(candidates[best_name]["subset"] if isinstance(candidates[best_name], dict)
                 else candidates[best_name])
    trace, scores = [], {}
    for neighbour in design.perturb(best, pool, REFINE_NEIGHBOURS, swaps=1, seed=31):
        name = f"swap-{candidate_hash(neighbour)}"
        scores[name] = [
            (await _observe(context, neighbour, targets, rep, "refine", name, "E"))["close_f1_official"]
            for rep in range(1, reps + 1)]
        trace.append({"name": name, "subset": list(neighbour), "mean": statistics.mean(scores[name])})
    winner = max(trace, key=lambda row: row["mean"]) if trace else None
    return {"best": tuple(winner["subset"]) if winner else best, "trace": trace, "scores": scores}


async def stage_confirm(context: Any, split: Mapping[str, list[str]],
                        finalists: Mapping[str, Sequence[str]],
                        reps: int = DEFAULT_CONFIRM_REPS) -> dict:
    """Evaluate finalists and baselines on the held-out C — the only headline evidence."""
    path = _observations_path(context)
    prior = _load_observations(path)
    if not any(row["stage"] == "attribution" for row in prior):
        raise RuntimeError("stage_confirm requires a completed attribution stage; none recorded")
    already = {row["name"] for row in prior if row["eval_set"] == "C"}
    unexpected = already - set(finalists)
    if unexpected:
        raise RuntimeError(f"confirmation set already evaluated for {sorted(unexpected)}")
    targets = split["C"]
    out: dict[str, dict] = {}
    for name, subset in finalists.items():
        values = [
            (await _observe(context, subset, targets, rep, "confirm", name, "C"))["close_f1_official"]
            for rep in range(1, reps + 1)]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        sem = sd / len(values) ** 0.5 if values else 0.0
        out[name] = {"mean": mean, "sd": sd, "ci": (mean - 1.96 * sem, mean + 1.96 * sem),
                     "scores": values}
    return out


def write_report(path: Path, results: Mapping[str, Any]) -> None:
    """Render the result: C comparison with CIs, falsification check, per-example table."""
    confirm = results.get("confirm", {})
    falsification = results.get("falsification", {})
    lines = [
        "# Example-selection optimization — result",
        "",
        "**Exploratory side experiment.** Not part of the official experiment; these numbers",
        "cannot enter an official ranking.",
        "",
        f"Split: pool {results['split']['P']}, search-eval {results['split']['E']}, "
        f"confirmation {results['split']['C']}.",
        "",
        "## Held-out confirmation (the result)",
        "",
        "Measured on `C`, which never informed selection. This is the only evidence here.",
        "",
        "| Candidate | Close F1 | SD | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for name, row in sorted(confirm.items(), key=lambda kv: -kv[1]["mean"]):
        lines.append(f"| `{name}` | {row['mean']:.4f} | {row['sd']:.4f} | "
                     f"[{row['ci'][0]:.4f}, {row['ci'][1]:.4f}] |")
    lines += [
        "",
        "## Search scores (selection-biased — not the result)",
        "",
        "Every figure below was measured on `E`, the set used to *choose* these candidates.",
        "Picking the maximum of a noisy objective inflates it, so these are **selection-biased**",
        "and are reported only to show the search worked, never as the finding.",
        "",
    ]
    for name, row in sorted(results.get("search", {}).items()):
        mean = row["mean"] if isinstance(row, dict) else row
        lines.append(f"- `{name}`: {mean:.4f} (selection-biased)")
    lines += ["", "## Falsification check", ""]
    if falsification.get("passed"):
        lines += ["The top-25 beat both the attenuated bottom-25 control and the random-25",
                  "reference, so the attribution model learned something real.", ""]
    else:
        lines += ["**The attribution model failed to learn.** The top-25 did not beat both the",
                  "bottom-25 control and the random-25 reference, so the winning set is not",
                  "evidence of anything and must not be reported as the best 25.", ""]
    lines += ["The bottom-25 control is **attenuated**: it necessarily shares 10 of its 25",
              "members with the top-25 (design INV-9), so a real effect appears reduced. Do not",
              "over-read a small margin against it.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _observations_path(context: Any) -> Path:
    """The observations JSONL, resolved beside the harness call log."""
    return Path(context.calls_path).parent / OBSERVATIONS


def _load_observations(path: Path) -> list[dict]:
    """Read recorded stage observations; an absent file means no stage has run."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


async def _observe(context: Any, subset: Sequence[str], targets: Sequence[str], repetition: int,
                   stage: str, name: str, eval_set: str) -> dict:
    """Evaluate one candidate and record the observation, reusing any prior identical one."""
    path = _observations_path(context)
    digest = candidate_hash(subset)
    # Memoised on the context: re-reading the whole JSONL per candidate made the lookup
    # quadratic in the number of observations, which is thousands by the refine stage.
    index = getattr(context, "_observation_index", None)
    if index is None:
        index = {(row["candidate_hash"], row["repetition"], row["eval_set"]): row["result"]
                 for row in _load_observations(path)}
        context._observation_index = index
    key = (digest, repetition, eval_set)
    if key in index:
        return index[key]
    result = await evaluate(subset, targets, repetition, context)
    index[key] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "name": name, "candidate_hash": digest,
                                 "subset": list(subset), "repetition": repetition,
                                 "eval_set": eval_set, "result": result}, ensure_ascii=False) + "\n")
    return result


def _check_budget(n_subsets: int, n_eval: int, n_confirm: int,
                  search_reps: int = DEFAULT_SEARCH_REPS,
                  confirm_reps: int = DEFAULT_CONFIRM_REPS) -> None:
    """Refuse a plan that would exceed the declared call ceiling, before any dispatch."""
    planned = planned_calls(n_subsets, search_reps, confirm_reps,
                            n_candidates=6, n_eval=n_eval, n_confirm=n_confirm)
    if planned > CALL_CEILING:
        raise RuntimeError(f"planned {planned} calls exceeds the ceiling {CALL_CEILING}")
