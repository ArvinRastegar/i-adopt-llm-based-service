"""Recompute cost, latency and accuracy of the three entity linkers from the stored run files.

A post hoc side analysis: it was assembled after the results were known, re-runs nothing and
calls no service. It reads only what the three runs stored, with the standard library, and
shares no code with `iadopt_linking`, so it is an independent recompute: it stops if a headline
number does not match the run reports. Counts stay exact (numerator/denominator receipts);
shares, intervals, seconds and prices are floats because this is a figure, not committed evidence.

Inputs   results/baseline-20261014/{outcomes.csv, provenance.json}
         results/jev-20261015/{outcomes.csv, confidence.csv, responses.jsonl, provenance.json}
         results/psnc-qwen3.8-27b-20261015/{the same four files}
         outputs/baseline-20261013.log, outputs/baseline-20261014-v1.log (scratch, optional)
Output   numbers.json beside this file; `render_figure.py` turns it into the figure.

Run from anywhere:  python3 compute_numbers.py
"""

from __future__ import annotations

import collections
import csv
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "results"
OUTPUTS = PROJECT / "outputs"

BASELINE = "baseline-20261014"
JEV = "jev-20261015"
PSNC = "psnc-qwen3.8-27b-20261015"
CHOICE_PASSES = ("choice-a1", "choice-a1-repeat", "choice-a2-reversed", "choice-a3")
PRIMARY = CHOICE_PASSES[0]
EARLIER_RERANKER_LOGS = ("baseline-20261013.log", "baseline-20261014-v1.log")
DRAWS = 10_000
SEED = 20261019

# OpenRouter endpoint prices, read from the public endpoints listing on 2026-10-19 (three
# readings of the same page agreed on the rows that matter). They change; read them again
# before relying on them.
PRICES_READ_ON = "2026-10-19"
QWEN_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints"
JEV_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints"
JEV_USD_PER_INPUT_TOKEN = 0.000000042  # output tokens are free
# provider, USD per input token, USD per output token, lists "logprobs", quantization
QWEN_ENDPOINTS = (
    ("Tenstorrent", "0.00000002", "0.0000005", True, "fp4"),
    ("Darkbloom", "0.000000035", "0.0000022", True, "fp4"),
    ("Chutes", "0.000000039", "0.00000125", False, "fp8"),
    ("Near AI", "0.00000004", "0.00000135", True, "fp8"),
    ("Reka", "0.00000004", "0.0000022", True, "fp8"),
    ("Wafer", "0.00000004", "0.0000023", True, "fp8"),
    ("DekaLLM", "0.00000004", "0.0000025", True, "unknown"),
    ("Ionstream", "0.0000000893", "0.00000235", True, "fp8"),
    ("DeepInfra", "0.00000015", "0.000001875", False, "bf16"),
    ("ModelRun", "0.0000002", "0.0000025", False, "fp4"),
    ("Mancer 2", "0.0000002", "0.0000025", True, "fp8"),
    ("Phala", "0.0000002", "0.0000025", False, "unknown"),
    ("AkashML", "0.000000225", "0.00000198", True, "fp8"),
    ("Parasail", "0.00000024", "0.0000022", True, "fp8"),
    ("CoreWeave", "0.0000004", "0.000003", True, "fp8"),
    ("Novita", "0.00000042", "0.000003", True, "unknown"),
    ("Alibaba", "0.000000425", "0.00000255", True, "unknown"),
    ("Cloudflare", "0.00000045", "0.0000032", True, "unknown"),
    ("Venice", "0.00000045", "0.0000032", False, "fp8"),
    ("Cerebras", "0.00000099", "0.00000149", True, "fp16"),
)

# Headline figures printed in the three run reports; the recompute must reproduce them.
FROM_REPORTS = {
    "reachable slots": 284,
    "reranker correct": 194,
    "Qwen3.8-27B correct": 206,
    "Jev correct": 213,
    "Jev answered requests": 1430,
    "Jev input tokens": 1_685_133,
    "Jev billed USD": 0.070812294,
    "Qwen3.8-27B requests": 1144,
    "Qwen3.8-27B input tokens": 565_176,
    "Qwen3.8-27B output tokens": 1144,
    "Qwen3.8-27B request seconds": 340.7,
}
PROGRESS_LINE = re.compile(r"score (\d+)/(\d+) slots \[(\d+)s elapsed")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into one dict of strings per row.

    Args: path: the file.
    Returns: the rows, keyed by the header.
    Raises: OSError when the file cannot be read.
    Side effects: filesystem read.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file into one dict per non-empty line.

    Args: path: the file.
    Returns: the records, in file order.
    Raises: OSError when the file cannot be read; ValueError for a line that is not JSON.
    Side effects: filesystem read.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    """Take a percentile with linear interpolation between neighbours.

    Args: values: at least one number, any order; q: the fraction, 0 to 1.
    Returns: the interpolated value.
    Raises: IndexError for an empty list.
    Side effects: none.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interval(draws: list[float]) -> list[float]:
    """Give the middle 95% of resampled values.

    Args: draws: one value per resample.
    Returns: the 2.5th and 97.5th percentiles, rounded to four places.
    Raises: IndexError for an empty list.
    Side effects: none.
    """
    return [round(percentile(draws, 0.025), 4), round(percentile(draws, 0.975), 4)]


def accuracy() -> dict:
    """Count correct slots among the reachable ones, with a margin of error that resamples terms.

    Slots of one term are not independent (`water` alone is 12 slots), so each resample draws whole
    terms with replacement, the same for all three linkers so that their gaps are paired.

    Args: none.
    Returns: reachable slots, terms, and per linker a receipt, share and 95% interval, plus the
    paired gaps and the correct counts of the other candidate orders.
    Raises: SystemExit when the three files disagree on the slots or on the baseline's own result.
    Side effects: filesystem reads.
    """
    base = read_csv(RESULTS / BASELINE / "outcomes.csv")
    jev = read_csv(RESULTS / JEV / "outcomes.csv")
    psnc = read_csv(RESULTS / PSNC / "outcomes.csv")
    slot_ids = [row["slot_id"] for row in base]
    if slot_ids != [r["slot_id"] for r in jev] or slot_ids != [r["slot_id"] for r in psnc]:
        sys.exit("the three outcomes.csv files do not list the same slots in the same order")
    reachable = [r["slot_id"] for r in jev if r["reachable"] == "1"]
    if reachable != [r["slot_id"] for r in base if r["category"] != "unreachable_vocabulary"]:
        sys.exit("the baseline and the Jev file disagree on which slots are reachable")
    term_of = {r["slot_id"]: r["query_key"] for r in jev}
    correct = {
        "reranker": {r["slot_id"]: int(r["correct"]) for r in base},
        "qwen38": {r["slot_id"]: int(r["llm_correct"]) for r in psnc},
        "jev": {r["slot_id"]: int(r["jev_correct"]) for r in jev},
    }
    for rows in (jev, psnc):  # each file repeats the baseline's own result: they must agree
        if any(int(r["qwen3_correct"]) != correct["reranker"][r["slot_id"]] for r in rows):
            sys.exit("a comparison file's copy of the baseline result differs from the baseline")

    by_term: dict[str, list[str]] = collections.defaultdict(list)
    for slot in reachable:
        by_term[term_of[slot]].append(slot)
    terms = sorted(by_term)
    sizes = [len(by_term[t]) for t in terms]
    tally = {m: [sum(correct[m][s] for s in by_term[t]) for t in terms] for m in correct}
    pairs = (("jev", "qwen38"), ("jev", "reranker"), ("qwen38", "reranker"))
    rng = random.Random(SEED)
    shares: dict[str, list[float]] = {m: [] for m in correct}
    gaps: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}
    for _ in range(DRAWS):
        picked = [rng.randrange(len(terms)) for _ in terms]
        total = sum(sizes[i] for i in picked)
        draw = {m: sum(tally[m][i] for i in picked) / total for m in correct}
        for m in correct:
            shares[m].append(draw[m])
        for a, b in pairs:
            gaps[(a, b)].append(draw[a] - draw[b])

    n = len(reachable)
    hits = {m: sum(correct[m][s] for s in reachable) for m in correct}
    out: dict = {"reachable_slots": n, "terms": len(terms)}
    for m in correct:
        out[m] = {"receipt": {"numerator": hits[m], "denominator": n},
                  "share": round(hits[m] / n, 4), "ci95": interval(shares[m])}
    out["paired"] = {
        f"{a}_minus_{b}": {"difference": round((hits[a] - hits[b]) / n, 4),
                           "ci95": interval(gaps[(a, b)])}
        for a, b in pairs
    }
    for m, folder in (("qwen38", PSNC), ("jev", JEV)):
        per_pass: dict[str, int] = collections.Counter()
        for row in read_csv(RESULTS / folder / "confidence.csv"):
            if row["reachable"] == "1":
                per_pass[row["pass"]] += int(row["correct"])
        out[m]["other_orders"] = {p: per_pass[p] for p in CHOICE_PASSES}
        if m == "jev":
            out[m]["yes_no_pass"] = per_pass["noul-n1"]
    return out


def summarise_calls(seconds: list[float], inputs: int, outputs: int, billed: float,
                    rejected: int, retried: int) -> dict:
    """Summarise a set of calls: how long they took, their tokens and what was billed.

    Args: seconds: each call's own request time; inputs, outputs: token totals; billed: the cost
    the service reported; rejected: responses stored as failures and asked again; retried: calls
    that needed more than one attempt.
    Returns: counts, total, mean, median, 95th percentile and maximum call time, tokens, billed USD.
    Raises: IndexError when `seconds` is empty.
    Side effects: none.
    """
    ordered = sorted(seconds)
    return {
        "stored_responses": len(ordered),
        "rejected_and_asked_again": rejected,
        "calls_retried": retried,
        "call_seconds_total": round(sum(ordered), 3),
        "call_seconds_mean": round(statistics.fmean(ordered), 4),
        "call_seconds_median": round(statistics.median(ordered), 4),
        "call_seconds_p95": round(percentile(ordered, 0.95), 4),
        "call_seconds_max": round(ordered[-1], 3),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "billed_usd": round(billed, 9),
    }


def llm_calls(folder: str) -> dict:
    """Read a chat or decisions run: per pass call time, tokens and the cost the service reported.

    Args: folder: the run folder under results/.
    Returns: `passes` by name, a `total`, the wall-clock seconds the invocations asked for and the
    worker count.
    Raises: OSError or KeyError when the stored files lack what a response should carry.
    Side effects: filesystem reads.
    """
    raw: dict[str, dict] = {}
    for rec in read_jsonl(RESULTS / folder / "responses.jsonl"):
        body = json.loads(rec["response_text"]) if rec.get("response_text") else {}
        usage = body.get("usage", {})
        p = raw.setdefault(rec["pass"], {"seconds": [], "in": 0, "out": 0, "usd": 0.0,
                                         "rejected": 0, "retried": 0})
        p["seconds"].append(rec["request_s"])
        p["in"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
        p["out"] += usage.get("output_tokens", usage.get("completion_tokens", 0))
        p["usd"] += usage.get("cost", 0.0)
        p["rejected"] += 1 if rec.get("parse_error") else 0
        p["retried"] += 1 if rec.get("attempts", 1) > 1 else 0
    passes = {name: summarise_calls(p["seconds"], p["in"], p["out"], p["usd"],
                                    p["rejected"], p["retried"]) for name, p in raw.items()}
    total = summarise_calls([s for p in raw.values() for s in p["seconds"]],
                            sum(p["in"] for p in raw.values()),
                            sum(p["out"] for p in raw.values()),
                            sum(p["usd"] for p in raw.values()),
                            sum(p["rejected"] for p in raw.values()),
                            sum(p["retried"] for p in raw.values()))
    provenance = json.loads((RESULTS / folder / "provenance.json").read_text(encoding="utf-8"))
    invocations = provenance["invocations"]
    return {
        "passes": passes,
        "total": total,
        "wall_seconds_as_run": round(sum(i["seconds"].get("asking", 0.0) for i in invocations), 1),
        "workers": invocations[0]["workers"],
    }


def reranker() -> dict:
    """Read the reranker's scoring time for the whole list, with two earlier runs of the same job.

    The weights load lazily inside the timed scoring phase, so the time includes loading the model.

    Args: none.
    Returns: the committed run's seconds, all runs' seconds and range, lists and candidates scored,
    seconds per slot, device and batch size.
    Raises: SystemExit unless exactly one invocation scored the baseline.
    Side effects: filesystem reads (the scratch logs only when they exist).
    """
    provenance = json.loads((RESULTS / BASELINE / "provenance.json").read_text(encoding="utf-8"))
    scoring = [i for i in provenance["invocations"] if i["seconds"].get("scoring")]
    if len(scoring) != 1:
        sys.exit("expected exactly one invocation that scored the baseline")
    committed = scoring[0]["seconds"]["scoring"]
    outcomes = read_csv(RESULTS / BASELINE / "outcomes.csv")
    lists = sum(int(r["pool_size"]) > 0 for r in outcomes)
    runs = [round(committed, 1)]
    for name in EARLIER_RERANKER_LOGS:
        path = OUTPUTS / name
        if not path.exists():
            continue
        done = [int(m.group(3)) for m in PROGRESS_LINE.finditer(path.read_text(errors="replace"))
                if m.group(1) == m.group(2)]
        if done:
            runs.append(float(done[-1]))
    model = scoring[0].get("model") or {}
    return {
        "seconds": round(committed, 1),
        "run_seconds": sorted(runs),
        "range": [min(runs), max(runs)],
        "lists_scored": lists,
        "candidates_scored": sum(int(r["pool_size"]) for r in outcomes),
        "seconds_per_slot": round(committed / lists, 3),
        "device": model.get("device"),
        "batch_size": model.get("batch_size"),
        "includes_model_loading": True,
    }


def provider_costs(tokens_in: int, tokens_out: int) -> dict:
    """Price these tokens at every OpenRouter provider of Qwen3.8-27B.

    The experiment needs log-probabilities, so the median and the range use only the providers that
    list them; the range over all 20 is kept beside it.

    Args: tokens_in, tokens_out: the tokens to price.
    Returns: the median cost and its provider, the cheapest and priciest provider, the range over all
    endpoints, the provider counts and every provider's cost.
    Raises: ValueError when no provider offers log-probabilities.
    Side effects: none.
    """
    rows = []
    for name, p_in, p_out, logprobs, quantization in QWEN_ENDPOINTS:
        rows.append({
            "provider": name,
            "offers_logprobs": logprobs,
            "quantization": quantization,
            "usd_per_million_input": round(float(p_in) * 1e6, 4),
            "usd_per_million_output": round(float(p_out) * 1e6, 4),
            "usd": round(tokens_in * float(p_in) + tokens_out * float(p_out), 6),
        })
    usable = sorted((r for r in rows if r["offers_logprobs"]), key=lambda r: r["usd"])
    median = statistics.median(r["usd"] for r in usable)
    nearest = min(usable, key=lambda r: abs(r["usd"] - median))
    return {
        "median_usd": round(median, 6),
        "median_provider": nearest["provider"],
        "cheapest": usable[0],
        "priciest": usable[-1],
        "all_endpoints_usd_range": [min(r["usd"] for r in rows), max(r["usd"] for r in rows)],
        "providers_offering_logprobs": len(usable),
        "providers_listed": len(rows),
        "providers": sorted(rows, key=lambda r: r["usd"]),
    }


def spread(summary: dict, names: tuple[str, ...]) -> list[float]:
    """Give the smallest and largest total call time over some passes.

    Args: summary: the result of `llm_calls`; names: the passes to compare.
    Returns: [shortest, longest] total seconds.
    Raises: KeyError for an unknown pass.
    Side effects: none.
    """
    totals = [summary["passes"][n]["call_seconds_total"] for n in names]
    return [min(totals), max(totals)]


def main() -> None:
    """Recompute everything, check it against the run reports and write numbers.json.

    Args: none.
    Returns: None.
    Raises: SystemExit when a headline number differs from its report; nothing is written then.
    Side effects: prints the checks; writes numbers.json beside this file.
    """
    acc = accuracy()
    jev = llm_calls(JEV)
    qwen = llm_calls(PSNC)
    rer = reranker()

    jev_list = jev["passes"][PRIMARY]
    qwen_list = qwen["passes"][PRIMARY]
    jev_usd = jev_list["input_tokens"] * JEV_USD_PER_INPUT_TOKEN
    qwen_cost = provider_costs(qwen_list["input_tokens"], qwen_list["output_tokens"])
    qwen_total_cost = provider_costs(qwen["total"]["input_tokens"], qwen["total"]["output_tokens"])

    jev_answered = jev["total"]["stored_responses"] - jev["total"]["rejected_and_asked_again"]
    rejected_input = jev["passes"]["choice-a2-reversed"]["input_tokens"] - jev_list["input_tokens"]
    checks = [
        ("reachable slots", acc["reachable_slots"], 0),
        ("reranker correct", acc["reranker"]["receipt"]["numerator"], 0),
        ("Qwen3.8-27B correct", acc["qwen38"]["receipt"]["numerator"], 0),
        ("Jev correct", acc["jev"]["receipt"]["numerator"], 0),
        ("Jev answered requests", jev_answered, 0),
        ("Jev input tokens", jev["total"]["input_tokens"] - rejected_input, 0),
        ("Jev billed USD", jev["total"]["billed_usd"], 1e-9),
        ("Qwen3.8-27B requests", qwen["total"]["stored_responses"], 0),
        ("Qwen3.8-27B input tokens", qwen["total"]["input_tokens"], 0),
        ("Qwen3.8-27B output tokens", qwen["total"]["output_tokens"], 0),
        ("Qwen3.8-27B request seconds", qwen["total"]["call_seconds_total"], 0.05),
    ]
    failed = []
    print("Recompute against the run reports:")
    for label, got, tolerance in checks:
        want = FROM_REPORTS[label]
        ok = abs(got - want) <= tolerance
        failed += [] if ok else [label]
        print(f"  {'ok ' if ok else 'BAD'} {label}: {got} (report: {want})")
    price_matches = abs(jev_list["billed_usd"] - jev_usd) < 1e-9
    print(f"  {'ok ' if price_matches else 'BAD'} Jev billed = input tokens x $0.042/M: "
          f"{jev_list['billed_usd']} vs {round(jev_usd, 9)}")
    if failed or not price_matches:
        sys.exit("recompute does not match the reports; no numbers.json written")

    numbers = {
        "meta": {
            "kind": "post hoc side analysis; re-runs nothing, calls no service",
            "list": "one list = the 286 slots that have candidates, asked once",
            "primary_pass": PRIMARY,
            "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "term (query_key)",
                          "terms": acc["terms"]},
            "prices_read_on": PRICES_READ_ON,
            "prices_sources": [QWEN_ENDPOINTS_URL, JEV_ENDPOINTS_URL],
            "reports_reproduced": [label for label, _, _ in checks],
        },
        "accuracy": acc,
        "latency": {
            "list_requests": jev_list["stored_responses"],
            "reranker": rer,
            "qwen38": {
                "seconds": qwen_list["call_seconds_total"],
                "range": spread(qwen, CHOICE_PASSES),
                "mean_call_seconds": qwen_list["call_seconds_mean"],
                "wall_seconds_per_list_as_run": round(qwen["wall_seconds_as_run"] / 4, 1),
                "workers": qwen["workers"],
                "passes": qwen["passes"],
            },
            "jev": {
                "seconds": jev_list["call_seconds_total"],
                "range": spread(jev, CHOICE_PASSES),
                "mean_call_seconds": jev_list["call_seconds_mean"],
                "wall_seconds_per_list_as_run": round(jev["wall_seconds_as_run"] / 5, 1),
                "workers": jev["workers"],
                "passes": jev["passes"],
            },
        },
        "cost": {
            "reranker": {"usd_per_list": None,
                         "note": "runs on the Mac: no per-call price (power and hardware not "
                                 "counted)"},
            "qwen38": {
                "usd_per_list": qwen_cost["median_usd"],
                "input_tokens_per_list": qwen_list["input_tokens"],
                "output_tokens_per_list": qwen_list["output_tokens"],
                "openrouter": qwen_cost,
                "free_at": "PSNC (the run was not billed)",
            },
            "jev": {
                "usd_per_list": round(jev_list["billed_usd"], 6),
                "input_tokens_per_list": jev_list["input_tokens"],
                "usd_per_million_input": round(JEV_USD_PER_INPUT_TOKEN * 1e6, 6),
                "billed_equals_tokens_times_price": price_matches,
                "billed_per_choice_pass": [jev["passes"][n]["billed_usd"] for n in CHOICE_PASSES],
            },
            "ratios": {
                "qwen38_median_over_jev": round(qwen_cost["median_usd"] / jev_list["billed_usd"], 2),
                "qwen38_cheapest_over_jev": round(
                    qwen_cost["cheapest"]["usd"] / jev_list["billed_usd"], 2),
                "qwen38_priciest_over_jev": round(
                    qwen_cost["priciest"]["usd"] / jev_list["billed_usd"], 2),
            },
        },
        "as_run": {
            "reranker": {"calls": "1 scoring run over 286 lists", "seconds_summed": rer["seconds"],
                         "wall_seconds": rer["seconds"], "usd": None},
            "qwen38": {
                "calls": qwen["total"]["stored_responses"],
                "input_tokens": qwen["total"]["input_tokens"],
                "output_tokens": qwen["total"]["output_tokens"],
                "seconds_summed": qwen["total"]["call_seconds_total"],
                "wall_seconds": qwen["wall_seconds_as_run"],
                "workers": qwen["workers"],
                "usd_psnc": 0.0,
                "usd_openrouter_median": qwen_total_cost["median_usd"],
                "usd_openrouter_range": [qwen_total_cost["cheapest"]["usd"],
                                         qwen_total_cost["priciest"]["usd"]],
            },
            "jev": {
                "calls": jev["total"]["stored_responses"],
                "input_tokens": jev["total"]["input_tokens"],
                "output_tokens": jev["total"]["output_tokens"],
                "seconds_summed": jev["total"]["call_seconds_total"],
                "wall_seconds": jev["wall_seconds_as_run"],
                "workers": jev["workers"],
                "usd_billed": jev["total"]["billed_usd"],
            },
        },
    }
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'numbers.json'}")


if __name__ == "__main__":
    main()
