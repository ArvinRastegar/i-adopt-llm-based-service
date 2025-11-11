#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
# I-ADOPT LLM Benchmark (Phase 1 only – Matrix Concept Extraction, Parallel)
# --------------------------------------------------------------------------- #
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import textwrap
from datetime import datetime
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import httpx
from openai import OpenAI, OpenAIError, APIStatusError
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# ▪ Static configuration
# --------------------------------------------------------------------------- #
load_dotenv()

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPT_DIR / "data" / "Json_schema.json"
DATA_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "test_set"
ONE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/one_shot"
THREE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/three_shot"
FIVE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/five_shot"

LOG_DIR = SCRIPT_DIR / "benchmarking_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"phase1_matrixConcepts_{datetime.now():%Y%m%d_%H%M%S}.log"

OUTBOOK_DIR = pathlib.Path("benchmarking_outputs")
OUTBOOK_DIR.mkdir(exist_ok=True)

MODEL_NAMES = ["qwen/qwen3-32b"]
TEMPERATURES = [0.5]

ONTO_KEYS = [
    "hasStatisticalModifier",
    "hasProperty",
    "hasObjectOfInterest",
    "hasMatrix",
    "hasContextObject",
    "hasConstraint",
]

# --------------------------------------------------------------------------- #
# ▪ Logging
# --------------------------------------------------------------------------- #
log_fmt = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_fmt,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logging.info(f"Logging to {LOG_FILE.resolve()}")

# --------------------------------------------------------------------------- #
# ▪ OpenAI client
# --------------------------------------------------------------------------- #
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

# --------------------------------------------------------------------------- #
# ▪ Prompt helpers
# --------------------------------------------------------------------------- #
_SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8").strip()

_SYSTEM_RULES = textwrap.dedent(
    """
    You are an ontology engineer.
    Your task is to output **one** JSON object that satisfies the
    JSON-Schema provided below.

    ▸ Copy *label* and *comment* verbatim from the user section.
    ▸ Do **NOT** introduce keys that are absent from the schema.
    ▸ Every value must respect the declared JSON type
      (e.g. hasProperty is a string, hasConstraint is an array, …).
    ▸ Reply with the JSON object only — no markdown fences, no narration.
"""
).strip()

_EXAMPLE_HDR = "\n\n### Examples (valid against the same schema)\n"
_USER_HDR = "\n\n### Variable to decompose\n"
_EXPECTED = "\n\n### Expected output\n*(only the JSON object)*"


def build_prompt(label: str, comment: str, examples: List[Dict[str, Any]] | None = None) -> str:
    examples = examples or []
    ex_block = (
        _EXAMPLE_HDR + "\n\n".join(json.dumps(e, indent=2, ensure_ascii=False) for e in examples) if examples else ""
    )
    return (
        f"{_SYSTEM_RULES}\n\n"
        f"### JSON-Schema\n{_SCHEMA_TEXT}\n"
        f"{ex_block}"
        f"{_USER_HDR}label: {label}\ncomment: {comment}"
        f"{_EXPECTED}"
    )


def load_examples(n: int) -> List[Dict[str, Any]]:
    if n == 0:
        return []
    if n == 1:
        folder = ONE_SHOT_DIR
    elif n == 3:
        folder = THREE_SHOT_DIR
    elif n == 5:
        folder = FIVE_SHOT_DIR
    else:
        raise ValueError("shot must be 0, 1, 3 or 5")
    paths = sorted(folder.glob("*.json"))
    return [json.load(open(p)) for p in paths[:n]]


# --------------------------------------------------------------------------- #
# ▪ LLM invocation helpers
# --------------------------------------------------------------------------- #
_JSON_FENCE_RE = re.compile(r"```(?:json)?", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_model(model: str, prompt: str, temperature: float) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return resp.choices[0].message.content
    except (APIStatusError, OpenAIError, httpx.HTTPError) as e:
        logging.warning(f"{model}: LLM request failed – {e!r}")
        return ""


def call_llm_loose(model: str, prompt: str, label: str, comment: str, temperature: float) -> Dict[str, Any]:
    for attempt in range(3):
        raw = call_model(model, prompt, temperature)
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
            break
        except json.JSONDecodeError:
            continue
    else:
        return {}

    # Force label/comment from ground truth
    data["label"] = label
    data["comment"] = comment
    for key in ONTO_KEYS:
        if key not in data:
            data[key] = [] if key == "hasConstraint" else ""
    return data


# --------------------------------------------------------------------------- #
# ▪ Phase 1 Runner
# --------------------------------------------------------------------------- #


def _run_one(model: str, gt: Dict[str, Any], prompt: str, shot: int, temperature: float) -> Dict[str, Any]:
    """Run a single Phase 1 LLM decomposition (no caching, includes full JSONs)."""
    try:
        logging.info("Running LLM | model=%s | shot=%d | T=%.2f | var=%r", model, shot, temperature, gt["label"])

        # Call LLM directly
        pred = call_llm_loose(model, prompt, gt["label"], gt["comment"], temperature)

        # Extract key information
        gt_matrix = gt.get("hasMatrix", "")
        pred_matrix = pred.get("hasMatrix", "")

        return {
            "variable": gt["label"],
            "ground_truth_matrix": gt_matrix,  # only hasMatrix field
            "predicted_matrix": pred_matrix,  # only hasMatrix field
            "ground_truth_json": gt,  # full ground truth JSON
            "predicted_json": pred,  # full LLM JSON
            "shot": shot,
        }

    except Exception as e:
        logging.error("%s | %s: worker crashed – %r", model, gt.get("label", "?"), e)
        return {}


# --------------------------------------------------------------------------- #
# ▪ Evaluation loop (Phase 1 only, parallelized)
# --------------------------------------------------------------------------- #
def evaluate(
    data_dir: pathlib.Path,
    shot_mode: int,
    max_vars: int = 30,
    models: List[str] | None = None,
    temps: List[float] | None = None,
    workers: int = 8,
) -> List[Dict[str, Any]]:
    temps = temps or TEMPERATURES
    models = models or MODEL_NAMES
    examples = load_examples(shot_mode)
    tasks: list[tuple] = []

    for gt_path in sorted(data_dir.glob("*.json"))[:max_vars]:
        gt = json.load(open(gt_path))
        if any(ex["label"] == gt["label"] for ex in examples):
            continue
        prompt = build_prompt(gt["label"], gt["comment"], examples)
        for model in models:
            for temp in temps:
                tasks.append((model, gt, prompt, shot_mode, temp))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, *task) for task in tasks]
        for f in as_completed(futs):
            res = f.result()
            if res:
                results.append(res)
    return results


# --------------------------------------------------------------------------- #
# ▪ Excel construction
# --------------------------------------------------------------------------- #
def build_excel(results_by_shot: Dict[int, List[Dict[str, Any]]]) -> None:
    merged: Dict[str, Dict[str, Any]] = {}
    for shot, rows in results_by_shot.items():
        for r in rows:
            var = r["variable"]
            merged.setdefault(
                var, {"ground_truth_matrix": r["ground_truth_matrix"], "ground_truth_json": r["ground_truth_json"]}
            )
            merged[var][shot] = {"predicted_matrix": r["predicted_matrix"], "predicted_json": r["predicted_json"]}

    # Sheet 1 – Matrix concept comparison + ground truth
    df_matrix = pd.DataFrame(
        [
            {
                "variable description": var,
                "ground truth": data.get("ground_truth_matrix", ""),
                "0-shot": data.get(0, {}).get("predicted_matrix", ""),
                "1-shot": data.get(1, {}).get("predicted_matrix", ""),
                "3-shot": data.get(3, {}).get("predicted_matrix", ""),
                "5-shot": data.get(5, {}).get("predicted_matrix", ""),
            }
            for var, data in merged.items()
        ]
    )

    # Sheet 2 – Full JSON comparison
    df_text = pd.DataFrame(
        [
            {
                "variable description": var,
                "ground truth": json.dumps(data.get("ground_truth_json", {}), ensure_ascii=False, indent=2),
                "LLM output 0-shot": json.dumps(
                    data.get(0, {}).get("predicted_json", {}), ensure_ascii=False, indent=2
                ),
                "LLM output 1-shot": json.dumps(
                    data.get(1, {}).get("predicted_json", {}), ensure_ascii=False, indent=2
                ),
                "LLM output 3-shot": json.dumps(
                    data.get(3, {}).get("predicted_json", {}), ensure_ascii=False, indent=2
                ),
                "LLM output 5-shot": json.dumps(
                    data.get(5, {}).get("predicted_json", {}), ensure_ascii=False, indent=2
                ),
            }
            for var, data in merged.items()
        ]
    )

    out_xlsx = OUTBOOK_DIR / f"phase1_matrixConcepts_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
        df_matrix.to_excel(wr, sheet_name="Matrix concept", index=False)
        df_text.to_excel(wr, sheet_name="Outputs comparison", index=False)

    logging.info("✓ Results saved → %s", out_xlsx.resolve())


# --------------------------------------------------------------------------- #
# ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="I-ADOPT LLM Phase 1 – Matrix concept comparison (parallel)")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR, help="Folder with ground-truth JSON files")
    parser.add_argument("--max-vars", type=int, default=30, help="Limit number of variables for run")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel threads for LLM calls")
    parser.add_argument(
        "--shot",
        type=int,
        choices=[0, 1, 3, 5],
        default=None,
        help="Prompting mode (0 / 1 / 3 / 5). If omitted, all four modes are executed.",
    )
    args = parser.parse_args()

    shots = [args.shot] if args.shot is not None else [0, 1, 3, 5]
    results_by_shot: Dict[int, List[Dict[str, Any]]] = {}

    for s in shots:
        logging.info("Running Phase 1 for %d-shot …", s)
        results_by_shot[s] = evaluate(args.data_dir, shot_mode=s, max_vars=args.max_vars, workers=args.workers)

    build_excel(results_by_shot)


if __name__ == "__main__":
    main()
