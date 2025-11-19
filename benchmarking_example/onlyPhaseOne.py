#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
# I-ADOPT – Phase 1 Only (Matrix + Constraint Concepts + Evaluation)
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
from functools import lru_cache
from itertools import product

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer, util

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
LOG_FILE = LOG_DIR / f"onlyPhaseOne{datetime.now():%Y%m%d_%H%M%S}.log"

OUTBOOK_DIR = pathlib.Path("benchmarking_outputs")
OUTBOOK_DIR.mkdir(exist_ok=True)

MODEL_NAMES = ["qwen/qwen3-32b"]  # can be extended later
TEMPERATURES = [0.5]  # can be extended later

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLOSE_THR = 0.80

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

    ▸ Copy *comment* verbatim from the user section.
    ▸ Do **NOT** introduce keys that are absent from the schema.
    ▸ Every value must respect the declared JSON type.
    ▸ Reply with the JSON object only — no markdown fences, no narration.
"""
).strip()

BASELINE_INSTRUCTIONS = _SYSTEM_RULES

MATRIX_EXPLANATION = textwrap.dedent(
    """
    Additional guidance:

    Identify the Object of Interest:
    The Object of Interest is the Entity whose Property is observed.

    Identify the Matrix (if available):
    If the Object of Interest is embedded in, or part of, another Entity,
    that Entity is the Matrix. Not all observations have a Matrix.

    hasMatrix:
    A Variable might have an Entity in which the ObjectOfInterest is contained.

    hasConstraint:
    A Variable has a Constraint that confines an Entity involved in the observation.

    Further Decompose Entities:
    Revisit identified Entities (Object of Interest, Matrix, Context Objects) and
    check whether they can be decomposed into more general concepts and Constraints.

    Important:
    The framework does not capture units, instruments, methods, or geographical
    location. These must NOT be placed into the JSON.
"""
).strip()

PROMPT_TEMPLATES = {
    "baseline": BASELINE_INSTRUCTIONS,
    "matrix_explainer": BASELINE_INSTRUCTIONS + "\n\n" + MATRIX_EXPLANATION,
}

_EXAMPLE_HDR = "\n\n### Examples (valid against the same schema)\n"
_USER_HDR = "\n\n### Variable to decompose\n"
_EXPECTED = "\n\n### Expected output\n*(only the JSON object)*"


def build_prompt(comment: str, examples: List[Dict[str, Any]] | None, prompt_version: str) -> str:
    examples = examples or []

    # Pick the template by name (fallback = baseline)
    instructions = PROMPT_TEMPLATES.get(prompt_version, PROMPT_TEMPLATES["baseline"])

    ex_block = (
        _EXAMPLE_HDR + "\n\n".join(json.dumps(e, indent=2, ensure_ascii=False) for e in examples) if examples else ""
    )

    return (
        f"{instructions}\n\n"
        f"### JSON-Schema\n{_SCHEMA_TEXT}\n"
        f"{ex_block}"
        f"{_USER_HDR}comment: {comment}"
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
# ▪ LLM invocation helpers (Phase 1 only)
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
    data: Dict[str, Any] = {}
    for _ in range(3):
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
    if not data:
        return {}

    # Enforce label/comment and presence of ONTO_KEYS
    # data["label"] = label
    data["comment"] = comment
    for key in ONTO_KEYS:
        if key not in data:
            data[key] = [] if key == "hasConstraint" else ""
    return data


# --------------------------------------------------------------------------- #
# ▪ Similarity & confusion helpers (from original benchmark, simplified)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _cosine(a: str, b: str, model_name: str) -> float:
    embedder = load_embedder(model_name)
    emb1 = embedder.encode(a, convert_to_tensor=True)
    emb2 = embedder.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def sim_string(a: str, b: str, close: bool, model_name: str = EMBED_MODEL_NAME) -> float:
    if not a or not b:
        return 0.0
    norm_a, norm_b = a.lower().strip(), b.lower().strip()
    if norm_a == norm_b:
        return 1.0
    return _cosine(norm_a, norm_b, model_name) if close else 0.0


def sim_asym(a: Dict[str, str], b: Dict[str, str], close: bool) -> float:
    keys = ("AsymmetricSystem", "hasSource", "hasTarget")
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return 0.0
    return sum(sim_string(a.get(k, ""), b.get(k, ""), close) for k in keys) / 3


def _sym_parts(obj: Any):
    if isinstance(obj, dict) and "SymmetricSystem" in obj and "hasPart" in obj:
        return obj.get("SymmetricSystem", ""), set(obj.get("hasPart", []))
    return "", set()


def sim_sym(a: Any, b: Any, close: bool) -> float:
    lbl_a, parts_a = _sym_parts(a)
    lbl_b, parts_b = _sym_parts(b)
    if not (lbl_a or lbl_b):
        return 0.0
    label_sim = sim_string(lbl_a, lbl_b, close)
    part_sim = len(parts_a & parts_b) / len(parts_a | parts_b) if (parts_a or parts_b) else 1.0
    return (label_sim + part_sim) / 2


_ON_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+)$")


def canonical_on(text: str) -> str:
    if not text:
        return ""
    m = _ON_PREFIX_RE.match(text)
    if m and m.group(1) in ONTO_KEYS:
        return m.group(2).strip()
    return text.strip()


def sim_constraint(a: Dict[str, str], b: Dict[str, str], close: bool) -> float:
    lbl_sim = sim_string(a.get("label", ""), b.get("label", ""), close)
    on_a = canonical_on(a.get("on", ""))
    on_b = canonical_on(b.get("on", ""))
    on_sim = sim_string(on_a, on_b, close)
    return (lbl_sim + on_sim) / 2


def confusion(gt, pred, close: bool) -> tuple[float, float, float, float]:
    # handles strings and systems
    if isinstance(gt, dict) and "AsymmetricSystem" in gt:
        score = sim_asym(gt, pred, close)
    elif isinstance(gt, dict) and "SymmetricSystem" in gt:
        score = sim_sym(gt, pred, close)
    else:
        score = sim_string(str(gt), str(pred), close)
    thr = CLOSE_THR if close else 1.0
    if gt:
        if pred and score >= thr:
            return 1.0, 0.0, 0.0, 0.0  # TP
        elif pred:
            return 0.0, 1.0, 0.0, 0.0  # FP
        else:
            return 0.0, 0.0, 1.0, 0.0  # FN
    else:
        return (0.0, 0.0, 0.0, 1.0) if not pred else (0.0, 1.0, 0.0, 0.0)


def confusion_constraints(
    gt_list: List[Dict[str, str]],
    pred_list: List[Dict[str, str]],
    close: bool,
    model_name: str = EMBED_MODEL_NAME,
) -> tuple[float, float, float, float]:
    if not gt_list and not pred_list:
        return 0.0, 0.0, 0.0, 1.0
    if not gt_list:
        return 0.0, 1.0, 0.0, 0.0
    n_gt, n_pred = len(gt_list), len(pred_list)
    unit = 1.0 / (2 * n_gt)
    thr = CLOSE_THR if close else 1.0
    S = np.zeros((n_gt, n_pred))
    for i, j in product(range(n_gt), range(n_pred)):
        S[i, j] = sim_constraint(gt_list[i], pred_list[j], close)
    tp = fp = fn = 0.0
    gt_used: set[int] = set()
    pred_used: set[int] = set()
    while S.size:
        idx = int(np.argmax(S))
        i, j = divmod(idx, S.shape[1])
        if S[i, j] < 0:
            break
        gt_used.add(i)
        pred_used.add(j)
        if sim_string(gt_list[i].get("label", ""), pred_list[j].get("label", ""), close, model_name) >= thr:
            tp += unit
        else:
            fp += unit
        if (
            sim_string(
                canonical_on(gt_list[i].get("on", "")),
                canonical_on(pred_list[j].get("on", "")),
                close,
                model_name,
            )
            >= thr
        ):
            tp += unit
        else:
            fp += unit
        S[i, :] = -1.0
        S[:, j] = -1.0
    fn += (n_gt - len(gt_used)) * 2 * unit
    fp += (n_pred - len(pred_used)) * 2 * unit
    total = tp + fp + fn
    if 1.0 - total > 1e-6:
        fp += 1.0 - total
    elif total - 1.0 > 1e-6:
        tp /= total
        fp /= total
        fn /= total
    return tp, fp, fn, 0.0


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# --------------------------------------------------------------------------- #
# ▪ Phase 1 worker
# --------------------------------------------------------------------------- #
def _run_one(
    model: str,
    temperature: float,
    prompt_version: str,
    gt: Dict[str, Any],
    prompt: str,
    shot: int,
) -> Dict[str, Any]:
    pred = call_llm_loose(model, prompt, gt["label"], gt["comment"], temperature)
    return {
        "variable": gt["label"],
        "ground_truth_json": gt,
        "predicted_json": pred,
        "model": model,
        "temperature": temperature,
        "prompt_version": prompt_version,
        "shot": shot,
    }


# --------------------------------------------------------------------------- #
# ▪ Evaluation loop: run Phase 1 for a given shot
# --------------------------------------------------------------------------- #
def evaluate(
    data_dir: pathlib.Path,
    shot_mode: int,
    prompt_version: str,
    max_vars: int,
    models: List[str],
    temps: List[float],
    workers: int,
) -> List[Dict[str, Any]]:
    examples = load_examples(shot_mode)
    tasks: list[tuple] = []

    for gt_path in sorted(data_dir.glob("*.json"))[:max_vars]:
        gt = json.load(open(gt_path))
        if any(ex["label"] == gt["label"] for ex in examples):
            continue

        prompt = build_prompt(gt["comment"], examples, prompt_version)

        logging.info(
            "\n%s\nPROMPT | version=%s | shot=%d | %s\n%s\n%s",
            "═" * 120,
            prompt_version,
            shot_mode,
            gt["label"],
            prompt,
            "═" * 120,
        )

        for model in models:
            for temp in temps:
                tasks.append((model, temp, prompt_version, gt, prompt, shot_mode))

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, *task) for task in tasks]
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
    return results


# --------------------------------------------------------------------------- #
# ▪ Compute summary metrics (F_exact / F_close, per-key & overall)
# --------------------------------------------------------------------------- #
def compute_summary_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in results:
        key = (r["model"], r["temperature"], r["prompt_version"], r["shot"])
        groups.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, Any]] = []

    for (model, temp, prompt_version, shot), rows in groups.items():
        # overall stats across all keys
        overall = {
            "exact": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
            "close": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
        }
        # per-key stats
        per_key = {
            key: {
                "exact": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
                "close": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
            }
            for key in ONTO_KEYS
        }

        for row in rows:
            gt = row["ground_truth_json"]
            pred = row["predicted_json"]

            for tag, close in (("exact", False), ("close", True)):
                for key in ONTO_KEYS:
                    if key == "hasConstraint":
                        gt_val = gt.get(key, []) or []
                        pred_val = pred.get(key, []) or []
                        tp, fp, fn, tn = confusion_constraints(gt_val, pred_val, close)
                    else:
                        gt_val = gt.get(key, "")
                        pred_val = pred.get(key, "")
                        tp, fp, fn, tn = confusion(gt_val, pred_val, close)

                    overall[tag]["tp"] += tp
                    overall[tag]["fp"] += fp
                    overall[tag]["fn"] += fn
                    overall[tag]["tn"] += tn

                    per_key[tag_key := key][tag]["tp"] += tp
                    per_key[tag_key][tag]["fp"] += fp
                    per_key[tag_key][tag]["fn"] += fn
                    per_key[tag_key][tag]["tn"] += tn

        row_out: Dict[str, Any] = {
            "Model": model,
            "Temperature": temp,
            "PromptVersion": prompt_version,
            "Shot": shot,
        }

        # overall F/P/R for exact & close
        for tag in ("exact", "close"):
            tp = overall[tag]["tp"]
            fp = overall[tag]["fp"]
            fn = overall[tag]["fn"]
            p, r, f = prf(tp, fp, fn)
            suffix = "exact" if tag == "exact" else "close"
            row_out[f"P_{suffix}"] = round(p, 3)
            row_out[f"R_{suffix}"] = round(r, 3)
            row_out[f"F_{suffix}"] = round(f, 3)

        # per-key F/P/R
        for key in ONTO_KEYS:
            for tag in ("exact", "close"):
                tp = per_key[key][tag]["tp"]
                fp = per_key[key][tag]["fp"]
                fn = per_key[key][tag]["fn"]
                p, r, f = prf(tp, fp, fn)
                suffix = "exact" if tag == "exact" else "close"
                prefix = key  # e.g. hasMatrix
                row_out[f"{prefix}_P_{suffix}"] = round(p, 3)
                row_out[f"{prefix}_R_{suffix}"] = round(r, 3)
                row_out[f"{prefix}_F_{suffix}"] = round(f, 3)

        summary_rows.append(row_out)

    df = pd.DataFrame(summary_rows)
    if not df.empty:
        df = df.sort_values(by="F_exact", ascending=False)
    return df


# --------------------------------------------------------------------------- #
# ▪ Excel construction (4 sheets)
# --------------------------------------------------------------------------- #
def build_excel(results: List[Dict[str, Any]]) -> None:
    if not results:
        logging.warning("No results to write to Excel.")
        return

    # Sheet 1 – Matrix concepts
    matrix_rows = []
    for r in results:
        gt = r["ground_truth_json"]
        pred = r["predicted_json"]
        gt_matrix = gt.get("hasMatrix", "")
        pred_matrix = pred.get("hasMatrix", "")
        matrix_rows.append(
            {
                "variable": r["variable"],
                "model": r["model"],
                "temperature": r["temperature"],
                "prompt_version": r["prompt_version"],
                "shot": r["shot"],
                "ground_truth_matrix": json.dumps(gt_matrix, ensure_ascii=False, indent=2),
                "predicted_matrix": json.dumps(pred_matrix, ensure_ascii=False, indent=2),
            }
        )
    df_matrix = pd.DataFrame(matrix_rows)

    # Sheet 2 – Constraint concepts
    constraint_rows = []
    for r in results:
        gt = r["ground_truth_json"]
        pred = r["predicted_json"]
        gt_constr = gt.get("hasConstraint", [])
        pred_constr = pred.get("hasConstraint", [])
        constraint_rows.append(
            {
                "variable": r["variable"],
                "model": r["model"],
                "temperature": r["temperature"],
                "prompt_version": r["prompt_version"],
                "shot": r["shot"],
                "ground_truth_constraints": json.dumps(gt_constr, ensure_ascii=False, indent=2),
                "predicted_constraints": json.dumps(pred_constr, ensure_ascii=False, indent=2),
            }
        )
    df_constraints = pd.DataFrame(constraint_rows)

    # Sheet 3 – Full LLM outputs
    json_rows = []
    for r in results:
        json_rows.append(
            {
                "variable": r["variable"],
                "model": r["model"],
                "temperature": r["temperature"],
                "prompt_version": r["prompt_version"],
                "shot": r["shot"],
                "ground_truth_json": json.dumps(r["ground_truth_json"], ensure_ascii=False, indent=2),
                "predicted_json": json.dumps(r["predicted_json"], ensure_ascii=False, indent=2),
            }
        )
    df_json = pd.DataFrame(json_rows)

    # Sheet 4 – Summary (overall + per-key metrics)
    df_summary = compute_summary_metrics(results)

    out_xlsx = OUTBOOK_DIR / f"onlyPhaseOne{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
        df_matrix.to_excel(wr, sheet_name="Matrix concepts", index=False)
        df_constraints.to_excel(wr, sheet_name="Constraint concepts", index=False)
        df_json.to_excel(wr, sheet_name="LLM outputs", index=False)
        df_summary.to_excel(wr, sheet_name="Summary", index=False)

    logging.info("✓ Results saved → %s", out_xlsx.resolve())


# --------------------------------------------------------------------------- #
# ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="I-ADOPT LLM Phase 1 – Matrix & Constraint comparison + evaluation")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--max-vars", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)

    parser.add_argument(
        "--shot",
        type=int,
        choices=[0, 1, 3, 5],
        help="If provided: run only this shot. If omitted: run all shots.",
    )

    parser.add_argument(
        "--prompt-version",
        type=str,
        choices=list(PROMPT_TEMPLATES.keys()),
        help="If provided: run only this prompt version. If omitted: run all prompt versions.",
    )

    args = parser.parse_args()

    # Determine which prompt versions to run
    if args.prompt_version is None:
        prompt_versions = list(PROMPT_TEMPLATES.keys())  # all
    else:
        prompt_versions = [args.prompt_version]  # only selected

    # Determine which shots to run
    if args.shot is None:
        shots = [0, 1, 3, 5]  # all
    else:
        shots = [args.shot]  # only selected

    all_results = []

    # Full evaluation grid
    for pv in prompt_versions:
        for shot in shots:
            print(f"\n=== Running prompt_version={pv} | shot={shot} ===\n")

            results = evaluate(
                data_dir=args.data_dir,
                shot_mode=shot,
                prompt_version=pv,
                max_vars=args.max_vars,
                models=MODEL_NAMES,
                temps=TEMPERATURES,
                workers=args.workers,
            )

            all_results.extend(results)

    build_excel(all_results)


if __name__ == "__main__":
    main()
