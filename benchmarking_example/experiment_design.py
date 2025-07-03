#!/usr/bin/env python3
"""
experiment_design.py
===========================

End-to-end benchmark of *LLM-powered* I-ADOPT decomposition.

Run

    python experiment_design.py --help

for CLI options.
"""

# --------------------------------------------------------------------------- #
# 0 ▪ Imports & constants
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
from threading import Lock
from typing import Any, Dict, List, Set, Tuple

import httpx
import pandas as pd
from openai import APIStatusError, OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter


# from jsonschema import validate, ValidationError

# ----- static config -------------------------------------------------------- #
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPT_DIR / "data" / "Json_schema.json"
DATA_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "test_set"
ONE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/one_shot"
THREE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/three_shot"
FIVE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/five_shot"

LOG_DIR = SCRIPT_DIR / "benchmarking_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"iadopt_run_{datetime.now():%Y%m%d_%H%M%S}.log"

MODEL_NAMES = [
    # "openai/gpt-4o",
    # "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    # "meta-llama/llama-3.3-8b-instruct",
    # "qwen/qwen3-8b",
    "microsoft/phi-4",
    # "microsoft/phi-4-reasoning",
    # "open-orca/mistral-7b-openorca",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    # "qwen/qwen-2.5-7b-instruct",
    # "qwen/qwen-2.5-coder-32b-instruct",
    # "qwen/qwen3-0.6b-04-28",
    # "qwen/qwen3-1.7b",
    # "qwen/qwen3-4b",
    "qwen/qwen3-14b",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-32b",
    # "meta-llama/llama-guard-4-12b",
    # "perplexity/sonar-reasoning-pro",
    # "google/gemini-2.5-pro",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "anthropic/claude-4-sonnet-20250522",
    # "intel/neural-chat-7b",
]


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

EXACT_THR = 0.90  # Levenshtein threshold for “exact”
CLOSE_THR = 0.90  # cosine threshold for “close”
MAX_RETRY = 1  # schema-retry attempts

OUTBOOK_DIR = pathlib.Path("benchmarking_outputs")
OUTBOOK_DIR.mkdir(exist_ok=True)

ONTO_KEYS = [
    "hasStatisticalModifier",
    "hasProperty",
    "hasObjectOfInterest",
    "hasMatrix",
    "hasContextObject",
    "hasConstraint",
]

# --------------------------------------------------------------------------- #
# 1 ▪ Logging & external clients
# --------------------------------------------------------------------------- #
# ─── LOGGING: console  + 2 file handlers ────────────────────────────── #
log_fmt = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_fmt,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)

PREPROC_LOG_FILE = LOG_DIR / f"iadopt_preprocess_{datetime.now():%Y%m%d_%H%M%S}.log"
_preproc_logger = logging.getLogger("preprocess")
_preproc_logger.setLevel(logging.INFO)
_preproc_logger.addHandler(logging.FileHandler(PREPROC_LOG_FILE, mode="w", encoding="utf-8"))

logging.info(f"Logging to {LOG_FILE.resolve()}")
logging.info(f"Pre-processing log → {PREPROC_LOG_FILE.resolve()}")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# --------------------------------------------------------------------------- #
# 2 ▪ Prompt helpers
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
    """
    Return a complete prompt string for the LLM.
    """
    examples = examples or []
    ex_block = (
        _EXAMPLE_HDR + "\n\n".join(json.dumps(e, indent=2, ensure_ascii=False) for e in examples) if examples else ""
    )

    return (
        f"{_SYSTEM_RULES}\n\n"
        f"### JSON-Schema\n{_SCHEMA_TEXT}"
        f"{ex_block}"
        f"{_USER_HDR}label: {label}\ncomment: {comment}"
        f"{_EXPECTED}"
    )


def load_examples(n: int) -> List[Dict[str, Any]]:
    """
    Return n illustrative JSON examples (0, 1, 3 or 5).
    """
    if n == 0:
        return []

    if n == 1:
        folder = ONE_SHOT_DIR
    elif n == 3:
        folder = THREE_SHOT_DIR
    elif n == 5:
        folder = FIVE_SHOT_DIR  # ← new branch
    else:
        raise ValueError("shot must be 0, 1, 3 or 5")

    paths = sorted(folder.glob("*.json"))
    return [json.load(open(p)) for p in paths[:n]]


# --------------------------------------------------------------------------- #
# 3 ▪ LLM invocation with schema validation / retry
# --------------------------------------------------------------------------- #
# SCHEMA_OBJ = json.load(open(SCHEMA_PATH))
_JSON_FENCE_RE = re.compile(r"```(?:json)?", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Strip code fences and parse the first JSON object found.
    """
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        raise json.JSONDecodeError("No JSON block found", cleaned, 0)
    return json.loads(match.group(0))


def call_model(model: str, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            extra_headers={"X-Title": "IADOPT-bench"},
            messages=[{"role": "user", "content": prompt}],
            timeout=30,  # network timeout
        )
        return resp.choices[0].message.content

    except APIStatusError as e:  # non-2xx JSON error
        logging.warning(f"{model}: HTTP {e.status_code} – {e.body!s}")
    except (OpenAIError, httpx.HTTPError) as e:  # network / SDK issues
        logging.warning(f"{model}: transport error – {e!r}")
    except json.JSONDecodeError as e:  # just in case
        logging.warning(f"{model}: invalid JSON payload – {e!r}")

    return ""  # uniform “failure” sentinel


def coerce_for_eval(rec: Dict[str, Any], fixes: dict) -> Dict[str, Any]:
    """
    Ensure every ONTO key exists with the right type.
    Record all corrections in *fixes* (dict mutated in place).
    """
    rec = dict(rec)

    # ---- missing / extra keys -----------------------------------------
    fixes["missing_keys"] = [k for k in ONTO_KEYS if k not in rec]
    fixes["extra_keys"] = [k for k in rec.keys() if k not in {"label", "comment", *ONTO_KEYS}]

    for k in fixes["missing_keys"]:
        rec[k] = [] if k == "hasConstraint" else ""

    # ---- hasProperty came as a dict -----------------------------------
    if isinstance(rec.get("hasProperty"), dict):
        fixes["coerced_property_dict"] = True
        fixes["orig_hasProperty"] = rec["hasProperty"]  # keep full dict
        rec["hasProperty"] = rec["hasProperty"].get("label", "")
    else:
        fixes["coerced_property_dict"] = False

    return rec


def call_llm_loose(model: str, prompt: str, exp_label: str, exp_comment: str) -> Dict[str, Any]:
    """
    Invoke *model*, capture its raw reply, coerce it into a dict that respects
    the schema, and **log every fix-up** (with context) to the preprocess logger.

    The function never raises. It returns either a clean dict (possibly empty)
    or {}, which upstream treats as a blank prediction.
    """
    raw = call_model(model, prompt)
    fixes: dict[str, Any] = {
        "model": model,
        "variable": exp_label,
        "non_json_prefix": False,
        "non_json_suffix": False,
        "unparsable_json": False,
        "label_overwritten": False,
        "comment_overwritten": False,
        "coerced_property_dict": False,  # filled in coerce_for_eval
        "missing_keys": [],
        "extra_keys": [],
    }

    # ---- detect stray text before / after JSON ------------------------
    if raw:
        pre = raw.split("{", 1)[0]
        post = raw.rsplit("}", 1)[-1] if "}" in raw else ""
        if pre.strip():
            fixes["non_json_prefix"] = True
            fixes["prefix_text"] = pre.strip()[:200]  # first 200 chars
        if post.strip():
            fixes["non_json_suffix"] = True
            fixes["suffix_text"] = post.strip()[:200]

    # ---- try to isolate a JSON object ---------------------------------
    try:
        data = extract_json(raw or "")
    except json.JSONDecodeError:
        # -------- keep the entire LLM response ------------------
        fixes["unparsable_json"] = True
        fixes["raw_llm_output"] = raw  # full text
        _preproc_logger.info("%s", json.dumps(fixes, ensure_ascii=False))
        return {}  # give up – blank pred

    # ---- preserve original label/comment for logging ------------------
    orig_label = data.get("label")
    orig_comment = data.get("comment")

    # ---- force ground-truth label / comment ---------------------------
    if orig_label != exp_label:
        fixes["label_overwritten"] = True
        fixes["orig_label"] = orig_label
    if orig_comment != exp_comment:
        fixes["comment_overwritten"] = True
        fixes["orig_comment"] = (orig_comment or "")[:400]
    data["label"] = exp_label
    data["comment"] = exp_comment

    # ---- key coercions & sanitisation ---------------------------------
    data = coerce_for_eval(data, fixes=fixes)

    # ---- write the row (only enlarged when flags ≠ False) -------------
    _preproc_logger.info("%s", json.dumps(fixes, ensure_ascii=False))

    return data


# --------------------------------------------------------------------------- #
# 4 ▪ Similarity helpers
# --------------------------------------------------------------------------- #
def sim_string(a: str, b: str, close: bool) -> float:
    """String similarity (exact or cosine)."""
    if not a or not b:
        return 0.0
    norm_a, norm_b = a.lower().strip(), b.lower().strip()
    if norm_a == norm_b:
        return 1.0
    if close:
        emb1 = embed_model.encode(norm_a, convert_to_tensor=True)
        emb2 = embed_model.encode(norm_b, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()
    return 0.0


def sim_asym(a: Dict[str, str], b: Dict[str, str], close: bool) -> float:
    """Similarity for AsymmetricSystems."""
    keys = ("AsymmetricSystem", "hasSource", "hasTarget")
    return (
        sum(sim_string(a.get(k, ""), b.get(k, ""), close) for k in keys) / 3
        if isinstance(a, dict) and isinstance(b, dict)
        else 0.0
    )


def _sym_parts(obj: Any) -> Tuple[str, Set[str]]:
    if isinstance(obj, dict) and "SymmetricSystem" in obj and "hasPart" in obj:
        return obj["SymmetricSystem"], set(obj["hasPart"])
    return "", set()


def sim_sym(a: Any, b: Any, close: bool) -> float:
    lbl_a, parts_a = _sym_parts(a)
    lbl_b, parts_b = _sym_parts(b)
    if not (lbl_a or lbl_b):
        return 0.0
    label_sim = sim_string(lbl_a, lbl_b, close)
    part_sim = len(parts_a & parts_b) / len(parts_a | parts_b) if (parts_a or parts_b) else 1.0
    return (label_sim + part_sim) / 2


def sim_constraint(a: Dict[str, str], b: Dict[str, str], close: bool) -> float:
    """Similarity for constraint dicts."""
    return (
        sim_string(a.get("label", ""), b.get("label", ""), close) + sim_string(a.get("on", ""), b.get("on", ""), close)
    ) / 2


# --------------------------------------------------------------------------- #
# 5 ▪ Confusion-matrix helpers
# --------------------------------------------------------------------------- #
def confusion(gt, pred, close: bool) -> Tuple[int, int, int, int]:
    """Confusion counts for a scalar field."""
    if isinstance(gt, dict) and "AsymmetricSystem" in gt:
        score = sim_asym(gt, pred, close)
    elif isinstance(gt, dict) and "SymmetricSystem" in gt:
        score = sim_sym(gt, pred, close)
    else:
        score = sim_string(str(gt), str(pred), close)

    thr = CLOSE_THR if close else 1.0
    if gt:
        if pred and score >= thr:
            return 1, 0, 0, 0  # TP
        elif pred:
            return 0, 1, 0, 0  # FP
        else:
            return 0, 0, 1, 0  # FN
    else:
        return (0, 0, 0, 1) if not pred else (0, 1, 0, 0)  # TN or FP


def confusion_constraints(
    gt_list: List[Dict[str, str]], pred_list: List[Dict[str, str]], close: bool
) -> Tuple[float, float, float, float]:
    """
    Normalised confusion counts for the constraint list key.

    • The **whole key is worth 1.0** – never more, never less.
    • Counts can be fractional (multiples of 1/len(gt_list)).
      ─  TP  = proportion of correctly matched GT constraints
      ─  FP  = proportion of *wrong* predictions
      ─  FN  = proportion of GT constraints that received **no** prediction
    • A prediction that tries to match a GT constraint but falls below the
      threshold is treated as **FP**, not FN, so the same error is never
      penalised twice.
    """
    # ── trivial cases first ───────────────────────────────────────────────
    if not gt_list and not pred_list:
        return 0.0, 0.0, 0.0, 1.0  # perfect TN
    if not gt_list:  # GT empty but predictions exist
        return 0.0, 1.0, 0.0, 0.0  # whole key is a FP

    n_gt = len(gt_list)
    unit = 1.0 / n_gt  # weight of one GT item

    tp = fp = fn = 0.0
    used_pred: set[int] = set()

    # ---------- match each GT constraint with its best prediction --------
    for gt_c in gt_list:
        best_score = 0.0
        best_idx = -1
        for idx, p in enumerate(pred_list):
            if idx in used_pred:
                continue
            score = sim_constraint(gt_c, p, close)
            if score > best_score:
                best_score, best_idx = score, idx

        # --- classify ----------------------------------------------------
        if best_score >= (CLOSE_THR if close else 1.0):
            tp += unit  # correct prediction
            used_pred.add(best_idx)
        elif best_idx != -1:  # prediction exists but wrong
            fp += unit
            used_pred.add(best_idx)  # mark so we don't count it again
        else:  # no prediction at all
            fn += unit

    # ---------- leftover unmatched predictions are pure FP ---------------
    leftover = len(pred_list) - len(used_pred)
    fp += leftover * unit

    # sanity: ensure rounding errors don’t push the total off 1.0
    total = tp + fp + fn
    if abs(total - 1.0) > 1e-6:
        # distribute the tiny residual on FP so the sum is exactly 1
        fp += 1.0 - total

    return tp, fp, fn, 0.0  # TN only occurs in the trivial case


# --------------------------------------------------------------------------- #
# 6 ▪ Metric utilities
# --------------------------------------------------------------------------- #
def prf(tp, fp, fn) -> Tuple[float, float, float]:
    """Precision, recall, F1."""
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard index."""
    return len(a & b) / len(a | b) if a or b else 1.0


# --------------------------------------------------------------------------- #
# 7 ▪ Flatten for Jaccard
# --------------------------------------------------------------------------- #
def atoms(rec: Dict[str, Any], mode: str) -> Set[str]:
    """
    Convert a record into atomic strings
    mode ∈ {'both', 'concept', 'text'}
    """
    out: Set[str] = set()
    if mode in ("both", "concept"):
        out |= {rec.get("hasProperty", ""), rec.get("hasStatisticalModifier", "")}
        ooi = rec.get("hasObjectOfInterest")
        if isinstance(ooi, dict):
            out.add(ooi.get("AsymmetricSystem", ooi.get("SymmetricSystem", "")))
        elif ooi:
            out.add(ooi)
    if mode in ("both", "text"):
        for c in rec.get("hasConstraint", []):
            out.update({c["label"], c["on"]})
    return {s for s in out if s}


# --------------------------------------------------------------------------- #
# 8 ▪ Prompt-deduplication helpers
# --------------------------------------------------------------------------- #
_PRINTED_PROMPTS: set[tuple[int, str]] = set()  # (shot_mode, variable label)
_PROMPT_LOCK = Lock()


# --------------------------------------------------------------------------- #
# 9 ▪ Evaluation worker  (returns {"_rows": [...]})
# --------------------------------------------------------------------------- #
def _run_one(
    model: str,
    gt: Dict[str, Any],
    prompt: str,
    shot: int,
    debug_chars: int,
) -> Dict[str, Any]:

    try:
        pred = call_llm_loose(model, prompt, exp_label=gt["label"], exp_comment=gt["comment"])

        # ---------- human-readable logs -----------------------------------
        logging.info("MODEL   | %-35s | shot=%d | %s", model, shot, gt["label"])
        logging.info("GROUND-TRUTH JSON:\n%s", json.dumps(gt, indent=2, ensure_ascii=False))
        logging.info("PREDICTED    JSON:\n%s", json.dumps(pred, indent=2, ensure_ascii=False))

        rows: list[dict] = []

        # ---------- metrics for 'exact' and 'close' -----------------------
        for tag, close in (("exact", False), ("close", True)):
            tp_tot = fp_tot = fn_tot = tn_tot = 0
            per_key = {}

            for key in ONTO_KEYS:
                gt_val = gt.get(key, [] if key == "hasConstraint" else "")
                pred_val = pred.get(key, [] if key == "hasConstraint" else "")
                if key == "hasConstraint":
                    tp, fp, fn, tn = confusion_constraints(gt_val, pred_val, close)
                else:
                    tp, fp, fn, tn = confusion(gt_val, pred_val, close)

                per_key[key] = (tp, fp, fn, tn)
                tp_tot += tp
                fp_tot += fp
                fn_tot += fn
                tn_tot += tn

            logging.info(
                "%s CONFUSION | TP=%0.3f FP=%0.3f FN=%0.3f TN=%0.3f | per-key=%s",
                tag.upper(),
                tp_tot,
                fp_tot,
                fn_tot,
                tn_tot,
                per_key,
            )

            prec, rec, f1 = prf(tp_tot, fp_tot, fn_tot)
            #  ── new: Jaccard similarities (same for exact & close) ──────────────────
            j_both = jaccard(atoms(gt, "both"), atoms(pred, "both"))
            j_concept = jaccard(atoms(gt, "concept"), atoms(pred, "concept"))
            j_text = jaccard(atoms(gt, "text"), atoms(pred, "text"))

            rows.append(
                {
                    "Variable": gt["label"],
                    "Model": model,
                    "Shot": shot,
                    "Metric": tag,  # exact / close
                    "TP": tp_tot,
                    "FP": fp_tot,
                    "FN": fn_tot,
                    "TN": tn_tot,
                    "P": round(prec, 3),
                    "R": round(rec, 3),
                    "F": round(f1, 3),
                    "J_both": round(j_both, 3),
                    "J_concept": round(j_concept, 3),
                    "J_text": round(j_text, 3),
                }
            )

        return {"_rows": rows}

    except Exception as e:
        logging.error("%s | %s: worker crashed – %r", model, gt["label"], e)
        return {"_rows": []}


# --------------------------------------------------------------------------- #
# 10 ▪ Evaluation loop
# --------------------------------------------------------------------------- #
def evaluate(
    data_dir: pathlib.Path,
    shot_mode: int,
    max_vars: int = 30,
    models: List[str] | None = None,
    debug_chars: int = 500,
    workers: int = 8,
) -> List[Dict[str, Any]]:

    models = models or MODEL_NAMES
    examples = load_examples(shot_mode)
    tasks: list[tuple] = []

    # ---------- enumerate (variable, model) pairs ------------------------
    for v_idx, gt_path in enumerate(sorted(data_dir.glob("*.json")), 1):
        if max_vars and v_idx > max_vars:
            break

        gt = json.load(open(gt_path))

        if any(ex["label"] == gt["label"] for ex in examples):
            logging.info("Skip %s (in-prompt example)", gt["label"])
            continue

        prompt = build_prompt(gt["label"], gt["comment"], examples)

        with _PROMPT_LOCK:
            key = (shot_mode, gt["label"])
            if key not in _PRINTED_PROMPTS:
                logging.info(
                    "\n%s\nPROMPT | shot=%d | %s\n%s\n%s",
                    "═" * 120,
                    shot_mode,
                    gt["label"],
                    prompt,
                    "═" * 120,
                )
                _PRINTED_PROMPTS.add(key)

        for m in models:
            tasks.append((m, gt, prompt, shot_mode))

    rows: list[dict] = []

    # ---------------- parallel execution ---------------------------------
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, *task, debug_chars) for task in tasks]

        for f in as_completed(futs):
            res = f.result()
            if res and "_rows" in res:
                rows.extend(res["_rows"])

    return rows


# --------------------------------------------------------------------------- #
# 11 ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="I-ADOPT LLM benchmark – parallel & multi-shot")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR, help="Folder with ground-truth JSON files")
    parser.add_argument(
        "--shot",
        type=int,
        choices=[0, 1, 3, 5],
        default=None,
        help="Prompting mode (0 / 1 / 3 / 5). " "If omitted, all four modes are executed.",
    )
    parser.add_argument("--max-vars", type=int, default=30, help="Debug: limit number of variables")
    parser.add_argument("--only-model", action="append", help="Debug: restrict to one or more models")
    parser.add_argument("--workers", type=int, default=32, help="Parallel requests")
    args = parser.parse_args()

    # ---------------- run requested shot modes ---------------------------
    shots = [args.shot] if args.shot is not None else [0, 1, 3, 5]
    all_rows: list[dict] = []

    for s in shots:
        rows = evaluate(
            args.data_dir,
            shot_mode=s,
            max_vars=args.max_vars,
            models=args.only_model,
            workers=args.workers,
        )
        all_rows.extend(rows)

    # ------------- Excel: one file for all settings ----------------------
    df = pd.DataFrame(all_rows)

    # per-variable sheet: use exact rows only, highest-F first
    df_exact = df[df["Metric"] == "exact"]
    df_sorted = df_exact.sort_values(by="F", ascending=False)

    # --------- wide summary with exact & close side-by-side ------------------
    def pick(rowset: pd.DataFrame, col: str, metric: str):
        """Return the mean value of *col* where Metric == metric, else nan."""
        sub = rowset.loc[rowset["Metric"] == metric, col]
        return sub.mean() if not sub.empty else float("nan")

    summary_rows = []
    for (model, shot), grp in df.groupby(["Model", "Shot"]):
        summary_rows.append(
            {
                "Model": model,
                "Shot": shot,
                "F_exact": pick(grp, "F", "exact"),
                "F_close": pick(grp, "F", "close"),
                "P_exact": pick(grp, "P", "exact"),
                "P_close": pick(grp, "P", "close"),
                "R_exact": pick(grp, "R", "exact"),
                "R_close": pick(grp, "R", "close"),
                "J_both": pick(grp, "J_both", "exact"),  # the J's are identical on both rows
                "J_concept": pick(grp, "J_concept", "exact"),
                "J_text": pick(grp, "J_text", "exact"),
            }
        )

    summary = pd.DataFrame(summary_rows).round(3).sort_values(by="F_exact", ascending=False)

    out_xlsx = OUTBOOK_DIR / f"iadopt_metrics_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
        summary.to_excel(wr, sheet_name="summary", index=False)
        df_sorted.to_excel(wr, sheet_name="per_variable", index=False)
        df.to_excel(wr, sheet_name="all_rows", index=False)

    logging.info("✓ Results saved → %s", out_xlsx.resolve())

    # ------------------------------------------------------------------ #
    # NEW: wide “prompt × model” matrix of F-exact scores
    #      • Top rows  →  mean F-exact for each prompt type (0-, 1-, 3-, 5-shot)
    #      • Below     →  individual prompt rows  ( "<shot>-shot | <variable>" )
    #      • Columns   →  models, ordered (descending) by their overall mean
    # ------------------------------------------------------------------ #
    df_exact = df[df["Metric"] == "exact"].copy()

    # ---------- build the individual-prompt slice -----------------------
    df_exact["Prompt"] = df_exact["Shot"].astype(str) + "-shot | " + df_exact["Variable"]
    prompt_matrix = df_exact.pivot_table(index="Prompt", columns="Model", values="F", aggfunc="first").round(3)

    # ---------- build the mean-per-shot slice ---------------------------
    shot_mean = df_exact.pivot_table(index="Shot", columns="Model", values="F", aggfunc="mean").round(3)
    # Give the means readable row labels that will sort as 0-, 1-, 3-, 5-shot
    shot_mean.index = shot_mean.index.map(lambda s: f"{int(s)}-shot | MEAN")
    shot_mean = shot_mean.sort_index(key=lambda s: s.str.extract(r"(\d+)").astype(int)[0])

    # ---------- decide one global column order --------------------------
    # Use the *overall* mean F-exact (across *all* prompts) to rank models.
    overall_means = pd.concat([shot_mean, prompt_matrix]).mean(axis=0)
    col_order = overall_means.sort_values(ascending=False).index.tolist()

    # Apply the ordering
    shot_mean = shot_mean[col_order]
    prompt_matrix = prompt_matrix[col_order]

    # ---------- stitch the two parts together ---------------------------
    f_matrix = pd.concat([shot_mean, prompt_matrix])

    # ---------- build the flattened ranking -----------------------------
    # ❶  aggregate on (Shot, Model)  → one score per prompt-type & model
    best_pairs = (
        df_exact.groupby(["Shot", "Model"], as_index=False)["F"]  # one row = one variable
        .mean()  # mean F-exact across *all* variables
        .rename(columns={"F": "F_exact"})
        .sort_values("F_exact", ascending=False)
    )
    # prettify the shot column (e.g. 0 → "0-shot")
    best_pairs["Prompt"] = best_pairs["Shot"].astype(int).astype(str) + "-shot"
    best_pairs = best_pairs[["Prompt", "Model", "F_exact"]]

    out_matrix = OUTBOOK_DIR / f"iadopt_Fexact_matrix_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    with pd.ExcelWriter(out_matrix, engine="openpyxl") as wr:
        # original wide matrix (unchanged)
        f_matrix.to_excel(wr, sheet_name="F_exact_matrix")
        # new sheet – one row per Shot × Model pair, best first
        #       A        B              C
        # 1  Prompt   Model        F_exact
        # 2  0-shot   openai/…        0.94
        # 3  1-shot   google/…        0.91
        best_pairs.to_excel(wr, sheet_name="best_pairs", index=False)

    logging.info("✓ F-exact matrix & ranking saved → %s", out_matrix.resolve())

    # ------------------------------------------------------------------ #
    # append a one-shot summary to the preprocess log  -------------
    # ------------------------------------------------------------------ #
    counter = Counter()
    with PREPROC_LOG_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("===="):  # skip summary if re-run
                continue
            rec = json.loads(line)
            counter["rows"] += 1
            for flag in (
                "non_json_prefix",
                "non_json_suffix",
                "unparsable_json",
                "label_overwritten",
                "comment_overwritten",
                "coerced_property_dict",
            ):
                counter[flag] += bool(rec.get(flag))
            counter["missing_keys"] += len(rec.get("missing_keys", []))
            counter["extra_keys"] += len(rec.get("extra_keys", []))

    summary_lines = ["", "===== SUMMARY ====="] + [f"{k}: {v}" for k, v in counter.items()]
    with PREPROC_LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    logging.info("✓ Pre-processing log saved → %s", PREPROC_LOG_FILE.resolve())


if __name__ == "__main__":
    main()
