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

MODEL_NAMES = [
    "qwen/qwen3-32b",
    # "qwen/qwen3-30b-a3b-instruct-2507",
    "meta-llama/llama-4-maverick",
    # "meta-llama/llama-3.3-70b-instruct",
    # "openai/gpt-5.1-chat",
    "qwen/qwen3-235b-a22b-thinking-2507",
]
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
    Follow the JSON-Schema exactly. Do not infer or invent new concepts.

    definition must be exactly the same string as provided.
    comment = short summary of the definition. Do not add new ideas.

    hasProperty = the main measurable property in the definition.
    hasObjectOfInterest = the thing that has this property.
    hasMatrix = the medium in which the object occurs. Never a method or location.

    If a required key is not in the definition, output an empty string for it.

    Output only the JSON object.
"""
).strip()

BASELINE_INSTRUCTIONS = _SYSTEM_RULES

STRICT_MINIMAL_GUIDE = textwrap.dedent(
    """
Additional rules:

• Only extract what is explicitly stated in the definition.
• hasProperty = the main measurable characteristic.
• hasObjectOfInterest = the thing that has that characteristic.
• hasMatrix = medium the object is in, only if directly stated.
• hasConstraint = only conditions explicitly stated.
• If unsure: leave it empty. Do not guess.
"""
).strip()

OBJ_MATRIX_TREE = textwrap.dedent(
    """
Decision rules:

1. Identify hasProperty first.
2. Identify hasObjectOfInterest:
   → the entity that carries the property.
3. Identify hasMatrix only if the definition clearly states
   the medium or material the object is inside.
4. If a phrase describes a condition/state, not a medium:
   → put it in hasConstraint.
5. Never use methods, units, instruments, or locations.
"""
).strip()

CONSTRAINT_FIRST_GUIDE = textwrap.dedent(
    """
Extraction order:

1. Copy definition exactly.
2. Extract hasProperty (main measurable characteristic).
3. Extract hasObjectOfInterest (entity with that property).
4. Extract hasMatrix only if the definition states a medium.
5. Extract hasConstraint last:
   • Only explicit limiting phrases.
   • label = short phrase
   • on = EXACT string from hasProperty or an entity
6. Never paraphrase or introduce new concepts.
"""
).strip()


PROMPT_TEMPLATES = {
    "strict_minimal": BASELINE_INSTRUCTIONS + "\n\n" + STRICT_MINIMAL_GUIDE,
    "object_matrix_tree": BASELINE_INSTRUCTIONS + "\n\n" + OBJ_MATRIX_TREE,
    "constraint_first": BASELINE_INSTRUCTIONS + "\n\n" + CONSTRAINT_FIRST_GUIDE,
}

_EXAMPLE_HDR = "\n\n### Examples (valid against the same schema)\n"
_USER_HDR = "\n\n### Variable's definition to decompose\n"
_EXPECTED = "\n\n### Expected output\n*(only the JSON object)*"


def build_prompt(definition: str, examples: List[Dict[str, Any]] | None, prompt_version: str) -> str:
    examples = examples or []

    # Pick the template by name (fallback = strict_minimal)
    instructions = PROMPT_TEMPLATES.get(prompt_version, PROMPT_TEMPLATES["strict_minimal"])

    ex_block = (
        _EXAMPLE_HDR + "\n\n".join(json.dumps(e, indent=2, ensure_ascii=False) for e in examples) if examples else ""
    )

    return (
        f"{instructions}\n\n"
        f"### JSON-Schema\n{_SCHEMA_TEXT}\n"
        f"{ex_block}"
        f"{_USER_HDR}definition: {definition}"
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
    """
    Robust API call with:
    - 3 retries
    - detection of HTML / empty responses
    - logs raw response on failure
    - returns "" if no valid content after retries
    """
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            text = resp.choices[0].message.content or ""

            # detect HTML or empty response
            if text.strip().startswith("<!DOCTYPE html") or text.strip().startswith("<html"):
                logging.warning(f"{model}: HTML error response on attempt {attempt}")
                continue

            if not text.strip():
                logging.warning(f"{model}: empty response on attempt {attempt}")
                continue

            return text

        except APIStatusError as e:
            logging.warning(f"{model}: APIStatusError attempt {attempt} – {e.status_code} – {getattr(e, 'body', '')}")
        except (OpenAIError, httpx.HTTPError) as e:
            logging.warning(f"{model}: transport error attempt {attempt} – {e!r}")
        except Exception as e:
            logging.warning(f"{model}: unexpected error attempt {attempt} – {e!r}")

    # after 3 failures
    logging.error(f"{model}: failed after 3 attempts")
    return ""


def call_llm_loose(model: str, prompt: str, definition: str, temperature: float) -> Dict[str, Any]:
    """
    - Calls call_model() with 3 retries
    - Extracts JSON robustly
    - Returns {} if no valid JSON after 3 attempts
    """
    for attempt in range(1, 4):
        raw = call_model(model, prompt, temperature)

        if not raw.strip():
            logging.warning(f"{model}: empty output on JSON extraction attempt {attempt}")
            continue

        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            logging.warning(f"{model}: no JSON block found on attempt {attempt}")
            continue

        try:
            data = json.loads(m.group(0))
        except Exception as e:
            logging.warning(f"{model}: JSON decode failure on attempt {attempt} – {e!r}")
            continue

        # success → post-process and return
        data["definition"] = definition
        for key in ONTO_KEYS:
            if key not in data:
                data[key] = [] if key == "hasConstraint" else ""
        return data

    # after 3 JSON extraction failures
    logging.error(f"{model}: could not extract JSON after 3 attempts")
    return {}


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


def _asym_parts(obj: Any) -> tuple[str, str]:
    """
    Extract the meaningful parts of an AsymmetricSystem for evaluation.

    We IGNORE the 'AsymmetricSystem' label (blank node ID, meaningless for scoring)
    and only use:
        - hasNumerator / hasDenominator  (preferred)
        - or hasSource / hasTarget       (fallback for older data)
    """
    if not isinstance(obj, dict):
        return "", ""

    numerator = obj.get("hasNumerator") or obj.get("hasSource") or ""
    denominator = obj.get("hasDenominator") or obj.get("hasTarget") or ""
    return numerator, denominator


def sim_asym(a: Dict[str, Any], b: Dict[str, Any], close: bool) -> float:
    """
    Similarity for AsymmetricSystem objects.

    We intentionally IGNORE the 'AsymmetricSystem' identifier and only compare:
      - hasNumerator vs hasNumerator  (preferred)
      - hasDenominator vs hasDenominator
      (or hasSource/hasTarget as fallback keys)

    Examples considered 100% correct:
      GT:  { "AsymmetricSystem": "bc_0_b6_b0",
             "hasNumerator": "water",
             "hasDenominator": "atmosphere" }

      PRED:{ "AsymmetricSystem": "",
             "hasNumerator": "water",
             "hasDenominator": "atmosphere" }

      PRED:{ "AsymmetricSystem": "whatever here",
             "hasNumerator": "water",
             "hasDenominator": "atmosphere" }
    """
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return 0.0

    num_a, den_a = _asym_parts(a)
    num_b, den_b = _asym_parts(b)

    # If there is no meaningful numerator/denominator, nothing to compare
    if not (num_a or den_a or num_b or den_b):
        return 0.0

    num_sim = sim_string(num_a, num_b, close)
    den_sim = sim_string(den_a, den_b, close)

    # Average of numerator + denominator similarity
    return (num_sim + den_sim) / 2


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


def normalize_constraint(c: Dict[str, str]) -> Dict[str, str]:
    """
    Canonicalize label/on for robust, order-invariant comparison.
    - Lowercase
    - Strip whitespace
    - Collapse multiple spaces
    - Apply canonical_on() to the 'on' field
    """
    if not isinstance(c, dict):
        return {"label": "", "on": ""}

    def norm_label(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def norm_on(s: str) -> str:
        s = canonical_on(s or "")
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    return {
        "label": norm_label(c.get("label", "")),
        "on": norm_on(c.get("on", "")),
    }


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
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[float, float, float, float]:
    """
    Constraint confusion with:
    - canonicalized 'label' and 'on'
    - greedy matching on similarity matrix (order-invariant)
    - same TP/FP/FN scaling as original implementation
    """

    # trivial cases
    if not gt_list and not pred_list:
        return 0.0, 0.0, 0.0, 1.0
    if not gt_list:
        # no GT but some predictions => all FP
        return 0.0, 1.0, 0.0, 0.0

    n_gt, n_pred = len(gt_list), len(pred_list)
    unit = 1.0 / (2 * n_gt)  # same scaling as before
    thr = CLOSE_THR if close else 1.0

    # normalize constraints (label & on)
    gt_norm = [normalize_constraint(c) for c in gt_list]
    pred_norm = [normalize_constraint(c) for c in pred_list]

    # similarity matrix S[i, j] based on normalized label+on
    S = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt_norm):
        for j, p in enumerate(pred_norm):
            lbl_sim = sim_string(g["label"], p["label"], close, model_name)
            on_sim = sim_string(g["on"], p["on"], close, model_name)
            S[i, j] = (lbl_sim + on_sim) / 2.0

    tp = fp = fn = 0.0
    gt_used: set[int] = set()
    pred_used: set[int] = set()

    # greedy matching on S (order-invariant but not index-based)
    while S.size:
        idx = int(np.argmax(S))
        i, j = divmod(idx, S.shape[1])
        if S[i, j] < 0:
            break

        gt_used.add(i)
        pred_used.add(j)

        # label contribution
        if sim_string(gt_norm[i]["label"], pred_norm[j]["label"], close, model_name) >= thr:
            tp += unit
        else:
            fp += unit

        # 'on' contribution
        if sim_string(gt_norm[i]["on"], pred_norm[j]["on"], close, model_name) >= thr:
            tp += unit
        else:
            fp += unit

        # mask row/col as used
        S[i, :] = -1.0
        S[:, j] = -1.0

    # remaining GT → FN (for label + on)
    fn += (n_gt - len(gt_used)) * 2 * unit
    # remaining predictions → FP (for label + on)
    fp += (n_pred - len(pred_used)) * 2 * unit

    # small numerical correction (keep tp+fp+fn ≈ 1.0)
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


def compute_confusion_for_pair(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes confusion scores ONCE for a single GT/PRED pair.
    Returns both per-key and aggregated totals for exact and close.
    """
    exact = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    close = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    per_key_exact = {}
    per_key_close = {}

    for key in ONTO_KEYS:
        gt_val = gt.get(key, "" if key != "hasConstraint" else [])
        pred_val = pred.get(key, "" if key != "hasConstraint" else [])

        # exact
        if key == "hasConstraint":
            tp, fp, fn, tn = confusion_constraints(gt_val, pred_val, close=False)
        else:
            tp, fp, fn, tn = confusion(gt_val, pred_val, close=False)
        per_key_exact[key] = (tp, fp, fn, tn)
        exact["tp"] += tp
        exact["fp"] += fp
        exact["fn"] += fn
        exact["tn"] += tn

        # close
        if key == "hasConstraint":
            tp2, fp2, fn2, tn2 = confusion_constraints(gt_val, pred_val, close=True)
        else:
            tp2, fp2, fn2, tn2 = confusion(gt_val, pred_val, close=True)
        per_key_close[key] = (tp2, fp2, fn2, tn2)
        close["tp"] += tp2
        close["fp"] += fp2
        close["fn"] += fn2
        close["tn"] += tn2

    return {
        "exact_totals": exact,
        "close_totals": close,
        "per_key_exact": per_key_exact,
        "per_key_close": per_key_close,
    }


# --------------------------------------------------------------------------- #
# ▪ Phase 1 worker (PATCHED FOR ORDERED LOGGING)
# --------------------------------------------------------------------------- #
from threading import Lock

_log_lock = Lock()


def _run_one(
    model: str,
    temperature: float,
    prompt_version: str,
    gt: Dict[str, Any],
    prompt: str,
    shot: int,
) -> Dict[str, Any]:

    logs = []
    # Header line
    logs.append(
        f"MODEL | {model} | shot={shot} | T={temperature:.2f} | " f"prompt={prompt_version} | variable={gt['label']}"
    )

    logs.append(f"PROMPT:\n{prompt}")
    logs.append("GROUND-TRUTH JSON (GT):\n" + json.dumps(gt, indent=2, ensure_ascii=False))

    definition = gt.get("definition") or gt.get("comment")
    pred = call_llm_loose(model, prompt, definition, temperature)

    logs.append("PREDICTED JSON:\n" + json.dumps(pred, indent=2, ensure_ascii=False))

    # compute confusion ONCE
    confusion_data = compute_confusion_for_pair(gt, pred)
    exact = confusion_data["exact_totals"]
    close = confusion_data["close_totals"]

    logs.append(
        "EXACT CONFUSION | "
        f"TP={exact['tp']:.3f} FP={exact['fp']:.3f} "
        f"FN={exact['fn']:.3f} TN={exact['tn']:.3f} | "
        f"per-key={confusion_data['per_key_exact']}"
    )

    logs.append(
        "CLOSE CONFUSION | "
        f"TP={close['tp']:.3f} FP={close['fp']:.3f} "
        f"FN={close['fn']:.3f} TN={close['tn']:.3f} | "
        f"per-key={confusion_data['per_key_close']}"
    )

    # PRINT ATOMICALLY — NO INTERLEAVING BETWEEN THREADS
    with _log_lock:
        logging.info("\n".join(logs))

    return {
        "variable": gt["label"],
        "ground_truth_json": gt,
        "predicted_json": pred,
        "model": model,
        "temperature": temperature,
        "prompt_version": prompt_version,
        "shot": shot,
        "confusion": confusion_data,
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
    models = models or MODEL_NAMES

    # -------------------------------------------------------------
    # For every variable: build *its* prompt, then create tasks
    # -------------------------------------------------------------
    for gt_path in sorted(data_dir.glob("*.json"))[:max_vars]:

        try:
            gt = json.load(open(gt_path))
        except Exception as e:
            logging.error(f"Failed to load {gt_path}: {e}")
            continue

        # avoid few-shot leakage
        if any(ex["label"] == gt["label"] for ex in examples):
            continue

        definition = gt.get("definition") or gt.get("comment")
        prompt = build_prompt(definition, examples, prompt_version)

        # -------------------------------------------------------------
        # NOW create tasks for this GT, not after the GT-loop!
        # -------------------------------------------------------------
        for model in models:
            for temp in temps:
                tasks.append((model, temp, prompt_version, gt, prompt, shot_mode))

    # -------------------------------------------------------------
    # Run all tasks in a thread pool
    # -------------------------------------------------------------
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
            conf = row["confusion"]

            # overall totals
            for tag, totals in (("exact", conf["exact_totals"]), ("close", conf["close_totals"])):
                overall[tag]["tp"] += totals["tp"]
                overall[tag]["fp"] += totals["fp"]
                overall[tag]["fn"] += totals["fn"]
                overall[tag]["tn"] += totals["tn"]

            # per-key totals
            for key in ONTO_KEYS:
                tp, fp, fn, tn = conf["per_key_exact"][key]
                per_key[key]["exact"]["tp"] += tp
                per_key[key]["exact"]["fp"] += fp
                per_key[key]["exact"]["fn"] += fn
                per_key[key]["exact"]["tn"] += tn

                tp, fp, fn, tn = conf["per_key_close"][key]
                per_key[key]["close"]["tp"] += tp
                per_key[key]["close"]["fp"] += fp
                per_key[key]["close"]["fn"] += fn
                per_key[key]["close"]["tn"] += tn

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

    # Prepare output file
    out_xlsx = OUTBOOK_DIR / f"onlyPhaseOne{datetime.now():%Y%m%d_%H%M%S}.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:

        # -------------------------------------------------------------
        # 1) ONE SHEET PER ONTO_KEY
        # -------------------------------------------------------------
        for key in ONTO_KEYS:
            rows = []
            for r in results:
                gt = r["ground_truth_json"]
                pred = r["predicted_json"]

                rows.append(
                    {
                        "variable": r["variable"],
                        "model": r["model"],
                        "temperature": r["temperature"],
                        "prompt_version": r["prompt_version"],
                        "shot": r["shot"],
                        "ground_truth": json.dumps(gt.get(key, ""), ensure_ascii=False, indent=2),
                        "predicted": json.dumps(pred.get(key, ""), ensure_ascii=False, indent=2),
                    }
                )

            df_key = pd.DataFrame(rows)
            sheet_name = f"{key} concepts"
            df_key.to_excel(wr, sheet_name=sheet_name[:31], index=False)
            # Excel sheet names must be <= 31 chars

        # -------------------------------------------------------------
        # 2) Full LLM outputs (existing)
        # -------------------------------------------------------------
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
        df_json.to_excel(wr, sheet_name="LLM outputs", index=False)

        # -------------------------------------------------------------
        # 3) Summary metrics (existing)
        # -------------------------------------------------------------
        df_summary = compute_summary_metrics(results)
        df_summary.to_excel(wr, sheet_name="Summary", index=False)

    logging.info("✓ Results saved → %s", out_xlsx.resolve())


# --------------------------------------------------------------------------- #
# ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="I-ADOPT LLM Phase 1 – Matrix & Constraint comparison + evaluation")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--only-model", action="append", help="Debug: restrict to one or more models")
    parser.add_argument("--max-vars", type=int, default=105)
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
                models=args.only_model,
                temps=TEMPERATURES,
                workers=args.workers,
            )

            all_results.extend(results)

    build_excel(all_results)


if __name__ == "__main__":
    main()
