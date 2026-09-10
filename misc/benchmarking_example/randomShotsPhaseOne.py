#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
# I-ADOPT – randomShotsPhaseOne.py 10.12.2025
# --------------------------------------------------------------------------- #
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import product

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer, util
import random
from threading import Lock
import torch

torch.set_grad_enabled(False)

# --------------------------------------------------------------------------- #
# ▪ Static configuration
# --------------------------------------------------------------------------- #
load_dotenv()
# --------------------------------------------------------------------------- #
# ▪ Global embedder (thread-safe, preloaded)
# --------------------------------------------------------------------------- #
EMBEDDER = None

# USED_EXAMPLE_SIGNATURES = set()

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "test_set"

LOG_DIR = SCRIPT_DIR / "benchmarking_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"randomShotsPhaseOne{datetime.now():%Y%m%d_%H%M%S}.log"

OUTBOOK_DIR = pathlib.Path("benchmarking_outputs")
OUTBOOK_DIR.mkdir(exist_ok=True)

MODEL_NAMES = [
    "qwen/qwen3-32b",
    # "qwen/qwen3-8b",
    # "qwen/qwen3-30b-a3b-instruct-2507",
    # "meta-llama/llama-3-8b-instruct",
    # "openai/gpt-4o",
    # "mistralai/mistral-7b-instruct",
    # "openai/gpt-4o-mini",
    # "meta-llama/llama-3.3-70b-instruct",
    # "openai/gpt-5.1-chat",
    # "qwen/qwen3-235b-a22b-thinking-2507",
    # "qwen/qwen3-235b-a22b-2507",
]

# --------------------------------------------------------------------------- #
# ▪ Table 1 (MODE_FIXED) – full grid search
#   Fixed examples, evaluate on (test_set minus the 5 fixed vars)
#   Run ALL combinations of:
#     - all models in MODEL_NAMES
#     - all temperatures in TEMPERATURES
#     - all prompt versions found in data/prompts/*.txt
#     - all shots in FIXED_GRID_SHOTS
# --------------------------------------------------------------------------- #
# FIXED_GRID_SHOTS = [0, 1, 3, 5]
FIXED_GRID_SHOTS = [5]
# Models = MODEL_NAMES
# Temperatures = TEMPERATURES
# PromptVersions = list_prompt_versions() (computed at runtime)

TEMPERATURES = [0, 0.5, 1, 2]  # can be extended later
TEMPERATURES = [0.5]

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
# ▪ Experiment modes
# --------------------------------------------------------------------------- #
MODE_FREE = "free-grid"
MODE_FIXED = "fixed-examples-grid"
MODE_BEST = "best-random-examples"

# --------------------------------------------------------------------------- #
# ▪ Fixed example pool (hard-coded, ordered)
#   IMPORTANT: these paths must match obj["__path"] values (e.g. "test_set/xxx.json")
# --------------------------------------------------------------------------- #
FIXED_EXAMPLE_PATHS = [
    "test_set/sfcWindmax.json",
    "test_set/DetritalNitrogenConc.json",
    "test_set/HeartRate.json",
    "test_set/SoilMoist.json",
    "test_set/SurfRunoff.json",
]

# --------------------------------------------------------------------------- #
# ▪ Best config for Table 2 (hard-coded; no CLI controls)
#   Update these once you’ve decided the best configuration.
# --------------------------------------------------------------------------- #
BEST_TABLE2_CONFIG = {
    "prompt_version": "strict_minimal",
    "shot": 5,
    "temperature": 0.5,
    "models": ["qwen/qwen3-32b", "qwen/qwen3-8b"],
}

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
SCHEMA_PATH = SCRIPT_DIR / "data" / "Json_schema.json"
PROMPT_DIR = SCRIPT_DIR / "data" / "prompts"  # data/prompts

_SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def list_prompt_versions() -> List[str]:
    """
    Discover available prompt versions from data/prompts/*.txt.
    Version name = filename stem (e.g. strict_minimal.txt -> 'strict_minimal').
    """
    if not PROMPT_DIR.exists():
        logging.warning("PROMPT_DIR %s does not exist", PROMPT_DIR)
        return []
    return sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))


@lru_cache(maxsize=None)
def load_prompt_instructions(prompt_version: str) -> str:
    """
    Load the instruction text for a given prompt_version from data/prompts/<name>.txt.
    Falls back to 'strict_minimal' if the requested file is missing.
    """
    available = list_prompt_versions()

    if not available:
        raise RuntimeError(f"No prompt templates found in {PROMPT_DIR}")

    # Fallback logic
    if not prompt_version:
        prompt_version = "strict_minimal"
    if prompt_version not in available:
        logging.warning(
            "Prompt version '%s' not found in %s. Falling back to 'strict_minimal'.",
            prompt_version,
            PROMPT_DIR,
        )
        prompt_version = "strict_minimal"

    path = PROMPT_DIR / f"{prompt_version}.txt"
    return path.read_text(encoding="utf-8").strip()


_EXAMPLE_HDR = "\n\n### Examples (valid against the same schema)\n"
_USER_HDR = "\n\n### Variable's definition to decompose\n"
_EXPECTED = "\n\n### Expected output\n*(only the JSON object)*"


def build_prompt(definition: str, examples: List[Dict[str, Any]] | None, prompt_version: str) -> str:
    examples = examples or []

    # Load instructions from file (fallback handled inside)
    instructions = load_prompt_instructions(prompt_version)

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


def random_sample_examples(all_vars, k=5):
    return random.sample(all_vars, k)


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
            if key not in data or data[key] is None:
                data[key] = [] if key == "hasConstraint" else ""
            elif key == "hasConstraint" and not isinstance(data[key], list):
                data[key] = []
        return data

    # after 3 JSON extraction failures
    logging.error(f"{model}: could not extract JSON after 3 attempts")
    return {}


# --------------------------------------------------------------------------- #
# ▪ Similarity & confusion helpers (from original benchmark, simplified)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def init_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        logging.info("Loading SentenceTransformer model once (thread-safe)...")
        EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return EMBEDDER


def _cosine(a: str, b: str, model_name: str) -> float:
    embedder = init_embedder()
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

    gt_list = gt_list or []
    pred_list = pred_list or []
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
_log_lock = Lock()


def _run_one(
    mode: str,  # NEW
    model: str,
    temperature: float,
    prompt_version: str,
    gt: Dict[str, Any],
    prompt: str,
    shot: int,
    repetition: int,
    example_labels: List[str],
    example_paths: List[str],
    tested_labels: List[str],
    tested_paths: List[str],
) -> Dict[str, Any]:
    logs = []
    # Header line
    logs.append(
        "MODE | {mode} | MODEL | {model} | shot={shot} | T={temp:.2f} | prompt={pv} | "
        "variable={var} | repetition={rep}".format(
            mode=mode,
            model=model,
            shot=shot,
            temp=temperature,
            pv=prompt_version,
            var=gt["label"],
            rep=repetition,
        )
    )

    logs.append(f"EXAMPLES USED (labels): {example_labels}")
    logs.append(f"EXAMPLES USED (paths):  {example_paths}")
    logs.append(f"TESTED VARIABLES (labels): {tested_labels}")
    logs.append(f"TESTED VARIABLES (paths):  {tested_paths}")

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
        "mode": mode,
        "variable": gt["label"],
        "ground_truth_json": gt,
        "predicted_json": pred,
        "model": model,
        "temperature": temperature,
        "prompt_version": prompt_version,
        "shot": shot,
        "confusion": confusion_data,
        # NEW: metadata needed for Excel
        "repetition": repetition,
        "example_labels": example_labels,
        "example_paths": example_paths,
        "tested_labels": tested_labels,
        "tested_paths": tested_paths,
    }


def _load_all_vars(data_dir: pathlib.Path, max_vars: int) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    all_paths = sorted(data_dir.glob("*.json"))[:max_vars]
    all_vars: List[Dict[str, Any]] = []
    by_path: Dict[str, Dict[str, Any]] = {}

    for p in all_paths:
        obj = json.load(open(p, "r", encoding="utf-8"))
        obj["__path"] = str(p.relative_to(DATA_DIR.parent))  # e.g. "test_set/xxx.json"
        all_vars.append(obj)
        by_path[obj["__path"]] = obj

    return all_vars, by_path


def _resolve_fixed_examples(by_path: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    missing = [p for p in FIXED_EXAMPLE_PATHS if p not in by_path]
    if missing:
        raise RuntimeError(f"Fixed example paths not found in loaded dataset: {missing}")
    return [by_path[p] for p in FIXED_EXAMPLE_PATHS]


def _deterministic_take(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    # deterministic order by __path (stable across machines)
    items_sorted = sorted(items, key=lambda x: x.get("__path", ""))
    return items_sorted[: min(n, len(items_sorted))]


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
    num_random_sets: int,
    test_per_set: int,
    mode: str,  # NEW
) -> List[Dict[str, Any]]:

    all_vars, by_path = _load_all_vars(data_dir, max_vars)
    results: List[Dict[str, Any]] = []

    fixed_pool = _resolve_fixed_examples(by_path)  # ordered list of 5 vars
    reserved_paths_all5 = {v["__path"] for v in fixed_pool}

    # repetition handling
    reps = 1 if mode == MODE_FIXED else num_random_sets

    for rep in range(reps):
        repetition_id = rep + 1
        logging.info(f"---- Mode={mode} | Repetition {repetition_id}/{reps} ----")

        models_to_run = models or MODEL_NAMES
        tasks = []

        # ----------------------------
        # MODE_FIXED: fixed examples + grid search
        # - examples in prompt are first K fixed vars (K=shot_mode)
        # - tested set is ALWAYS (all_vars - all 5 fixed vars), regardless of shot
        # - tested set is deterministic
        # ----------------------------
        if mode == MODE_FIXED:
            if shot_mode > len(FIXED_EXAMPLE_PATHS):
                raise ValueError(f"shot_mode={shot_mode} > fixed example pool size={len(FIXED_EXAMPLE_PATHS)}")

            examples = fixed_pool[:shot_mode] if shot_mode > 0 else []
            example_labels = [ex["label"] for ex in examples]
            example_paths = [ex["__path"] for ex in examples]

            tested_pool = [v for v in all_vars if v["__path"] not in reserved_paths_all5]
            tested_vars = _deterministic_take(tested_pool, test_per_set)
            tested_labels = [v["label"] for v in tested_vars]
            tested_paths = [v["__path"] for v in tested_vars]

        # ----------------------------
        # MODE_BEST: best config (hard-coded) + random example sets + 2 models
        # - exclude the 5 fixed vars from both sampling and evaluation
        # - examples randomly sampled from remaining pool
        # - tested vars randomly sampled from remaining after excluding examples
        # ----------------------------
        elif mode == MODE_BEST:
            base_pool = [
                v for v in all_vars if v["__path"] not in reserved_paths_all5
            ]  # this mode should change the exluded variables to the ones from the random set of variables in the prompt. We have to fix this later. We are not using this mode for now.

            if shot_mode > 0:
                examples = random.sample(base_pool, k=shot_mode)
                example_labels = [ex["label"] for ex in examples]
                example_paths = [ex["__path"] for ex in examples]
            else:
                examples, example_labels, example_paths = [], [], []

            remaining = [v for v in base_pool if v["label"] not in set(example_labels)]
            tested_vars = random.sample(remaining, min(test_per_set, len(remaining)))
            tested_labels = [v["label"] for v in tested_vars]
            tested_paths = [v["__path"] for v in tested_vars]

        # ----------------------------
        # MODE_FREE: keep current behavior (random examples + random tested subset)
        # ----------------------------
        elif mode == MODE_FREE:
            if shot_mode > 0:
                examples = random_sample_examples(all_vars, k=shot_mode)
                example_labels = [ex["label"] for ex in examples]
                example_paths = [ex["__path"] for ex in examples]
            else:
                examples, example_labels, example_paths = [], [], []

            remaining = [v for v in all_vars if v["label"] not in set(example_labels)]
            if not remaining:
                logging.warning("No remaining variables to test after excluding examples.")
                continue

            tested_vars = random.sample(remaining, min(test_per_set, len(remaining)))
            tested_labels = [v["label"] for v in tested_vars]
            tested_paths = [v["__path"] for v in tested_vars]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # build tasks (models x temps x tested_vars)
        for gt in tested_vars:
            definition = gt.get("definition") or gt.get("comment")
            prompt = build_prompt(definition, examples, prompt_version)

            for model in models_to_run:
                for temp in temps:
                    tasks.append(
                        (
                            mode,  # NEW (passed into _run_one)
                            model,
                            temp,
                            prompt_version,
                            gt,
                            prompt,
                            shot_mode,
                            repetition_id,
                            example_labels,
                            example_paths,
                            tested_labels,
                            tested_paths,
                        )
                    )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run_one, *task) for task in tasks]
            for f in as_completed(futs):
                try:
                    r = f.result()
                except Exception as e:
                    logging.exception("Worker failed but run continues")
                    continue
                if r:
                    results.append(r)

    return results


# --------------------------------------------------------------------------- #
# ▪ Compute summary metrics (F_exact / F_close, per-key & overall)
# --------------------------------------------------------------------------- #
def compute_summary_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}

    for r in results:
        key = (
            r.get("mode", ""),
            r["model"],
            r["temperature"],
            r["prompt_version"],
            r["shot"],
            r["repetition"],
            tuple(r["example_labels"]),
            tuple(r["example_paths"]),
            tuple(r["tested_labels"]),
            tuple(r["tested_paths"]),
        )
        groups.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, Any]] = []

    for key, rows in groups.items():
        (
            mode,
            model,
            temp,
            prompt_version,
            shot,
            repetition,
            ex_labels_t,
            ex_paths_t,
            test_labels_t,
            test_paths_t,
        ) = key

        example_labels = list(ex_labels_t)
        example_paths = list(ex_paths_t)
        tested_labels = list(test_labels_t)
        tested_paths = list(test_paths_t)

        # overall stats across all keys
        overall = {
            "exact": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
            "close": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
        }
        # per-key stats
        per_key = {
            k: {
                "exact": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
                "close": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
            }
            for k in ONTO_KEYS
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
            for k in ONTO_KEYS:
                tp, fp, fn, tn = conf["per_key_exact"][k]
                per_key[k]["exact"]["tp"] += tp
                per_key[k]["exact"]["fp"] += fp
                per_key[k]["exact"]["fn"] += fn
                per_key[k]["exact"]["tn"] += tn

                tp, fp, fn, tn = conf["per_key_close"][k]
                per_key[k]["close"]["tp"] += tp
                per_key[k]["close"]["fp"] += fp
                per_key[k]["close"]["fn"] += fn
                per_key[k]["close"]["tn"] += tn

        row_out: Dict[str, Any] = {
            "Mode": mode,
            "Model": model,
            "Temperature": temp,
            "PromptVersion": prompt_version,
            "Shot": shot,
            "Repetition": repetition,
            "ExampleLabels": json.dumps(example_labels, ensure_ascii=False),
            "ExamplePaths": json.dumps(example_paths, ensure_ascii=False),
            "TestedLabels": json.dumps(tested_labels, ensure_ascii=False),
            "TestedPaths": json.dumps(tested_paths, ensure_ascii=False),
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
        for k in ONTO_KEYS:
            for tag in ("exact", "close"):
                tp = per_key[k][tag]["tp"]
                fp = per_key[k][tag]["fp"]
                fn = per_key[k][tag]["fn"]
                p, r, f = prf(tp, fp, fn)
                suffix = "exact" if tag == "exact" else "close"
                prefix = k  # e.g. hasMatrix
                row_out[f"{prefix}_P_{suffix}"] = round(p, 3)
                row_out[f"{prefix}_R_{suffix}"] = round(r, 3)
                row_out[f"{prefix}_F_{suffix}"] = round(f, 3)

        summary_rows.append(row_out)

    df = pd.DataFrame(summary_rows)
    if not df.empty:
        # You can sort additionally by Repetition if you like
        df = df.sort_values(by=["Model", "Shot", "Repetition", "F_exact"], ascending=[True, True, True, False])
    return df


# --------------------------------------------------------------------------- #
# ▪ Excel construction (4 sheets)
# --------------------------------------------------------------------------- #
def build_excel(results: List[Dict[str, Any]]) -> None:
    if not results:
        logging.warning("No results to write to Excel.")
        return

    out_xlsx = OUTBOOK_DIR / f"randomShotsPhaseOne{datetime.now():%Y%m%d_%H%M%S}.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:

        # 1) ONE SHEET PER ONTO_KEY
        for key in ONTO_KEYS:
            rows = []
            for r in results:
                gt = r["ground_truth_json"]
                pred = r["predicted_json"]

                rows.append(
                    {
                        "variable": r["variable"],
                        "model": r["model"],
                        "mode": r.get("mode", ""),
                        "temperature": r["temperature"],
                        "prompt_version": r["prompt_version"],
                        "shot": r["shot"],
                        "repetition": r["repetition"],
                        "example_labels": json.dumps(r["example_labels"], ensure_ascii=False),
                        "example_paths": json.dumps(r["example_paths"], ensure_ascii=False),
                        "ground_truth": json.dumps(gt.get(key, ""), ensure_ascii=False, indent=2),
                        "predicted": json.dumps(pred.get(key, ""), ensure_ascii=False, indent=2),
                    }
                )

            df_key = pd.DataFrame(rows)
            sheet_name = f"{key} concepts"
            df_key.to_excel(wr, sheet_name=sheet_name[:31], index=False)

        # 2) Full LLM outputs
        json_rows = []
        for r in results:
            json_rows.append(
                {
                    "variable": r["variable"],
                    "model": r["model"],
                    "mode": r.get("mode", ""),
                    "temperature": r["temperature"],
                    "prompt_version": r["prompt_version"],
                    "shot": r["shot"],
                    "repetition": r["repetition"],
                    "example_labels": json.dumps(r["example_labels"], ensure_ascii=False),
                    "example_paths": json.dumps(r["example_paths"], ensure_ascii=False),
                    "ground_truth_json": json.dumps(r["ground_truth_json"], ensure_ascii=False, indent=2),
                    "predicted_json": json.dumps(r["predicted_json"], ensure_ascii=False, indent=2),
                }
            )
        df_json = pd.DataFrame(json_rows)
        df_json.to_excel(wr, sheet_name="LLM outputs", index=False)

        # 3) Summary metrics (with example + tested sets)
        df_summary = compute_summary_metrics(results)
        df_summary.to_excel(wr, sheet_name="Summary", index=False)

    logging.info("✓ Results saved → %s", out_xlsx.resolve())


# --------------------------------------------------------------------------- #
# ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    init_embedder()
    parser = argparse.ArgumentParser(description="I-ADOPT LLM Phase 1 – Matrix & Constraint comparison + evaluation")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--only-model", action="append", help="Debug: restrict to one or more models")
    parser.add_argument("--max-vars", type=int, default=105)
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers to use.")
    parser.add_argument("--random-sets", type=int, default=10, help="Number of random example sets to run per shot.")

    parser.add_argument(
        "--shot",
        type=int,
        choices=[0, 1, 3, 5],
        default=5,
        help="If provided: run only this shot. If omitted: run all shots.",
    )

    parser.add_argument(
        "--prompt-version",
        type=str,
        choices=list_prompt_versions() or None,  # if no files, allow anything and fallback
        # default="strict_minimal",
        help="If provided: run only this prompt version. If omitted: run all prompt versions found in data/prompts.",
    )

    parser.add_argument(
        "--test-per-set", type=int, default=105, help="Number of GT variables to evaluate per random-shot set."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MODE_FREE, MODE_FIXED, MODE_BEST],
        default=MODE_FREE,
        help="Experiment mode",
    )

    args = parser.parse_args()

    # prompt versions
    if args.mode == MODE_BEST:
        prompt_versions = [BEST_TABLE2_CONFIG["prompt_version"]]
    elif args.mode == MODE_FIXED:
        prompt_versions = list_prompt_versions() or ["strict_minimal"]  # ALL prompts
    else:
        prompt_versions = (
            [args.prompt_version] if args.prompt_version else (list_prompt_versions() or ["strict_minimal"])
        )

    # shots
    if args.mode == MODE_BEST:
        shots = [BEST_TABLE2_CONFIG["shot"]]
    elif args.mode == MODE_FIXED:
        shots = FIXED_GRID_SHOTS  # ALL shots
    else:
        shots = [args.shot]

    # models + temps
    if args.mode == MODE_BEST:
        temps = [BEST_TABLE2_CONFIG["temperature"]]
        models = BEST_TABLE2_CONFIG["models"]
    else:
        temps = TEMPERATURES
        models = MODEL_NAMES if args.mode == MODE_FIXED else args.only_model

    # Full evaluation grid
    all_results: List[Dict[str, Any]] = []

    for pv in prompt_versions:
        for shot in shots:
            print(f"\n=== Running prompt_version={pv} | shot={shot} ===\n")

            results_single = evaluate(
                data_dir=args.data_dir,
                shot_mode=shot,
                prompt_version=pv,
                max_vars=args.max_vars,
                models=models,
                temps=temps,
                workers=args.workers,
                num_random_sets=(1 if args.mode == MODE_FIXED else args.random_sets),
                test_per_set=args.test_per_set,
                mode=args.mode,
            )

            all_results.extend(results_single)

    build_excel(all_results)


if __name__ == "__main__":
    main()
