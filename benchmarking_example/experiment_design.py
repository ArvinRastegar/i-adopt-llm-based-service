#!/usr/bin/env python3
"""
experiment_design.py • 2025-06-12
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
import argparse, json, logging, os, pathlib, re, textwrap
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
import pandas as pd
from openai import BadRequestError, OpenAI
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed

# from jsonschema import validate, ValidationError


# ----- static config -------------------------------------------------------- #
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCHEMA_PATH = "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/benchmarking_example/benchmarking_outputs/Json_schema.json"
DATA_DIR = SCRIPT_DIR / "data" / "Json_preferred"
ONE_SHOT_DIR = DATA_DIR / "one_shot"
THREE_SHOT_DIR = DATA_DIR / "three_shot"
FIVE_SHOT_DIR = DATA_DIR / "five_shot"
OUTBOOK_DIR = SCRIPT_DIR / "benchmarking_outputs"
OUTBOOK_DIR.mkdir(exist_ok=True)
LOG_FILE = OUTBOOK_DIR / f"iadopt_run_{datetime.now():%Y%m%d_%H%M%S}.log"

# MODEL_NAMES = [
#     "openai/gpt-4o",
#     "openai/gpt-4o-mini",
#     "openai/gpt-4.1-mini",
#     "openai/gpt-4.1",
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "Qwen/Qwen3-8B",
#     "allenai/OLMo-2-0425-1B-Instruct",
#     "allenai/OLMo-2-1124-7B-Instruct",
#     "allenai/OLMo-2-1124-13B-Instruct",
#     "allenai/OLMo-2-0325-32B-Instruct",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
# ]

MODEL_NAMES = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "google/gemma-2-9b-it",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b",
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
# ─── LOGGING: console  +  file --- everything duplicated -──────────────────── #
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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# --------------------------------------------------------------------------- #
# 2 ▪ Prompt helpers
# --------------------------------------------------------------------------- #

SCHEMA_PATH = Path(
    "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/"
    "benchmarking_example/benchmarking_outputs/Json_schema.json"
)
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
    Return a complete prompt string ready to send to the LLM.

    Parameters
    ----------
    label, comment : str
        Ground-truth values that must be copied unchanged.

    examples : list[dict] | None
        Zero, one, or three example JSON objects that are already
        compliant with the schema.  They will be rendered verbatim.
    """
    examples = examples or []
    ex_block = (
        _EXAMPLE_HDR + "\n\n".join(json.dumps(e, indent=2, ensure_ascii=False) for e in examples) if examples else ""
    )

    # assemble
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
SCHEMA_OBJ = json.load(open(SCHEMA_PATH))


def extract_json(text: str) -> Dict[str, Any]:
    """
    Strip code fences and parse the first JSON object found.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON block found", cleaned, 0)
    return json.loads(match.group(0))


def call_model(model: str, prompt: str) -> str:
    """
    Low-level OpenRouter call. Returns raw assistant text or "" on HTTP-error.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            extra_headers={"X-Title": "IADOPT-bench"},
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except BadRequestError as e:
        logging.error(f"{model}: {e}")
        return ""


def coerce_for_eval(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee that *every* ONTO_KEY exists with the right baseline type."""
    rec = dict(rec)
    for k in ONTO_KEYS:
        if k not in rec:
            rec[k] = [] if k == "hasConstraint" else ""
    # common hallucination: property given as {"label": "..."}
    if isinstance(rec["hasProperty"], dict):
        rec["hasProperty"] = rec["hasProperty"].get("label", "")
    return rec


# def call_llm_schema_safe(model: str, prompt: str, exp_label: str, exp_comment: str) -> Dict[str, Any]:
#     """
#     Call *model* (≤MAX_RETRY) until the JSON passes schema validation.
#     The returned dict is *forced* to have the expected label & comment.
#     """
#     for attempt in range(1, MAX_RETRY + 1):
#         raw = call_model(model, prompt)
#         if not raw:
#             break
#         try:
#             data = extract_json(raw)
#             validate(data, SCHEMA_OBJ)
#             # ── enforce identical label & comment ───────────────────────────
#             if data.get("label") != exp_label:
#                 logging.warning(f"{model}: overwrote label “{data.get('label')}”")
#             if data.get("comment") != exp_comment:
#                 logging.warning(f"{model}: overwrote comment")
#             data["label"] = exp_label
#             data["comment"] = exp_comment
#             return data
#         except (json.JSONDecodeError, ValidationError) as err:
#             logging.warning(f"{model}: schema fail {attempt}/{MAX_RETRY}: {err}")
#             time.sleep(1)
#     logging.error(f"{model}: gave up after {MAX_RETRY} retries")
#     return {}


def call_llm_loose(model: str, prompt: str, exp_label: str, exp_comment: str) -> Dict[str, Any]:
    """
    Call *model* once and return the JSON we can parse.

    • No json-schema validation.
    • If we cannot parse anything → return {}.
    • label / comment are overwritten with the expected ground-truth values.
    """
    raw = call_model(model, prompt)
    if not raw:  # model failed (HTTP 4xx/5xx etc.)
        return {}

    try:
        data = extract_json(raw)
    except json.JSONDecodeError:
        logging.warning(f"{model}: could not parse JSON – using empty dict")
        return {}

    # ── enforce the fixed fields ────────────────────────────────────────────
    if data.get("label") != exp_label:
        logging.info(f"{model}: label changed → forcing back to GT")
        data["label"] = exp_label
    if data.get("comment") != exp_comment:
        logging.info(f"{model}: comment changed → forcing back to GT")
        data["comment"] = exp_comment

    return coerce_for_eval(data)


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
    """Confusion counts for a single scalar field."""
    if isinstance(gt, dict) and "AsymmetricSystem" in gt:
        score = sim_asym(gt, pred, close)
    elif isinstance(gt, dict) and "SymmetricSystem" in gt:
        score = sim_sym(gt, pred, close)
    else:
        score = sim_string(str(gt), str(pred), close)

    thr = CLOSE_THR if close else 1.0
    if gt:
        if pred and score >= thr:
            return 1, 0, 0, 0
        return 0, 0, 1, 0
    else:
        return (0, 1, 0, 0) if pred else (0, 0, 0, 1)


def confusion_constraints(gt_list, pred_list, close):
    """Confusion for lists of constraints."""
    tp = fn = 0
    for gt_c in gt_list:
        best = max((sim_constraint(gt_c, p, close) for p in pred_list), default=0.0)
        if best >= (CLOSE_THR if close else 1.0):
            tp += 1
        else:
            fn += 1
    fp = max(0, len(pred_list) - tp)
    return tp, fp, fn, 0


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
# 8 ▪ Evaluation loop  (returns list[dict] instead of writing files)
# --------------------------------------------------------------------------- #
def _run_one(
    model: str,
    gt: Dict[str, Any],
    examples: List[Dict[str, Any]],
    debug_chars: int,
) -> Dict[str, Any]:
    """
    Worker executed in a background thread.

    Returns
    -------
    dict   A single “row” for the final dataframe or {} if the call failed.
    """
    prompt = build_prompt(gt["label"], gt["comment"], examples)
    pred = call_llm_loose(model, prompt, exp_label=gt["label"], exp_comment=gt["comment"])

    logging.info(  # verbose dev-log
        "· %s | %s\nprompt>>> %.{}s …\nparsed=%s".format(debug_chars),
        gt["label"],
        model,
        prompt,
        json.dumps(pred)[:debug_chars],
    )

    # ---------------------- metric calc (unchanged) -------------------------
    metrics: Dict[str, float] = {}
    for close in (False, True):
        tp = fp = fn = 0
        for key in ONTO_KEYS:
            gt_val = gt.get(key, [] if key == "hasConstraint" else "")
            pred_val = pred.get(key, [] if key == "hasConstraint" else "")
            if key == "hasConstraint":
                tpi, fpi, fni, _ = confusion_constraints(gt_val, pred_val, close)
            else:
                tpi, fpi, fni, _ = confusion(gt_val, pred_val, close)
            tp += tpi
            fp += fpi
            fn += fni
        prec, rec, f1 = prf(tp, fp, fn)
        tag = "close" if close else "exact"
        metrics |= {f"P_{tag}": round(prec, 3), f"R_{tag}": round(rec, 3), f"F_{tag}": round(f1, 3)}

    for mode in ("both", "concept", "text"):
        metrics[f"J_{mode}"] = round(jaccard(atoms(gt, mode), atoms(pred, mode)), 3)

    return {"Variable": gt["label"], "Model": model, **metrics}


def evaluate(
    data_dir: pathlib.Path,
    shot_mode: int,
    max_vars: int = 30,
    models: List[str] | None = None,
    debug_chars: int = 500,
    workers: int = 16,
) -> List[Dict[str, Any]]:

    models = models or MODEL_NAMES
    examples = load_examples(shot_mode)
    tasks = []

    # ---------- enumerate (variable, model) pairs --------------------------
    for v_idx, gt_path in enumerate(sorted(data_dir.glob("*.json")), 1):
        if max_vars and v_idx > max_vars:
            break
        gt = json.load(open(gt_path))

        if any(ex["label"] == gt["label"] for ex in examples):
            logging.info(f"Skip {gt['label']} (in-prompt example)")
            continue

        for m in models:
            tasks.append((m, gt))

    rows: list[dict] = []

    # ------------------ parallel execution ---------------------------------
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, m, g, examples, debug_chars) for m, g in tasks]
        for f in as_completed(futs):
            res = f.result()
            if res:
                rows.append(res)

    return rows  # <-- caller handles aggregation / XLSX


# --------------------------------------------------------------------------- #
# 9 ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser(description="I-ADOPT LLM benchmark – now parallel & multi-shot")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR, help="Folder with ground-truth JSON files")
    parser.add_argument(
        "--shot",
        type=int,
        choices=[0, 1, 3, 5],  # ← added 5
        default=None,  # omit → run all four
        help="Prompting mode (0 / 1 / 3 / 5). " "If omitted, all four modes are executed.",
    )
    parser.add_argument("--max-vars", type=int, default=30, help="Debug: limit #variables")  # ← default 10
    parser.add_argument("--only-model", action="append", help="Debug: restrict to one or more models")
    parser.add_argument("--workers", type=int, default=16, help="Parallel requests")  # ← default 8
    args = parser.parse_args()

    # ------------------------------------------------------------------ run
    # run all requested shot modes
    shots = [args.shot] if args.shot is not None else [0, 1, 3, 5]

    all_rows: list[dict] = []

    for s in shots:
        rows = evaluate(
            args.data_dir, shot_mode=s, max_vars=args.max_vars, models=args.only_model, workers=args.workers
        )
        for r in rows:  # tag each row with the approach used
            r["Shot"] = s
        all_rows.extend(rows)

    # ------------- single Excel containing *all* shot-settings ----------
    df = pd.DataFrame(all_rows)
    summary = df.groupby(["Model", "Shot"]).mean(numeric_only=True).round(3).reset_index()
    # ← aggregate over all shots

    out_xlsx = OUTBOOK_DIR / f"iadopt_metrics_{datetime.now():%Y%m%d}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
        summary.to_excel(wr, sheet_name="summary", index=False)
        df.to_excel(wr, sheet_name="per_variable", index=False)
    logging.info(f"✓ Results saved → {out_xlsx.resolve()}")


if __name__ == "__main__":
    main()
