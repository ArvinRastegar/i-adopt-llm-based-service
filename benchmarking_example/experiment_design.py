#!/usr/bin/env python3
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
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from functools import lru_cache
from itertools import product
import numpy as np
from dotenv import load_dotenv
import requests
import urllib.parse

from datasets import load_dataset

import torch

import requests_cache

session = requests_cache.CachedSession('wikidata_cache')

load_dotenv()

# from jsonschema import validate, ValidationError

# ----- static config -------------------------------------------------------- #
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPT_DIR / "data" / "Json_schema.json"
DATA_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "test_set"
# DATA_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "random_splits" / "0" / "test_set"
ONE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/one_shot"
THREE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/three_shot"
FIVE_SHOT_DIR = SCRIPT_DIR / "data/Json_preferred/five_shot"
# FIVE_SHOT_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "random_splits" / "4" / "five_shot"

LOG_DIR = SCRIPT_DIR / "benchmarking_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"iadopt_run_{datetime.now():%Y%m%d_%H%M%S}.log"

MODEL_NAMES = [
    # "openai/gpt-4o",
    # "openai/gpt-4o-mini",
    # "openai/gpt-4.1",
    # "meta-llama/llama-3.3-8b-instruct",
    # "qwen/qwen3-8b",
    # "microsoft/phi-4",
    # "microsoft/phi-4-reasoning",
    # "open-orca/mistral-7b-openorca",
    # "deepseek/deepseek-r1-0528-qwen3-8b",
    # "deepseek/deepseek-r1-0528",
    # "deepseek/deepseek-chat-v3-0324",
    # "qwen/qwen3-235b-a22b",
    # "google/gemini-2.0-flash-001",
    # "deepseek/deepseek-r1-distill-qwen-14b",
    # "deepseek/deepseek-r1-distill-qwen-32b",
    # "qwen/qwen-2.5-32b-instruct",
    # "qwen/qwen-2.5-coder-32b-instruct",
    # "qwen/qwen3-0.6b-04-28",
    # "qwen/qwen3-1.7b",
    # "qwen/qwen3-4b",
    # "qwen/qwen3-14b",
    # "qwen/qwen3-30b-a3b",
    "qwen/qwen3-32b",
    # "meta-llama/llama-guard-4-12b",
    # "perplexity/sonar-reasoning-pro",
    # "google/gemini-2.5-pro",
    # "meta-llama/llama-4-maverick-17b-128e-instruct",
    # "anthropic/claude-4-sonnet-20250522",
    # "intel/neural-chat-7b",
    # "openai/o3-pro",
]


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLOSE_THR = 0.80  # cosine threshold for “close”
TEMPERATURES = [0.5]  # can be overridden via CLI

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

MAPPINGS_OOI_PATH = "Terms_OoI.csv"
MAPPINGS_MATRIX_PATH = "Terms_Matrix.csv"
MAPPINGS_PROPERTY_PATH = "Terms_Property.csv"

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


client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

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
        f"### JSON-Schema\n{_SCHEMA_TEXT}\n"
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


def call_model(model: str, prompt: str, temperature: float) -> str:
    """Single chat completion – returns the *raw* text (may be unparsable)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
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


def call_llm_loose(model: str, prompt: str, orig_label: str, orig_comment: str, temperature: float) -> Dict[str, Any]:
    """
    Invoke *model*, capture its raw reply, coerce it into a dict that respects
    the schema, and **log every fix-up** (with context) to the preprocess logger.

    The function never raises. It returns either a clean dict (possibly empty)
    or {}, which upstream treats as a blank prediction.
    • retries up to 3 times on unparsable JSON
    • records 'retry_count' in the preprocess log
    """
    attempts, data, fixes = 0, {}, {}
    while attempts < 3:
        attempts += 1
        raw = call_model(model, prompt, temperature)
        fixes = {  # ← rebuild every round
            "model": model,
            "variable": orig_label,
            "retry_count": attempts,
            "non_json_prefix": False,
            "non_json_suffix": False,
            "unparsable_json": False,
            "label_overwritten": False,
            "comment_overwritten": False,
            "coerced_property_dict": False,
            "missing_keys": [],
            "extra_keys": [],
        }

        # ---------- try to isolate JSON ---------------------------------
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            fixes["unparsable_json"] = True
            fixes["raw_llm_output"] = raw  # keep for forensics
            _preproc_logger.info(json.dumps(fixes, ensure_ascii=False))
            if attempts < 3:
                continue  #  ↺  try again
            return {}  #  ✗  give up after 3 tries

        try:
            data = json.loads(m.group(0))
            break  # ✓ parsed, exit loop
        except json.JSONDecodeError:
            fixes["unparsable_json"] = True
            fixes["raw_llm_output"] = raw
            _preproc_logger.info(json.dumps(fixes, ensure_ascii=False))
            if attempts < 3:
                continue
            return {}

    fixes["retry_count"] = attempts  # final value (1-3)
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

    # ---- preserve original label/comment for logging ------------------
    pred_label = data.get("label")
    pred_comment = data.get("comment")

    # ---- force ground-truth label / comment ---------------------------
    if pred_label != orig_label:
        fixes["label_overwritten"] = True
        fixes["orig_label"] = orig_label
    if pred_comment != orig_comment:
        fixes["comment_overwritten"] = True
        fixes["orig_comment"] = (orig_comment or "")[:400]
    data["label"] = orig_label
    data["comment"] = orig_comment

    # ---- key coercions & sanitisation ---------------------------------
    data = coerce_for_eval(data, fixes=fixes)

    # ---- write the row (only enlarged when flags ≠ False) -------------
    _preproc_logger.info("%s", json.dumps(fixes, ensure_ascii=False))

    return data


# --------------------------------------------------------------------------- #
# 4 ▪ Similarity helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load & cache a Sentence‑Transformer model (memoised)."""
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def load_crossencoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> CrossEncoder:
    """Load & cache a Sentence‑Transformer model (memoised)."""
    # model.model.config.pad_token_id = model.tokenizer.pad_token_id
    # return CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())
    return CrossEncoder(model_name)

def _cosine(a: str, b: str, model_name: str) -> float:
    """Cosine similarity on sentence embeddings (helper)."""
    embedder = load_embedder(model_name)
    emb1 = embedder.encode(a, convert_to_tensor=True)
    emb2 = embedder.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def sim_string(a: str, b: str, close: bool, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Return similarity between two *strings*.

    • *exact* mode → 1.0 if lower‑cased strings match exactly, else 0.0
    • *close* mode → cosine similarity (embeddings)
    """
    if not a or not b:
        return 0.0
    norm_a, norm_b = a.lower().strip(), b.lower().strip()
    if norm_a == norm_b:
        return 1.0
    return _cosine(norm_a, norm_b, model_name) if close else 0.0


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


def sim_constraint(
    a: Dict[str, str],
    b: Dict[str, str],
    close: bool,
) -> float:
    """Similarity between two constraint dicts (label + on)."""
    lbl_sim = sim_string(a.get("label", ""), b.get("label", ""), close)

    # ‡ NEW: strip any "<OntoKey>: " prefix before comparison
    on_a = canonical_on(a.get("on", ""))
    on_b = canonical_on(b.get("on", ""))
    on_sim = sim_string(on_a, on_b, close)

    return (lbl_sim + on_sim) / 2


_ON_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+)$")


def canonical_on(text: str) -> str:
    """
    Remove a leading '<OntoKey>:' prefix from *text* when the prefix
    matches one of the recognised ONTO_KEYS (case-sensitive).

        "hasObjectOfInterest: 3-star hotel"  →  "3-star hotel"
        "hasMatrix: soil"                    →  "soil"
        "random prefix: foo"                 →  unchanged
    """
    if not text:
        return ""
    m = _ON_PREFIX_RE.match(text)
    if m and m.group(1) in ONTO_KEYS:
        return m.group(2).strip()
    return text.strip()


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
    gt_list: List[Dict[str, str]],
    pred_list: List[Dict[str, str]],
    close: bool,
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[float, float, float, float]:
    """
    Fine-grained TP / FP / FN / TN for *hasConstraint*.

    • Every ground-truth field (label / on) carries equal weight
          unit = 1 / (2 · len(gt_list))
    • Order never matters – constraints are matched with a greedy
      best-score algorithm.
    • Extra, unmatched predictions add **pure FP** (both fields).
    • Only when TP + FP + FN < 1 (numerical drift) we pad FP so the sum
      reaches 1.  We **never shrink** FP.
    """
    # ── degenerate cases -------------------------------------------------
    if not gt_list and not pred_list:
        return 0.0, 0.0, 0.0, 1.0  # perfect TN
    if not gt_list:  # no GT  →  everything is FP
        return 0.0, 1.0, 0.0, 0.0

    n_gt, n_pred = len(gt_list), len(pred_list)
    unit = 1.0 / (2 * n_gt)
    thr = CLOSE_THR if close else 1.0

    # ---------- build similarity matrix for pairing --------------------
    S = np.zeros((n_gt, n_pred))
    for i, j in product(range(n_gt), range(n_pred)):
        S[i, j] = sim_constraint(gt_list[i], pred_list[j], close)

    tp = fp = fn = 0.0
    gt_used: set[int] = set()
    pred_used: set[int] = set()

    # ---------- greedy best-score matching -----------------------------
    while S.size:
        i, j = divmod(int(np.argmax(S)), S.shape[1])
        if S[i, j] < 0:  # all remaining scores are –1
            break

        gt_used.add(i)
        pred_used.add(j)

        # label
        if sim_string(gt_list[i].get("label", ""), pred_list[j].get("label", ""), close, model_name) >= thr:
            tp += unit
        else:
            fp += unit

        # on
        if (
            sim_string(
                canonical_on(gt_list[i].get("on", "")), canonical_on(pred_list[j].get("on", "")), close, model_name
            )
            >= thr
        ):
            tp += unit
        else:
            fp += unit

        # invalidate matched row & column
        S[i, :] = -1.0
        S[:, j] = -1.0

    # ---------- unmatched GT → FN  -------------------------------------
    fn += (n_gt - len(gt_used)) * 2 * unit

    # ---------- unmatched predictions → FP -----------------------------
    fp += (n_pred - len(pred_used)) * 2 * unit

    # ---------- numeric guard (only pad *up* to 1) ---------------------
    total = tp + fp + fn
    # 1) typical floating-point drift  →  pad *up* to 1
    if 1.0 - total > 1e-6:
        fp += 1.0 - total
    # 2) over-prediction (total > 1)   →  **scale *down***, preserving ratios
    elif total - 1.0 > 1e-6:
        tp /= total
        fp /= total
        fn /= total

    return tp, fp, fn, 0.0  # TN only in trivial cases


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
    """Flatten a decomposed‑variable JSON into atomic strings.

    *mode* ∈ {"both", "concept", "text"} determines which parts to include.
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
            out.add(c.get("label", ""))
            out.add(canonical_on(c.get("on", "")))
    return {s for s in out if s}


# --------------------------------------------------------------------------- #
# 8 ▪ Prompt-deduplication helpers
# --------------------------------------------------------------------------- #
_PRINTED_PROMPTS: set[tuple[int, str]] = set()  # (shot_mode, variable label)
_PROMPT_LOCK = Lock()

def format_queries(query, instruction=None):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"

def get_wikidata_entity(term, naive_approach=True, context="", model_name="all-MiniLM-L6-v2"):
    """Returns the associated wikidata URI (in format http://www.wikidata.org/entity/Q??) 
    to the given term if there is a match, if not, the same term is returned.
    """
    output_entity = term
    encoded_term = urllib.parse.quote_plus(term)
    headers = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0",
}
    output = session.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={encoded_term}&language=en&format=json", headers=headers)
    if output.status_code == 200:
        output_search = output.json()["search"]
        if len(output_search) > 0:
            # Naive approach: Use the first search entry
            if naive_approach:
                output_entity = "http://www.wikidata.org/entity/"+output_search[0]["id"]
            else:
                model = load_crossencoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
                task = "Given a web search query, retrieve relevant passages that answer the query"
                queries = [f'Definition of "{term}" in context: "{context}"'] * len(output_search)
                documents = [f"label:\"{search_entry['label']}\", description: \"{search_entry['description'] if 'description' in search_entry else ""}\"" for search_entry in output_search]
                pairs = [
                    [format_queries(query, task), format_document(doc)]
                    for query, doc in zip(queries, documents)
                ]
                scores = model.predict(pairs)
                most_similar = scores.argmax().item()
                logging.info(f"Query: {queries[0]}")
                for i,score in enumerate(scores):
                    logging.info(f"{score:.2f}\t{documents[i]}")
                if scores[most_similar]>0.5:
                    output_entity = "http://www.wikidata.org/entity/"+output_search[most_similar]["id"]
    else:
        logging.warning("Error while calling the Wikidata API")
    return output_entity


def link2wikidata(input_dict: Dict[str, Any], naive_approach=True) -> Dict[str, Any]:
    """Given a prediction dictionary with terms, looks for their associated Wikidata URIs,
    and returns a dictionary with their links if a match is found, and with the same terms if not.
    """ 
    input_dict_copy = input_dict.copy()
    for key in ONTO_KEYS:
        val = input_dict.get(key, [] if key == "hasConstraint" else "")
        if key == "hasConstraint":
            if len(val)>0:
                input_dict_copy[key] = [{
                    "label": get_wikidata_entity(constraint["label"], naive_approach=naive_approach, context=f"{input_dict["label"]}"), 
                    "on": get_wikidata_entity(constraint["on"], naive_approach=naive_approach, context=f"{input_dict["label"]}") 
                    } for constraint in input_dict[key]
                ]
        else:
            if isinstance(val, dict) and "AsymmetricSystem" in val:
                asym_keys = ("AsymmetricSystem", "hasSource", "hasTarget")
                input_dict_copy[key] = {
                    asym_key: get_wikidata_entity(val[asym_key], naive_approach=naive_approach, context=f"{input_dict["label"]}") 
                    for asym_key in asym_keys 
                }
            elif isinstance(val, dict) and "SymmetricSystem" in val:
                input_dict_copy[key] = {
                    "SymmetricSystem": get_wikidata_entity(val["SymmetricSystem"], naive_approach=naive_approach, context=f"{input_dict["label"]}"), 
                    "hasPart": [
                        get_wikidata_entity(part, naive_approach=naive_approach, context=f"{input_dict["label"]}") 
                        for part in val["hasPart"]
                    ]
                }
            else:
                if not val=="":
                    input_dict_copy[key] = get_wikidata_entity(val, naive_approach=naive_approach, context=f"{input_dict["label"]}")
    return input_dict_copy


def load_wikidata_mappings() -> Dict[str, Any]:
    ooi_terms = load_dataset("csv", data_files=MAPPINGS_OOI_PATH, sep=";")["train"]
    matrix_terms = load_dataset("csv", data_files=MAPPINGS_MATRIX_PATH, sep=";")["train"]
    property_terms = load_dataset("csv",data_files=MAPPINGS_PROPERTY_PATH,sep = ";")["train"]
    ooi_terms_filtered = ooi_terms.select(range(22))
    matrix_terms_filtered = matrix_terms.select(range(13))
    ooi_mappings = {concept:wikidata_uri.split("/")[-1] if wikidata_uri else None  for concept,wikidata_uri in zip(ooi_terms_filtered["Concept"],ooi_terms_filtered["Wikidata"])}
    matrix_mappings = {concept:wikidata_uri.split("/")[-1] if wikidata_uri else None  for concept,wikidata_uri in zip(matrix_terms_filtered["Concept"],matrix_terms_filtered["Wikidata"])}
    property_mappings = {concept:wikidata_uri.split("/")[-1] if wikidata_uri else None  for concept,wikidata_uri in zip(property_terms["Concept"],property_terms["Wikidata"])}
    mappings = ooi_mappings | matrix_mappings | property_mappings
    return mappings


def get_wikidata_entity_from_mappings(term, mappings):
    return "http://www.wikidata.org/entity/"+mappings.get(term) if mappings.get(term) else term


def linkGT2wikidata(input_dict: Dict[str, Any], mappings: Dict[str, Any]) -> Dict[str, Any]:
    """This function expects a ground truth decomposition, and a dict with mappings 
    (term->{wikidata_entity_id (format: Q???),None}. The function will convert terms to 
    wikidata URIs (in format http://www.wikidata.org/entity/Q???) using the mappings. 
    Terms in decomposition must be present in the mappings dict, if the term is not in 
    the mappings or there is not a link, the same term will be used.
    """
    input_dict_copy = input_dict.copy()
    for key in ONTO_KEYS:
        val = input_dict.get(key, [] if key == "hasConstraint" else "")
        if key == "hasConstraint":
            if len(val)>0:
                input_dict_copy[key] = [
                    {
                        "label": get_wikidata_entity_from_mappings(constraint["label"], mappings), 
                        "on": get_wikidata_entity_from_mappings(constraint["on"], mappings) 
                    } for constraint in input_dict[key]]
        else:
            if isinstance(val, dict) and "AsymmetricSystem" in val:
                asym_keys = ("AsymmetricSystem", "hasSource", "hasTarget")
                input_dict_copy[key] = {asym_key: get_wikidata_entity_from_mappings(val[asym_key], mappings) for asym_key in asym_keys }
            elif isinstance(val, dict) and "SymmetricSystem" in val:
                input_dict_copy[key] = {
                    "SymmetricSystem": get_wikidata_entity_from_mappings(val["SymmetricSystem"], mappings), 
                    "hasPart": [get_wikidata_entity_from_mappings(part,mappings) for part in val["hasPart"]]
                }
            else:
                if not val=="":
                    input_dict_copy[key] = get_wikidata_entity_from_mappings(val,mappings)
    return input_dict_copy

# --------------------------------------------------------------------------- #
# 9 ▪ Evaluation worker  (returns {"_rows": [...]})
# --------------------------------------------------------------------------- #
def _run_one(
    model: str,
    gt: Dict[str, Any],
    prompt: str,
    shot: int,
    temperature: float,
    mappings: Dict[str, Any],
    naive_approach: bool
) -> Dict[str, Any]:
    # pred = call_llm_loose(model, prompt, gt["label"], gt["comment"], temperature=temperature)

    try:
        pred = call_llm_loose(
            model, prompt, orig_label=gt["label"], orig_comment=gt["comment"], temperature=temperature
        )

        # Link prediction and ground truth entities to wikidata
        pred_with_links = link2wikidata(pred, naive_approach=naive_approach)
        gt_with_links = linkGT2wikidata(gt, mappings)

        # ---------- human-readable logs -----------------------------------
        logging.info("MODEL | %-35s | shot=%d | T=%.2f | %s", model, shot, temperature, gt["label"])
        logging.info("GROUND-TRUTH JSON:\n%s", json.dumps(gt, indent=2, ensure_ascii=False))
        logging.info("PREDICTED    JSON:\n%s", json.dumps(pred, indent=2, ensure_ascii=False))
        logging.info("GROUND-TRUTH JSON WITH WIKIDATA LINKS:\n%s", json.dumps(gt_with_links, indent=2, ensure_ascii=False))
        logging.info("PREDICTED    JSON WITH WIKIDATA LINKS:\n%s", json.dumps(pred_with_links, indent=2, ensure_ascii=False))

        # Evaluate with links (This may be an option to include as an argument for the experiment_design.py program)
        gt = gt_with_links
        pred = pred_with_links

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

            # example: {'hasStatisticalModifier_TP': 0, 'hasStatisticalModifier_FP': 0, ...}
            per_key_unwrapped = {
                key + "_" + metric: per_key[key][i]
                for key in per_key
                for i, metric in enumerate(["TP", "FP", "FN", "TN"])
            }

            rows.append(
                {
                    "Variable": gt["label"],
                    "Model": model,
                    "Shot": shot,
                    "Temp": temperature,
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
                    **per_key_unwrapped,
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
    # debug_chars: int = 500,
    temps: List[float] | None = None,
    workers: int = 8,
    naive_approach: bool = True
) -> List[Dict[str, Any]]:

    temps = temps or TEMPERATURES
    models = models or MODEL_NAMES
    examples = load_examples(shot_mode)
    tasks: list[tuple] = []

    wikidata_mappings = load_wikidata_mappings()
    logging.info("WIKIDATA_MAPPINGS:\n%s", json.dumps(wikidata_mappings, indent=2, ensure_ascii=False))
    logging.info("NAIVE APPROACH:%s", str(naive_approach))

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

        for model in models:
            for temp in temps:
                tasks.append((model, gt, prompt, shot_mode, temp, wikidata_mappings, naive_approach))

    rows: list[dict] = []

    # ---------------- parallel execution ---------------------------------
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, *task) for task in tasks]

        for f in as_completed(futs):
            res = f.result()
            if res and "_rows" in res:
                rows.extend(res["_rows"])

    return rows


# --------------------------------------------------------------------------- #
# 11 ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="I-ADOPT LLM benchmark – shot × temperature")
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
    parser.add_argument("--temps", type=float, nargs="+", help="Override the default temperature grid")
    parser.add_argument('--naive_approach', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    temps = args.temps or TEMPERATURES
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
            naive_approach=args.naive_approach
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
    for (model, shot, temp), grp in df.groupby(["Model", "Shot", "Temp"]):
        # example: {'hasStatisticalModifier_F_exact': 0, 'hasStatisticalModifier_F_close': 0, ...}
        per_key_metrics = {}
        for key in ONTO_KEYS:
            sub = grp.loc[grp["Metric"] == "exact", key + "_" + "TP"]
            tp_tot_exact = sub.sum() if not sub.empty else float("nan")
            sub = grp.loc[grp["Metric"] == "exact", key + "_" + "FP"]
            fp_tot_exact = sub.sum() if not sub.empty else float("nan")
            sub = grp.loc[grp["Metric"] == "exact", key + "_" + "FN"]
            fn_tot_exact = sub.sum() if not sub.empty else float("nan")
            prec_exact, rec_exact, f1_exact = prf(tp_tot_exact, fp_tot_exact, fn_tot_exact)

            sub = grp.loc[grp["Metric"] == "close", key + "_" + "TP"]
            tp_tot_close = sub.sum() if not sub.empty else float("nan")
            sub = grp.loc[grp["Metric"] == "close", key + "_" + "FP"]
            fp_tot_close = sub.sum() if not sub.empty else float("nan")
            sub = grp.loc[grp["Metric"] == "close", key + "_" + "FN"]
            fn_tot_close = sub.sum() if not sub.empty else float("nan")
            prec_close, rec_close, f1_close = prf(tp_tot_close, fp_tot_close, fn_tot_close)

            per_key_metrics[key + "_" + "F_exact"] = f1_exact
            per_key_metrics[key + "_" + "F_close"] = f1_close
            per_key_metrics[key + "_" + "P_exact"] = prec_exact
            per_key_metrics[key + "_" + "P_close"] = prec_close
            per_key_metrics[key + "_" + "R_exact"] = rec_exact
            per_key_metrics[key + "_" + "R_close"] = rec_close

        summary_rows.append(
            {
                "Model": model,
                "Shot": shot,
                "temp": temp,
                "F_exact": pick(grp, "F", "exact"),
                "F_close": pick(grp, "F", "close"),
                "P_exact": pick(grp, "P", "exact"),
                "P_close": pick(grp, "P", "close"),
                "R_exact": pick(grp, "R", "exact"),
                "R_close": pick(grp, "R", "close"),
                "J_both": pick(grp, "J_both", "exact"),  # the J's are identical on both rows
                "J_concept": pick(grp, "J_concept", "exact"),
                "J_text": pick(grp, "J_text", "exact"),
                **per_key_metrics,
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
    best_pairs = (
        df_exact.groupby(["Shot", "Temp", "Model"], as_index=False)["F"]
        .mean()
        .rename(columns={"F": "F_exact"})
        .sort_values("F_exact", ascending=False)
    )
    best_pairs["Prompt"] = best_pairs["Shot"].astype(str) + "-shot"
    best_pairs = best_pairs[["Prompt", "Temp", "Model", "F_exact"]]

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
