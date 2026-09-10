#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
# I-ADOPT Benchmark – PATTERN-GUIDED Decomposition + Wikidata Linking
# --------------------------------------------------------------------------- #
"""
patternGuided.py
================

A pattern-guided benchmark workflow, comparable to ``phaseOneThreeMerged.py``.

What this workflow does (multi-component, decision-tree driven)
---------------------------------------------------------------
For each of the 102 benchmark variables it:

1. Loads the I-ADOPT decision-tree forest (pre-parsed JSON in
   ``Decisition_trees_json/``).
2. Traverses, per I-ADOPT *component*, to select a Variable Design Pattern (VDP):
     * Property : DT0 -> DT1 -> DT2 -> DT3/DT4 (sub-tree jumps followed)
     * Object of Interest : DT5
     * Matrix : DT7   (only when a medium/matrix is present)
   For each tree we make ONE batched LLM call (reasoning OFF, low temperature)
   that answers every question node at once; Python then walks the graph edges
   deterministically to the selected pattern leaf. Constraints are NOT traversed
   here (DT8/DT9/DT10) – the decomposition LLM fills ``hasConstraint`` freely,
   exactly like the baseline.
3. Loads each selected VDP and builds a single pattern-guided decomposition
   prompt that embeds the FULL pattern YAML inline (no GitHub links).
4. Calls the LLM to produce the I-ADOPT JSON (same structure as the baseline).
5. Runs the existing JSON-schema validation.
6. Runs rule-based pattern validation + gold-signature pattern TP/FP/FN/TN.
7. Links extracted entities to Wikidata (reused unchanged from the baseline).
8. Scores against gold using the baseline ``compute_confusion_for_pair``.
9. Writes results + scores to Excel (baseline-comparable + extra pattern info).

How it differs from ``phaseOneThreeMerged.py``
----------------------------------------------
The baseline uses one broad LLM prompt. This workflow first traverses the
decision-tree forest to choose one pattern per component, then prompts the LLM
with those patterns as modelling guidance. Everything downstream (dataset
loading, LLM client, JSON schema, JSON parsing, ``compute_confusion_for_pair``
scoring, Wikidata linking, Excel summary) is VENDORED verbatim from the baseline
into this file (this script no longer imports ``phaseOneThreeMerged``) so the
two runs use identical inputs/scoring/outputs and stay directly comparable.

Decision-tree source
---------------------
The upstream repo deleted ``DT0.yaml``/``DT1.yaml`` and now ships:
  * ``Decisition_trees/DT*.mmd``        – Mermaid flowcharts (human view)
  * ``Decisition_trees_json/DT*.json``  – pre-parsed graphs "for traversing"
We traverse the JSON. The misspelled folder name ``Decisition_trees`` is kept
exactly as upstream.

How to clone the pattern repository
-----------------------------------
::

    cd /Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service
    mkdir -p external
    git clone https://github.com/mabablue/I-ADOPT-patterns-playground.git \\
        external/I-ADOPT-patterns-playground

Override the path with ``IADOPT_PATTERNS_REPO`` env var or ``--patterns-repo``.

How to run the benchmark
------------------------
::

    python benchmarking_example/patternGuided.py \\
        --patterns-repo external/I-ADOPT-patterns-playground

    # cheap sanity check, no LLM / network
    python benchmarking_example/patternGuided.py --self-test

    # show which patterns are selected per variable (no decomposition)
    python benchmarking_example/patternGuided.py --dry-run-pattern-selection --limit 5

Where the Excel output is written
---------------------------------
``benchmarking_outputs/patternGuided_<YYYYMMDD_HHMMSS>.xlsx`` by default; the
timestamp is always appended even when ``--output`` is given. A matching detail
log is written under ``benchmarking_example/benchmarking_logs/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer, util

# Optional cached HTTP session for Wikidata (same as the original workflow).
try:
    import requests_cache

    _CACHE_SESSION = requests_cache.CachedSession("wikidata_cache", backend="sqllite", expire_after=None)
    _REQUESTS = _CACHE_SESSION
except Exception:
    _REQUESTS = requests

_THIS_DIR = pathlib.Path(__file__).resolve().parent

# =========================================================================== #
# VENDORED CORE (previously imported from phaseOneThreeMerged.py)
# --------------------------------------------------------------------------- #
# This script is now self-contained: the dataset loading, LLM client, JSON
# schema, JSON parsing/repair, similarity + confusion scoring, Wikidata linking
# and Excel-summary logic below are copied verbatim from the original baseline
# so that the pattern-guided run uses the SAME inputs, scoring and outputs and
# stays directly comparable. Do not change the scoring behaviour here.
# =========================================================================== #
load_dotenv()

DEFAULT_DATA_DIR = pathlib.Path(
    "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/benchmarking_example/data/Json_preferred/test_set"
)

SCHEMA_PATH = _THIS_DIR / "data" / "Json_schema.json"

ONE_SHOT_DIR = _THIS_DIR / "data" / "Json_preferred" / "one_shot"
THREE_SHOT_DIR = _THIS_DIR / "data" / "Json_preferred" / "three_shot"
FIVE_SHOT_DIR = _THIS_DIR / "data" / "Json_preferred" / "five_shot"

OUTBOOK_DIR = pathlib.Path("benchmarking_outputs")
OUTBOOK_DIR.mkdir(exist_ok=True)

MODEL_NAMES = [
    "qwen/qwen3.6-flash",
    # "qwen/qwen3-32b",
    # "qwen/qwen3.5-397b-a17b",
]

TEMPERATURES = [0.5]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLOSE_THR = 0.80
CROSS_ENCODER_MODEL = None

ONTO_KEYS = [
    "hasStatisticalModifier",
    "hasProperty",
    "hasObjectOfInterest",
    "hasMatrix",
    "hasContextObject",
    "hasConstraint",
]

# OpenAI client (OpenRouter) – same configuration as the baseline.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

# JSON schema text + prompt headers (used to build prompts and validate output).
_SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8").strip()
_EXAMPLE_HDR = "\n\n### Examples (valid against the same schema)\n"
_USER_HDR = "\n\n### Variable's definition to decompose\n"
_EXPECTED_HDR = "\n\n### Expected output\n*(only the JSON object)*"

_RERANK_LOCK = Lock()

# JSON extraction regexes (used to parse loose LLM output).
_JSON_FENCE_RE = re.compile(r"```(?:json)?", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# Project / repo paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = _THIS_DIR.parent
DEFAULT_PATTERNS_REPO = PROJECT_ROOT / "external" / "I-ADOPT-patterns-playground"
DECISION_TREE_DIR_NAME = "Decisition_trees"  # misspelled exactly as upstream
DECISION_TREE_JSON_DIR_NAME = "Decisition_trees_json"  # pre-parsed graphs
PATTERN_DIR_NAME = "pattern"
SHACL_DIR_NAME = "SHACL-definitions"
CONSTRAINTS_CSV_NAME = "Constraints_patterns.csv"

# Per-component entry trees. Property may chain through sub-tree jumps.
PROPERTY_START_TREE = "DT0"
OOI_START_TREE = "DT5"
MATRIX_START_TREE = "DT7"

MAX_TREE_HOPS = 12  # guard against runaway sub-tree jumps

# --------------------------------------------------------------------------- #
# Logging – this workflow gets its own logfile.
# --------------------------------------------------------------------------- #
LOG_DIR = _THIS_DIR / "benchmarking_logs"
LOG_DIR.mkdir(exist_ok=True)
PG_LOG_FILE = LOG_DIR / f"patternGuided_{datetime.now():%Y%m%d_%H%M%S}.log"

logger = logging.getLogger("patternGuided")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | patternGuided | %(message)s")
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _fh = logging.FileHandler(PG_LOG_FILE, mode="w", encoding="utf-8")
    _fh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.addHandler(_fh)
logger.info("Pattern-guided logging to %s", PG_LOG_FILE.resolve())

# Atomic multi-line logging across worker threads.
_log_lock = Lock()


# --------------------------------------------------------------------------- #
# Vendored: example loading + prompt example formatting
# --------------------------------------------------------------------------- #
def strip_all_uri_fields(obj: Any) -> Any:
    """Remove ANY dict key containing 'URI' recursively (for in-prompt examples)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if "URI" in k:
                continue
            if k.startswith("__"):
                continue
            out[k] = strip_all_uri_fields(v)
        return out
    if isinstance(obj, list):
        return [strip_all_uri_fields(x) for x in obj]
    return obj


def format_example_block(ex: Dict[str, Any], idx: int) -> str:
    """Render one gold example for the prompt (definition + URI-stripped JSON)."""
    definition = ex.get("definition") or ex.get("comment") or ""
    ex_no_uris = strip_all_uri_fields(ex)
    return (
        f"\n\n#### Example {idx}\n"
        f"Variable's definition to decompose: {definition}\n\n"
        f"Expected output:\n{json.dumps(ex_no_uris, indent=2, ensure_ascii=False)}"
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
    return [json.load(open(p, "r", encoding="utf-8")) for p in paths[:n]]


# --------------------------------------------------------------------------- #
# Vendored: LLM invocation (robust) + coercion
# --------------------------------------------------------------------------- #
def call_model(model: str, prompt: str, temperature: float) -> str:
    """Call the chat model with retries; returns the raw text (or '')."""
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            text = resp.choices[0].message.content or ""
            stripped = text.strip()
            if stripped.startswith("<!DOCTYPE html") or stripped.startswith("<html"):
                logger.warning("%s: HTML error response on attempt %d", model, attempt)
                continue
            if not stripped:
                logger.warning("%s: empty response on attempt %d", model, attempt)
                continue
            return text
        except APIStatusError as e:
            logger.warning(
                "%s: APIStatusError attempt %d – %s – %s", model, attempt, e.status_code, getattr(e, "body", "")
            )
        except (OpenAIError, httpx.HTTPError) as e:
            logger.warning("%s: transport error attempt %d – %r", model, attempt, e)
        except Exception as e:
            logger.warning("%s: unexpected error attempt %d – %r", model, attempt, e)
    logger.error("%s: failed after 3 attempts", model)
    return ""


def coerce_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise an LLM prediction so every ONTO_KEY exists with a sane type."""
    pred = dict(pred or {})
    for k in ONTO_KEYS:
        if k not in pred or pred[k] is None:
            pred[k] = [] if k == "hasConstraint" else ""
        elif k == "hasConstraint" and not isinstance(pred[k], list):
            pred[k] = []
    if isinstance(pred.get("hasProperty"), dict):
        pred["hasProperty"] = pred["hasProperty"].get("label", "") or ""
    return pred


# --------------------------------------------------------------------------- #
# Vendored: similarity & confusion helpers (scoring – DO NOT change behaviour)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _cosine(a: str, b: str, model_name: str) -> float:
    emb = load_embedder(model_name)
    e1 = emb.encode(a, convert_to_tensor=True)
    e2 = emb.encode(b, convert_to_tensor=True)
    return util.cos_sim(e1, e2).item()


def sim_string(a: str, b: str, close: bool, model_name: str = EMBED_MODEL_NAME) -> float:
    if not a or not b:
        return 0.0
    na, nb = a.lower().strip(), b.lower().strip()
    if na == nb:
        return 1.0
    return _cosine(na, nb, model_name) if close else 0.0


def _sym_parts(obj: Any) -> Tuple[str, Set[str]]:
    if isinstance(obj, dict) and "SymmetricSystem" in obj and "hasPart" in obj:
        return obj.get("SymmetricSystem", ""), set(obj.get("hasPart", []))
    return "", set()


def sim_sym(a: Any, b: Any, close: bool) -> float:
    la, pa = _sym_parts(a)
    lb, pb = _sym_parts(b)
    if not (la or lb):
        return 0.0
    label_sim = sim_string(la, lb, close)
    part_sim = len(pa & pb) / len(pa | pb) if (pa or pb) else 1.0
    return (label_sim + part_sim) / 2


def sim_asym(a: Dict[str, Any], b: Dict[str, Any], close: bool) -> float:
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return 0.0
    src_a = a.get("hasSource") or a.get("hasNumerator") or ""
    tgt_a = a.get("hasTarget") or a.get("hasDenominator") or ""
    src_b = b.get("hasSource") or b.get("hasNumerator") or ""
    tgt_b = b.get("hasTarget") or b.get("hasDenominator") or ""
    if not (src_a or tgt_a or src_b or tgt_b):
        return 0.0
    return (sim_string(src_a, src_b, close) + sim_string(tgt_a, tgt_b, close)) / 2


_ON_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+)$")


def canonical_on(text: str) -> str:
    if not text:
        return ""
    m = _ON_PREFIX_RE.match(text)
    if m and m.group(1) in ONTO_KEYS:
        return m.group(2).strip()
    return text.strip()


def normalize_constraint(c: Dict[str, str]) -> Dict[str, str]:
    if not isinstance(c, dict):
        return {"label": "", "on": ""}

    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        return re.sub(r"\s+", " ", s)

    return {"label": norm(c.get("label", "")), "on": norm(canonical_on(c.get("on", "")))}


def confusion(gt, pred, close: bool) -> Tuple[float, float, float, float]:
    if isinstance(gt, dict) and "AsymmetricSystem" in gt:
        score = sim_asym(gt, pred, close)
    elif isinstance(gt, dict) and "SymmetricSystem" in gt:
        score = sim_sym(gt, pred, close)
    else:
        score = sim_string(str(gt), str(pred), close)

    thr = CLOSE_THR if close else 1.0

    if gt:
        if pred and score >= thr:
            return 1.0, 0.0, 0.0, 0.0
        if pred:
            return 0.0, 1.0, 0.0, 0.0
        return 0.0, 0.0, 1.0, 0.0
    else:
        return (0.0, 0.0, 0.0, 1.0) if not pred else (0.0, 1.0, 0.0, 0.0)


def confusion_constraints(
    gt_list: List[Dict[str, str]],
    pred_list: List[Dict[str, str]],
    close: bool,
    model_name: str = EMBED_MODEL_NAME,
) -> Tuple[float, float, float, float]:
    if not gt_list and not pred_list:
        return 0.0, 0.0, 0.0, 1.0
    if not gt_list:
        return 0.0, 1.0, 0.0, 0.0

    gt_list = gt_list or []
    pred_list = pred_list or []
    n_gt, n_pred = len(gt_list), len(pred_list)
    unit = 1.0 / (2 * n_gt)
    thr = CLOSE_THR if close else 1.0

    gt_norm = [normalize_constraint(c) for c in gt_list]
    pred_norm = [normalize_constraint(c) for c in pred_list]

    S = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt_norm):
        for j, p in enumerate(pred_norm):
            lbl_sim = sim_string(g["label"], p["label"], close, model_name)
            on_sim = sim_string(g["on"], p["on"], close, model_name)
            S[i, j] = (lbl_sim + on_sim) / 2.0

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

        if sim_string(gt_norm[i]["label"], pred_norm[j]["label"], close, model_name) >= thr:
            tp += unit
        else:
            fp += unit

        if sim_string(gt_norm[i]["on"], pred_norm[j]["on"], close, model_name) >= thr:
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


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def atoms(rec: Dict[str, Any], mode: str) -> Set[str]:
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


def compute_confusion_for_pair(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    exact = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    close = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    per_key_exact = {}
    per_key_close = {}

    for key in ONTO_KEYS:
        gt_val = gt.get(key, [] if key == "hasConstraint" else "")
        pred_val = pred.get(key, [] if key == "hasConstraint" else "")

        if key == "hasConstraint":
            tp, fp, fn, tn = confusion_constraints(gt_val, pred_val, close=False)
        else:
            tp, fp, fn, tn = confusion(gt_val, pred_val, close=False)

        per_key_exact[key] = (tp, fp, fn, tn)
        exact["tp"] += tp
        exact["fp"] += fp
        exact["fn"] += fn
        exact["tn"] += tn

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
# Vendored: Phase-3 Wikidata linking + URI evaluation
# --------------------------------------------------------------------------- #
def _qid_from_uri_or_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"(Q\d+)", s)
    return m.group(1) if m else None


def canonicalize_uri_for_compare(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    q = _qid_from_uri_or_text(uri)
    if q:
        return f"https://www.wikidata.org/wiki/{q}"
    u = uri.strip().replace("http://", "https://")
    return u[:-1] if u.endswith("/") else u


def _to_wiki_url(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    q = _qid_from_uri_or_text(uri)
    return f"https://www.wikidata.org/wiki/{q}" if q else canonicalize_uri_for_compare(uri)


# --- Qwen3 reranker formatting (recommended templates) ---
QWEN3_RERANK_PREFIX = (
    "<|im_start|>system\n"
    " Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_RERANK_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def format_queries(query: str, task: str = DEFAULT_RERANK_TASK) -> str:
    return f"{QWEN3_RERANK_PREFIX}<Instruct>: {task}\n<Query>: {query}\n"


def format_document(doc: str) -> str:
    return f"<Document>: {doc}{QWEN3_RERANK_SUFFIX}"


def get_wikidata_entity(
    term: str,
    approach: str = "naive",
    context: str = "",
    model_name: str = EMBED_MODEL_NAME,
    threshold: float = 0.0,
) -> Optional[str]:
    if not term:
        return None

    encoded = urllib.parse.quote_plus(term)
    headers = {"User-Agent": "IADOPT-Linker/1.0 (+benchmark script)"}

    try:
        resp = _REQUESTS.get(
            f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={encoded}&language=en&format=json",
            headers=headers,
            timeout=20,
        )
        if resp.status_code != 200:
            logger.warning("Wikidata API HTTP %s for %r", resp.status_code, term)
            return None

        search = resp.json().get("search", [])
        if not search:
            return None

        if approach == "naive":
            return _to_wiki_url(search[0]["id"])

        if approach == "embedding":
            embedder = load_embedder(model_name)
            qv = embedder.encode(f'Definition of "{term}" in context: "{context}"')
            docs = [f'label: "{s.get("label","")}", description: "{s.get("description","")}"' for s in search]
            dv = embedder.encode(docs)
            sims = util.cos_sim(qv, dv).cpu().numpy().ravel()
            idx = int(sims.argmax())
            return _to_wiki_url(search[idx]["id"])

        if approach == "cross-encoder":
            reranker = CROSS_ENCODER_MODEL
            query = f'Definition of "{term}" in context: "{context}"'
            documents = [f'label: "{s.get("label","")}", description: "{s.get("description","")}"' for s in search]
            pairs = [[format_queries(query, DEFAULT_RERANK_TASK), format_document(doc)] for doc in documents]

            with _RERANK_LOCK:
                scores = reranker.predict(pairs, show_progress_bar=False)

            ranked = sorted(zip(search, scores), key=lambda x: float(x[1]), reverse=True)
            best_s, best_score = ranked[0]
            return _to_wiki_url(best_s["id"]) if float(best_score) >= float(threshold) else None

        return _to_wiki_url(search[0]["id"])

    except Exception as e:
        logger.warning("Wikidata API error for %r: %r", term, e)
        return None


def enrich_with_uris(
    pred: Dict[str, Any],
    approach: str = "naive",
    model_name: str = EMBED_MODEL_NAME,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    if approach == "none":
        return pred

    out = json.loads(json.dumps(pred))  # deep copy

    def add_uri_field(container: Dict[str, Any], key: str, label_value: Any):
        if isinstance(label_value, str) and label_value.strip():
            uri = get_wikidata_entity(
                label_value,
                approach=approach,
                context=pred.get("definition", ""),
                model_name=model_name,
                threshold=threshold,
            )
            if uri:
                container[f"{key}URI"] = _to_wiki_url(uri)

    for p in ["hasProperty", "hasMatrix", "hasObjectOfInterest", "hasContextObject"]:
        if p in out and isinstance(out[p], str):
            add_uri_field(out, p, out[p])

    for p in ["hasMatrix", "hasObjectOfInterest", "hasContextObject"]:
        val = out.get(p)
        if isinstance(val, dict):
            if "AsymmetricSystem" in val:
                for kk in ["AsymmetricSystem", "hasSource", "hasTarget"]:
                    if val.get(kk):
                        uri = get_wikidata_entity(
                            val[kk],
                            approach=approach,
                            context=pred.get("definition", ""),
                            model_name=model_name,
                            threshold=threshold,
                        )
                        if uri:
                            val[f"{kk}URI"] = _to_wiki_url(uri)

            if "SymmetricSystem" in val:
                if val.get("SymmetricSystem"):
                    uri = get_wikidata_entity(
                        val["SymmetricSystem"],
                        approach=approach,
                        context=pred.get("definition", ""),
                        model_name=model_name,
                        threshold=threshold,
                    )
                    if uri:
                        val["SymmetricSystemURI"] = _to_wiki_url(uri)

                parts = val.get("hasPart", [])
                if isinstance(parts, list) and parts:
                    part_uris = []
                    for part in parts:
                        if isinstance(part, str) and part.strip():
                            uri = get_wikidata_entity(
                                part,
                                approach=approach,
                                context=pred.get("definition", ""),
                                model_name=model_name,
                                threshold=threshold,
                            )
                            part_uris.append(_to_wiki_url(uri) if uri else None)
                        else:
                            part_uris.append(None)
                    if any(part_uris):
                        val["hasPartURIs"] = part_uris

    return out


def _iter_uri_assertions(gt: Dict[str, Any]) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for key in ["hasPropertyURI", "hasMatrixURI", "hasObjectOfInterestURI", "hasContextObjectURI"]:
        if gt.get(key):
            out.append((key, gt[key]))

    for root in ["hasMatrix", "hasObjectOfInterest", "hasContextObject"]:
        node = gt.get(root)
        if isinstance(node, dict):
            for k in ["AsymmetricSystemURI", "SymmetricSystemURI", "hasSourceURI", "hasTargetURI"]:
                if node.get(k):
                    out.append((f"{root}.{k}", node[k]))
            if isinstance(node.get("hasPartURIs"), list):
                out.append((f"{root}.hasPartURIs", node["hasPartURIs"]))
    return out


def _get_pred_uri_at_path(pred: Dict[str, Any], path: str) -> Any:
    cur: Any = pred
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def compare_uris(
    gt: Dict[str, Any], pred_enriched: Dict[str, Any]
) -> Tuple[int, int, float, float, int, Dict[str, bool]]:
    assertions = _iter_uri_assertions(gt)
    total = 0
    correct = 0
    per_field_ok: Dict[str, bool] = {}

    for path, expected in assertions:
        total += 1
        pred_val = _get_pred_uri_at_path(pred_enriched, path)

        ok = False
        if isinstance(expected, list):
            if isinstance(pred_val, list) and len(pred_val) == len(expected):
                ok = all(
                    canonicalize_uri_for_compare(p) == canonicalize_uri_for_compare(g)
                    for p, g in zip(pred_val, expected)
                )
        else:
            ok = canonicalize_uri_for_compare(pred_val) == canonicalize_uri_for_compare(expected)

        per_field_ok[path.replace(".", "_")] = bool(ok)
        correct += 1 if ok else 0

    acc = (correct / total) if total else 1.0
    predicted_non_null = sum(
        1 for path, _ in assertions if _get_pred_uri_at_path(pred_enriched, path) not in (None, "", [])
    )
    coverage = (predicted_non_null / total) if total else 1.0
    return total, correct, acc, coverage, predicted_non_null, per_field_ok


def load_gt_files_recursive(data_dir: pathlib.Path, max_vars: int) -> List[Tuple[pathlib.Path, Dict[str, Any]]]:
    paths = sorted(data_dir.rglob("*.json"))
    if max_vars:
        paths = paths[:max_vars]
    out = []
    for p in paths:
        obj = json.load(open(p, "r", encoding="utf-8"))
        out.append((p, obj))
    return out


def compute_summary_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-variable confusion totals into a summary table (verbatim)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in results:
        key = (
            r["model"],
            r["temperature"],
            r["prompt_version"],
            r["shot"],
            r["link_approach"],
            r["link_model_name"],
            r["link_threshold"],
        )
        groups.setdefault(key, []).append(r)

    rows: List[Dict[str, Any]] = []
    for key, rs in groups.items():
        model, temp, pv, shot, approach, link_model, thr = key

        exact = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        close = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        per_key = {
            k: {
                "exact": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
                "close": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
            }
            for k in ONTO_KEYS
        }
        j_both, j_concept, j_text = [], [], []
        uri_acc, uri_cov, uri_pred = [], [], []

        for r in rs:
            conf = r["confusion"]
            for tag, tot in (("exact", conf["exact_totals"]), ("close", conf["close_totals"])):
                target = exact if tag == "exact" else close
                target["tp"] += tot["tp"]
                target["fp"] += tot["fp"]
                target["fn"] += tot["fn"]
                target["tn"] += tot["tn"]
            for k in ONTO_KEYS:
                tp, fp, fn, tn = conf["per_key_exact"][k]
                per_key[k]["exact"]["tp"] += tp
                per_key[k]["exact"]["fp"] += fp
                per_key[k]["exact"]["fn"] += fn
                per_key[k]["exact"]["tn"] += tn
                tp2, fp2, fn2, tn2 = conf["per_key_close"][k]
                per_key[k]["close"]["tp"] += tp2
                per_key[k]["close"]["fp"] += fp2
                per_key[k]["close"]["fn"] += fn2
                per_key[k]["close"]["tn"] += tn2
            j_both.append(r.get("j_both", 0.0))
            j_concept.append(r.get("j_concept", 0.0))
            j_text.append(r.get("j_text", 0.0))
            uri_acc.append(r.get("uris_acc", 0.0))
            uri_cov.append(r.get("uris_coverage", 0.0))
            uri_pred.append(r.get("uris_predicted", 0))

        p_e, r_e, f_e = prf(exact["tp"], exact["fp"], exact["fn"])
        p_c, r_c, f_c = prf(close["tp"], close["fp"], close["fn"])

        out: Dict[str, Any] = {
            "Model": model,
            "Temperature": temp,
            "PromptVersion": pv,
            "Shot": shot,
            "LinkApproach": approach,
            "LinkModelName": link_model,
            "LinkThreshold": thr,
            "P_exact": round(p_e, 3),
            "R_exact": round(r_e, 3),
            "F_exact": round(f_e, 3),
            "P_close": round(p_c, 3),
            "R_close": round(r_c, 3),
            "F_close": round(f_c, 3),
            "J_both_mean": round(float(np.mean(j_both)) if j_both else 0.0, 3),
            "J_concept_mean": round(float(np.mean(j_concept)) if j_concept else 0.0, 3),
            "J_text_mean": round(float(np.mean(j_text)) if j_text else 0.0, 3),
            "URI_acc_mean": round(float(np.mean(uri_acc)) if uri_acc else 0.0, 3),
            "URI_coverage_mean": round(float(np.mean(uri_cov)) if uri_cov else 0.0, 3),
            "URI_predicted_mean": round(float(np.mean(uri_pred)) if uri_pred else 0.0, 3),
        }
        for k in ONTO_KEYS:
            for tag in ("exact", "close"):
                tp = per_key[k][tag]["tp"]
                fp = per_key[k][tag]["fp"]
                fn = per_key[k][tag]["fn"]
                p, r_, f = prf(tp, fp, fn)
                out[f"{k}_P_{tag}"] = round(p, 3)
                out[f"{k}_R_{tag}"] = round(r_, 3)
                out[f"{k}_F_{tag}"] = round(f, 3)
        rows.append(out)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["F_exact", "URI_acc_mean"], ascending=[False, False])
    return df


# =========================================================================== #
# >>> PATTERN-GUIDED LOGIC BEGINS <<<
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 1. URL / id helpers
# --------------------------------------------------------------------------- #
_GH_BLOB_RE = re.compile(r"github\.com/[^/]+/[^/]+/(?:blob|edit|raw|tree)/[^/]+/(?P<path>.+)$")
_VDP_ID_RE = re.compile(r"(VDP\.?\d+)", re.IGNORECASE)
_DT_ID_RE = re.compile(r"(DT\d+)", re.IGNORECASE)
_HREF_RE = re.compile(r"href='([^']+)'")


def resolve_github_blob_url_to_local_path(url_or_path: str, repo_root: pathlib.Path) -> Optional[pathlib.Path]:
    """Map a GitHub blob/edit/raw/tree URL to a local path under ``repo_root``.

    Plain local paths are passed through (resolved relative to repo_root).
    Example: .../blob/main/pattern/VDP11.yaml -> <repo_root>/pattern/VDP11.yaml
    """
    if not url_or_path:
        return None
    s = str(url_or_path).strip()
    m = _GH_BLOB_RE.search(s)
    if m:
        return repo_root / m.group("path").strip().strip("/")
    p = pathlib.Path(s)
    return p if p.is_absolute() else repo_root / p


def extract_pattern_id(url_or_path: str) -> str:
    """Extract a normalised VDP id (e.g. ``VDP11``) from a URL/path/label."""
    if not url_or_path:
        return ""
    m = _VDP_ID_RE.search(str(url_or_path))
    if m:
        return m.group(1).upper().replace(".", "")
    return pathlib.Path(str(url_or_path)).stem


def extract_tree_id(url_or_path: str) -> str:
    """Extract a decision-tree id (e.g. ``DT4``) from a URL/path/label."""
    m = _DT_ID_RE.search(str(url_or_path or ""))
    return m.group(1).upper() if m else str(url_or_path or "")


def _clean_label(text: str) -> str:
    """Strip Mermaid HTML (``<br>`` etc.) and collapse whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


# Top-level keys we understand in a VDP file, with their value shape.
_VDP_TOP_KEYS = {
    "type",
    "id",
    "name",
    "comment",
    "design",
    "instances",
    "involved_components",
    "constrained_component",
    "associated_patterns",
    "associated_pattern",
    "associated_mapping",
    "example",
    "gh_issue",
    "rules",
    "decomposition",
    "pattern_rule",
}
_VDP_LIST_KEYS = {"instances", "involved_components", "associated_patterns", "associated_pattern"}
_TOP_KEY_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_ ()/-]*?):\s?(?P<val>.*)$")


def _tolerant_yaml_mapping(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Line-based fallback parser for slightly-malformed VDP YAML files.

    Several upstream patterns have unquoted multi-line scalar values that strict
    YAML rejects. This recovers a best-effort mapping: a line ``key: value``
    starts a field (only for known top-level VDP keys); subsequent indented or
    non-key lines are folded into the current scalar; ``- item`` lines build a
    list. It is intentionally conservative and never raises.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    out: Dict[str, Any] = {}
    cur_key: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()

        # list item for the current key
        if stripped.startswith("- ") or stripped == "-":
            if cur_key in _VDP_LIST_KEYS or isinstance(out.get(cur_key), list):
                out.setdefault(cur_key, [])
                if not isinstance(out[cur_key], list):
                    out[cur_key] = []
                item = stripped[1:].strip()
                if item:
                    out[cur_key].append(item)
            continue

        m = _TOP_KEY_RE.match(line) if not line[0].isspace() else None
        key = m.group("key").strip() if m else None
        if m and key in _VDP_TOP_KEYS:
            val = m.group("val").strip()
            cur_key = key
            if key in _VDP_LIST_KEYS:
                out[key] = []
            elif val in ("~", ""):
                out[key] = None if val == "~" else ""
            else:
                out[key] = val
        else:
            # continuation of the current scalar value
            if cur_key and isinstance(out.get(cur_key), str):
                out[cur_key] = (out[cur_key] + " " + stripped).strip()

    return out or None


# --------------------------------------------------------------------------- #
# 2. Pattern repository loader (JSON trees + VDP YAML + constraints CSV)
# --------------------------------------------------------------------------- #
class PatternRepository:
    """Loads decision-tree graphs (JSON), VDP patterns (YAML), constraints CSV.

    All loaders are defensive: a missing/invalid file is logged and yields
    ``None``/``{}``/``[]`` rather than crashing the benchmark.
    """

    def __init__(self, repo_root: pathlib.Path):
        self.repo_root = pathlib.Path(repo_root).resolve()
        self.tree_dir = self.repo_root / DECISION_TREE_DIR_NAME
        self.tree_json_dir = self.repo_root / DECISION_TREE_JSON_DIR_NAME
        self.pattern_dir = self.repo_root / PATTERN_DIR_NAME
        self.shacl_dir = self.repo_root / SHACL_DIR_NAME
        self._tree_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._pattern_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        if not self.repo_root.exists():
            logger.warning("Patterns repo root does not exist: %s", self.repo_root)
        if not self.tree_json_dir.exists():
            logger.warning("Decision-tree JSON dir does not exist: %s", self.tree_json_dir)

    # -- low level -------------------------------------------------------- #
    def _safe_load_yaml(self, path: pathlib.Path) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if data is None:
                logger.warning("YAML file is empty: %s", path)
                return None
            if not isinstance(data, dict):
                logger.warning("YAML not a mapping (%s): %s", type(data).__name__, path)
                return {"__raw__": data}
            return data
        except FileNotFoundError:
            logger.warning("YAML file not found: %s", path)
            return None
        except yaml.YAMLError as e:
            # Several upstream VDP files have unquoted multi-line scalar values
            # (e.g. a `comment:` that wraps across lines) which strict YAML
            # rejects. Fall back to a tolerant line-based parse so the pattern
            # still contributes its content instead of being dropped.
            recovered = _tolerant_yaml_mapping(path)
            if recovered:
                logger.warning("YAML parse error in %s: %r — recovered via tolerant parse", path, e)
                return recovered
            logger.warning("YAML parse error in %s: %r", path, e)
            return None
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Unexpected error loading YAML %s: %r", path, e)
            return None

    # -- decision trees (JSON graphs) ------------------------------------- #
    def load_decision_tree(self, tree_id_or_path: str) -> Optional[Dict[str, Any]]:
        """Load a decision tree as a normalised graph dict.

        Accepts a bare id (``DT4``), a ``DT4.json``/``DT4.mmd`` name, a local
        path, or a GitHub URL. Returns the graph produced by ``_normalise_tree``
        or ``None`` on failure.
        """
        tid = extract_tree_id(tree_id_or_path) or str(tree_id_or_path)
        if tid in self._tree_cache:
            return self._tree_cache[tid]
        path = self.tree_json_dir / f"{tid}.json"
        graph: Optional[Dict[str, Any]] = None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            graph = _normalise_tree(tid, raw)
        except FileNotFoundError:
            logger.warning("Decision-tree JSON not found: %s", path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Error loading decision-tree JSON %s: %r", path, e)
        self._tree_cache[tid] = graph
        return graph

    # -- VDP patterns ----------------------------------------------------- #
    def load_variable_design_pattern(self, pattern_url_or_path: str) -> Optional[Dict[str, Any]]:
        key = str(pattern_url_or_path)
        if key in self._pattern_cache:
            return self._pattern_cache[key]
        # Accept a bare id like "VDP17" as well as URLs/paths.
        if _VDP_ID_RE.fullmatch(key.strip()) or (key.upper().startswith("VDP") and "/" not in key):
            path: Optional[pathlib.Path] = self.pattern_dir / f"{extract_pattern_id(key)}.yaml"
        else:
            path = resolve_github_blob_url_to_local_path(pattern_url_or_path, self.repo_root)
        if path is None:
            logger.warning("Could not resolve VDP reference: %s", pattern_url_or_path)
            self._pattern_cache[key] = None
            return None
        data = self._safe_load_yaml(path)
        self._pattern_cache[key] = data
        return data

    def load_constraints_patterns_csv(self) -> List[Dict[str, str]]:
        path = self.tree_dir / CONSTRAINTS_CSV_NAME
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        except FileNotFoundError:
            logger.warning("Constraints CSV not found: %s", path)
            return []
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Error loading constraints CSV %s: %r", path, e)
            return []

    def shacl_files(self) -> List[pathlib.Path]:
        if not self.shacl_dir.exists():
            return []
        return sorted(self.shacl_dir.glob("*.ttl"))


# --------------------------------------------------------------------------- #
# 3. Decision-tree graph normalisation (Mermaid-parser JSON -> simple graph)
# --------------------------------------------------------------------------- #
def _classify_vertex(text: str) -> Tuple[str, Optional[str]]:
    """Classify a vertex by its label/href.

    Returns (kind, ref) where kind is 'pattern' (ref=VDPxx),
    'subtree' (ref=DTx), or 'question' (ref=None).
    """
    m = _HREF_RE.search(text or "")
    if m:
        url = m.group(1)
        if "/pattern/" in url:
            return "pattern", extract_pattern_id(url)
        if ".mmd" in url or "Decisition_trees" in url:
            return "subtree", extract_tree_id(url)
    return "question", None


def _normalise_tree(tree_id: str, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn the verbose Mermaid-parser JSON into a compact traversable graph.

    Output::

        {
          "id": "DT4",
          "title": "...",
          "root": "A",
          "nodes": {id: {"kind": "question|pattern|subtree",
                          "label": str, "ref": Optional[str]}},
          "edges": {src_id: [{"label": "true|false|Area|...", "to": id}]},
        }
    """
    parsed = (raw or {}).get("parsed") or {}
    gd = parsed.get("graph_data") or {}
    vertices = gd.get("vertices") or {}
    edges = gd.get("edges") or []
    if not vertices:
        return None

    nodes: Dict[str, Dict[str, Any]] = {}
    for vid, v in vertices.items():
        text = v.get("text", "") if isinstance(v, dict) else ""
        kind, ref = _classify_vertex(text)
        nodes[vid] = {"kind": kind, "label": _clean_label(text), "ref": ref}

    out_edges: Dict[str, List[Dict[str, Any]]] = {}
    indeg = {vid: 0 for vid in nodes}
    for e in edges:
        src, dst = e.get("start"), e.get("end")
        if src not in nodes or dst not in nodes:
            continue
        lbl = e.get("text")
        if isinstance(lbl, bool):
            lbl = "true" if lbl else "false"
        elif lbl is None:
            lbl = ""
        else:
            lbl = str(lbl).strip()
        out_edges.setdefault(src, []).append({"label": lbl, "to": dst})
        indeg[dst] = indeg.get(dst, 0) + 1

    # Root = the (typically unique) node with no incoming edges. Prefer "A".
    roots = [vid for vid, d in indeg.items() if d == 0]
    root = "A" if "A" in roots else (roots[0] if roots else next(iter(nodes)))

    fm = raw.get("frontmatter") or {}
    return {
        "id": tree_id,
        "title": fm.get("title", tree_id),
        "root": root,
        "nodes": nodes,
        "edges": out_edges,
    }


def collect_tree_questions(graph: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return the question nodes reachable in a tree as {id, question}."""
    out = []
    for vid, n in graph["nodes"].items():
        if n["kind"] == "question" and n.get("label"):
            out.append({"id": vid, "question": n["label"]})
    return out


# --------------------------------------------------------------------------- #
# 4. Batched decision-tree traversal (one LLM call per tree, reasoning OFF)
# --------------------------------------------------------------------------- #
def _call_model_no_reasoning(model: str, prompt: str, temperature: float) -> str:
    """Call the LLM with reasoning/thinking disabled for speed.

    Uses the OpenRouter client. ``reasoning={"enabled": False}`` is accepted by
    OpenRouter for models that support thinking; harmless for those that don't.
    Falls back to ``call_model`` if the client rejects the extra args.
    """
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
                extra_body={"reasoning": {"enabled": False}},
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
        except TypeError:
            # Older client without extra_body support.
            return call_model(model, prompt, temperature)
        except Exception as e:
            logger.warning("%s: tree-call error attempt %d – %r", model, attempt, e)
    return ""


def _parse_answer_map(raw: str) -> Optional[Dict[str, str]]:
    """Parse a JSON object mapping node-id -> answer label from a raw response."""
    if not raw or not raw.strip():
        return None
    cleaned = _JSON_FENCE_RE.sub("", raw).strip()
    m = _JSON_BLOCK_RE.search(cleaned)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # Accept either {id: "true"} or {id: {"answer": ...}}.
    out: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, dict) and "answer" in v:
            v = v["answer"]
        if isinstance(v, bool):
            v = "true" if v else "false"
        out[str(k)] = str(v).strip()
    return out


def answer_tree(model: str, graph: Dict[str, Any], definition: str, tree_temp: float) -> Dict[str, str]:
    """One batched LLM call: answer every question node in ``graph``.

    Returns {node_id: answer_label}. Retries once with a stricter prompt.
    """
    questions = collect_tree_questions(graph)
    if not questions:
        return {}

    # Valid answer labels per question node = outgoing edge labels.
    lines = []
    for q in questions:
        labels = [e["label"] for e in graph["edges"].get(q["id"], []) if e["label"]]
        opts = " | ".join(labels) if labels else "true | false"
        lines.append(f'  "{q["id"]}": {q["question"]}   (choose one of: {opts})')
    questions_block = "\n".join(lines)

    base_prompt = (
        "You are traversing an I-ADOPT decision tree titled "
        f"\"{graph['title']}\" to classify a scientific variable.\n"
        "Answer EACH question below for the variable. Pick exactly one of the "
        "allowed options shown in parentheses for each question id.\n\n"
        f"### Variable definition\n{definition}\n\n"
        f"### Questions (id: question (options))\n{questions_block}\n\n"
        "### Output format\n"
        "Return ONLY a JSON object mapping each question id to your chosen "
        'option string, e.g. {"B": "true", "C": "false"}. '
        "Use the exact option spelling. No other text."
    )
    stricter = base_prompt + (
        "\n\nIMPORTANT: previous output could not be parsed. Output EXACTLY one "
        "JSON object of {id: option}. Nothing else."
    )

    for attempt, prompt in enumerate((base_prompt, stricter), start=1):
        raw = _call_model_no_reasoning(model, prompt, tree_temp)
        parsed = _parse_answer_map(raw)
        if parsed:
            return parsed
        logger.warning("Tree %s: answer-map parse failed (attempt %d)", graph["id"], attempt)
    return {}


def _match_edge(edges: List[Dict[str, Any]], answer: str) -> Optional[Dict[str, Any]]:
    """Pick the outgoing edge whose label matches ``answer`` (case-insensitive).

    Booleans are normalised; named labels match on lowercased equality, then on
    substring as a lenient fallback.
    """
    if not edges:
        return None
    a = (answer or "").strip().lower()
    a_bool = a in ("true", "yes", "1", "y")
    # First: exact (normalised) match.
    for e in edges:
        lbl = e["label"].strip().lower()
        if lbl in ("true", "false"):
            if (lbl == "true") == a_bool and a in ("true", "false", "yes", "no", "1", "0", "y", "n"):
                return e
        elif lbl == a:
            return e
    # Second: lenient substring for named labels.
    for e in edges:
        lbl = e["label"].strip().lower()
        if lbl and (lbl in a or a in lbl):
            return e
    # Single unlabelled edge -> follow it.
    if len(edges) == 1:
        return edges[0]
    return None


def traverse_component(
    repo: PatternRepository,
    start_tree: str,
    definition: str,
    model: str,
    tree_temp: float,
    component: str,
) -> Dict[str, Any]:
    """Traverse one component's tree(s), following sub-tree jumps, to a pattern.

    Returns a dict with the selected pattern + the full decision path.
    """
    result: Dict[str, Any] = {
        "component": component,
        "start_tree": extract_tree_id(start_tree),
        "selected_pattern_id": None,
        "selected_pattern_path": None,
        "selected_pattern_name": None,
        "decision_path": [],  # list of {tree, node, question, answer, to}
        "errors": [],
    }

    visited_trees: set = set()
    tree_id = extract_tree_id(start_tree)

    for _hop in range(MAX_TREE_HOPS):
        if tree_id in visited_trees:
            result["errors"].append(f"cycle re-entering {tree_id}; stopping")
            break
        visited_trees.add(tree_id)

        graph = repo.load_decision_tree(tree_id)
        if not graph:
            result["errors"].append(f"could not load tree {tree_id}")
            break

        answers = answer_tree(model, graph, definition, tree_temp)

        # Walk the graph from root using the batched answers.
        node_id = graph["root"]
        for _step in range(MAX_TREE_HOPS * 4):
            node = graph["nodes"].get(node_id)
            if not node:
                result["errors"].append(f"{tree_id}: missing node {node_id}")
                break

            if node["kind"] == "pattern":
                pid = node["ref"]
                result["selected_pattern_id"] = pid
                result["selected_pattern_path"] = f"{repo.pattern_dir}/{pid}.yaml" if pid else None
                result["decision_path"].append(
                    {"tree": tree_id, "node": node_id, "question": "", "answer": "", "to": pid}
                )
                return result

            if node["kind"] == "subtree":
                result["decision_path"].append(
                    {"tree": tree_id, "node": node_id, "question": node["label"], "answer": "", "to": node["ref"]}
                )
                tree_id = node["ref"]
                break  # load the next tree (outer loop)

            # question node: choose an edge using the batched answer
            edges = graph["edges"].get(node_id, [])
            if not edges:
                result["errors"].append(f"{tree_id}: dead-end at {node_id}")
                node_id = None
                break
            ans = answers.get(node_id, "")
            edge = _match_edge(edges, ans)
            if edge is None:
                result["errors"].append(f"{tree_id}: no edge for node {node_id} answer={ans!r}")
                node_id = None
                break
            result["decision_path"].append(
                {"tree": tree_id, "node": node_id, "question": node["label"], "answer": ans, "to": edge["to"]}
            )
            node_id = edge["to"]
        else:
            result["errors"].append(f"{tree_id}: step budget exhausted")
            break

        if node_id is None:
            break  # error already recorded
        # else: we hit a subtree jump -> continue outer loop with new tree_id

    else:
        result["errors"].append("tree-hop budget exhausted")

    return result


def needs_matrix(definition: str) -> bool:
    """Heuristic: does the variable likely involve a matrix/medium?

    Used to decide whether to traverse the MATRIX tree. Conservative: a quick
    keyword check; the LLM still decides the actual matrix value.
    """
    d = (definition or "").lower()
    cues = (
        " in ",
        " within ",
        "atmosphere",
        "water",
        "soil",
        "air",
        "ocean",
        "sediment",
        "blood",
        "serum",
        "tissue",
        "medium",
        "column",
        "matrix",
    )
    return any(c in d for c in cues)


def traverse_all(
    repo: PatternRepository,
    definition: str,
    model: str,
    tree_temp: float,
    include_matrix: bool = True,
) -> Dict[str, Any]:
    """Run per-component traversal (Property, Object of Interest, Matrix)."""
    components: Dict[str, Any] = {}
    components["property"] = traverse_component(repo, PROPERTY_START_TREE, definition, model, tree_temp, "property")
    components["object_of_interest"] = traverse_component(
        repo, OOI_START_TREE, definition, model, tree_temp, "object_of_interest"
    )
    if include_matrix and needs_matrix(definition):
        components["matrix"] = traverse_component(repo, MATRIX_START_TREE, definition, model, tree_temp, "matrix")
    return components


def selected_pattern_ids(components: Dict[str, Any]) -> List[str]:
    return [c["selected_pattern_id"] for c in components.values() if c.get("selected_pattern_id")]


def decision_path_str(components: Dict[str, Any]) -> str:
    """Compact, human-readable multi-component path for logs/Excel."""
    parts = []
    for comp, c in components.items():
        chain = " -> ".join(
            f"{s['tree']}:{s['node']}={s['answer']}" if s["answer"] else f"{s['tree']}:{s['node']}->{s['to']}"
            for s in c.get("decision_path", [])
        )
        parts.append(f"[{comp}] {chain} => {c.get('selected_pattern_id')}")
    return "  ||  ".join(parts)


# --------------------------------------------------------------------------- #
# 5. Pattern-guided decomposition prompt (full YAML inline, no links)
# --------------------------------------------------------------------------- #
def _vdp_field(vdp: Dict[str, Any], *names: str, default: Any = None) -> Any:
    if not isinstance(vdp, dict):
        return default
    for n in names:
        if n in vdp and vdp[n] not in (None, "", [], "~"):
            return vdp[n]
    return default


def _as_list(val: Any) -> List[Any]:
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def involved_components_of(vdp: Optional[Dict[str, Any]]) -> List[str]:
    """Return the pattern's involved components (handles schema variants)."""
    if not vdp:
        return []
    involved = _as_list(_vdp_field(vdp, "involved_components"))
    if not involved:
        affected = _vdp_field(vdp, "affected")
        if isinstance(affected, dict):
            involved = _as_list(affected.get("components"))
    return [str(c).strip() for c in involved if c not in (None, "", "~")]


# URL-valued fields we never want as raw links in the prompt.
_URL_VALUE_FIELDS_DROP = ("id", "gh_issue")
_URL_VALUE_FIELDS_TO_IDS = ("associated_patterns", "associated_pattern", "associated_mapping")


def _sanitize_vdp_for_prompt(vdp: Dict[str, Any], pattern_id: str) -> Dict[str, Any]:
    """Copy of the VDP with GitHub URLs dropped / rewritten to plain ids."""
    clean: Dict[str, Any] = {}
    for k, v in vdp.items():
        if k in _URL_VALUE_FIELDS_DROP:
            continue
        if k in _URL_VALUE_FIELDS_TO_IDS:
            ids = [extract_pattern_id(x) for x in _as_list(v) if x not in (None, "", "~")]
            if ids:
                clean[k] = ids
            continue
        clean[k] = v
    clean["pattern_id"] = pattern_id
    return clean


def render_pattern_block(component: str, vdp: Optional[Dict[str, Any]], pattern_id: str) -> str:
    """Render one selected pattern: short header + full sanitized YAML inline."""
    header = f"# Component: {component} -> {pattern_id or '(none selected)'}"
    if not vdp:
        return f"{header}\n(no pattern details available)"
    name = _vdp_field(vdp, "name")
    sanitized = _sanitize_vdp_for_prompt(vdp, pattern_id)
    try:
        yaml_text = yaml.safe_dump(sanitized, sort_keys=False, allow_unicode=True).strip()
    except Exception:  # pragma: no cover - defensive
        yaml_text = json.dumps(sanitized, ensure_ascii=False, indent=2)
    title = f"{header} ({name})" if name else header
    return f"{title}\n{yaml_text}"


def build_pattern_prompt(
    definition: str,
    examples: List[Dict[str, Any]],
    components: Dict[str, Any],
    vdps: Dict[str, Optional[Dict[str, Any]]],
) -> str:
    """Build the multi-component pattern-guided decomposition prompt.

    Embeds the FULL YAML of every selected pattern (Property, OoI, Matrix).
    Reuses the baseline schema text and example-block formatting so the output
    stays schema-compatible with the baseline.
    """
    blocks = []
    for comp, c in components.items():
        pid = c.get("selected_pattern_id")
        blocks.append(render_pattern_block(comp, vdps.get(comp), pid))
    patterns_block = "\n\n".join(blocks) if blocks else "(no patterns selected)"

    ex_block = ""
    if examples:
        formatted = [format_example_block(ex, i + 1) for i, ex in enumerate(examples)]
        ex_block = _EXAMPLE_HDR + "".join(formatted)

    # Structured-system guidance: when the selected Property/OoI pattern implies a
    # System object of interest, the model MUST emit the FULL structured object
    # (with real component labels), never the bare type word "AsymmetricSystem".
    selected = {c.get("selected_pattern_id") for c in components.values() if c.get("selected_pattern_id")}
    system_note = ""
    if selected & _ASYM_OOI_PATTERNS:
        system_note = (
            "\nThe selected pattern(s) imply that hasObjectOfInterest is an "
            "ASYMMETRIC SYSTEM (a ratio/relation between two different entities). "
            "You MUST output it as a JSON object with REAL entity labels, e.g.:\n"
            '  "hasObjectOfInterest": {\n'
            '    "AsymmetricSystem": "<short name of the whole system>",\n'
            '    "hasSource": "<the numerator entity>",\n'
            '    "hasTarget": "<the denominator entity>",\n'
            '    "hasNumerator": "<the numerator entity>",\n'
            '    "hasDenominator": "<the denominator entity>"\n'
            "  }\n"
            'NEVER output the literal string "AsymmetricSystem" as the value, and '
            "never leave the parts empty — fill them with the actual entities from "
            "the definition."
        )
    elif selected & _SYM_OOI_PATTERNS:
        system_note = (
            "\nThe selected pattern(s) imply that hasObjectOfInterest is a "
            "SYMMETRIC SYSTEM (two or more equal parts forming one system). You "
            "MUST output it as a JSON object with REAL entity labels, e.g.:\n"
            '  "hasObjectOfInterest": {\n'
            '    "SymmetricSystem": "<short name of the whole system>",\n'
            '    "hasPart": ["<entity 1>", "<entity 2>"]\n'
            "  }\n"
            'NEVER output the literal string "SymmetricSystem" as the value, and '
            "never leave hasPart empty."
        )

    instructions = (
        "Decompose the scientific variable below into the I-ADOPT JSON structure.\n"
        "You are given one selected Variable Design Pattern per I-ADOPT component "
        "(property, object of interest, and possibly matrix). You MUST use these "
        "patterns as modelling guidance: follow each pattern's `design`, `rules` "
        "and `involved_components` to decide how to fill the corresponding I-ADOPT "
        "fields. Do NOT invent a decomposition outside what the patterns permit.\n"
        "For Entity/System fields (hasObjectOfInterest, hasMatrix, hasContextObject): "
        "use a plain string for a simple entity, OR a structured object for a system. "
        "If you use a system, you MUST include its parts with real labels — never "
        'output the bare type word ("AsymmetricSystem"/"SymmetricSystem") as the value.'
        f"{system_note}\n"
        "Follow the JSON-Schema exactly. `definition` must be exactly the provided "
        "string. If a key is not supported by the patterns or not present in the "
        "definition, use an empty string (or empty list for hasConstraint).\n"
        "Output ONLY the JSON object."
    )

    return (
        f"{instructions}\n\n"
        f"### Selected I-ADOPT Variable Design Patterns (guidance)\n{patterns_block}\n\n"
        f"### JSON-Schema\n{_SCHEMA_TEXT}\n"
        f"{ex_block}"
        f"{_USER_HDR}Variable's definition to decompose: {definition}"
        f"{_EXPECTED_HDR}"
    )


# Bare type words that are meaningless as an Entity/System value on their own.
_BARE_SYSTEM_WORDS = {"asymmetricsystem", "symmetricsystem", "system", "entity", "entityorsystem"}
_SYSTEM_FIELDS = ("hasObjectOfInterest", "hasMatrix", "hasContextObject")


def _repair_degenerate_systems(pred: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Detect/repair Entity-or-System fields that the LLM filled degenerately.

    A value like ``"hasObjectOfInterest": "AsymmetricSystem"`` (the bare type
    word, with no parts) is meaningless. We blank such values so they do not
    masquerade as a real prediction, and likewise blank structured systems whose
    component parts are all empty. Returns (repaired_pred, notes).
    """
    notes: List[str] = []
    for field in _SYSTEM_FIELDS:
        val = pred.get(field)
        if isinstance(val, str) and val.strip().lower().replace(" ", "") in _BARE_SYSTEM_WORDS:
            notes.append(f"{field}='{val}' is a bare system type word with no parts; blanked")
            pred[field] = ""
        elif isinstance(val, dict):
            if "AsymmetricSystem" in val:
                parts = [val.get("hasSource"), val.get("hasTarget"), val.get("hasNumerator"), val.get("hasDenominator")]
                if not any(isinstance(x, str) and x.strip() for x in parts):
                    notes.append(f"{field} is an AsymmetricSystem with no parts; blanked")
                    pred[field] = ""
            elif "SymmetricSystem" in val:
                parts = val.get("hasPart") or []
                if not (isinstance(parts, list) and any(isinstance(x, str) and x.strip() for x in parts)):
                    notes.append(f"{field} is a SymmetricSystem with no parts; blanked")
                    pred[field] = ""
    return pred, notes


def pattern_guided_decompose(
    model: str, prompt: str, gt_label: str, definition: str, temperature: float
) -> Tuple[Dict[str, Any], str, List[str]]:
    """LLM decomposition using the vendored parse/coerce primitives.

    Returns (prediction_dict, last_raw_response, repair_notes). A response whose
    structured-system field is degenerate (e.g. the bare word "AsymmetricSystem")
    triggers one stricter retry before we accept and repair it.
    """
    last_raw = ""
    stricter_suffix = (
        "\n\nIMPORTANT: a previous answer used the bare word "
        '"AsymmetricSystem"/"SymmetricSystem" as a field value. That is invalid. '
        "If a field is a system, output the full object with real part labels; "
        "otherwise output a plain entity string."
    )
    for attempt in range(1, 4):
        use_prompt = prompt if attempt == 1 else (prompt + stricter_suffix)
        raw = call_model(model, use_prompt, temperature)
        last_raw = raw or last_raw
        if not raw.strip():
            continue
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            logger.warning("%s: no JSON block on decompose attempt %d", model, attempt)
            continue
        try:
            data = json.loads(m.group(0))
        except Exception as e:
            logger.warning("%s: JSON decode failure attempt %d – %r", model, attempt, e)
            continue
        data["label"] = gt_label
        data["definition"] = definition
        pred = coerce_prediction(data)
        repaired, notes = _repair_degenerate_systems(pred)
        # If a degenerate system was found and we still have retries, try again
        # for a properly-structured answer before accepting the blanked version.
        if notes and attempt < 3:
            logger.warning("%s: degenerate system on attempt %d (%s); retrying", model, attempt, notes)
            continue
        return repaired, last_raw, notes
    logger.error("%s: could not extract pattern-guided JSON after 3 attempts", model)
    return {}, last_raw, ["no valid JSON after 3 attempts"]


# --------------------------------------------------------------------------- #
# 6. Validation layers
# --------------------------------------------------------------------------- #
try:
    import jsonschema as _jsonschema  # optional

    _SCHEMA_OBJ = json.loads(_SCHEMA_TEXT)
except Exception:  # pragma: no cover - jsonschema optional
    _jsonschema = None
    _SCHEMA_OBJ = None

_SCHEMA_REQUIRED = ["label", "definition", "comment", "hasProperty", "hasObjectOfInterest"]


def validate_against_json_schema(output_json: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Existing JSON-schema validation (jsonschema if available, else key check)."""
    errors: List[str] = []
    if not isinstance(output_json, dict):
        return False, ["output is not a JSON object"]
    if _jsonschema is not None and _SCHEMA_OBJ is not None:
        validator = _jsonschema.Draft202012Validator(_SCHEMA_OBJ)
        for err in sorted(validator.iter_errors(output_json), key=lambda e: list(e.path)):
            errors.append(f"{'/'.join(str(p) for p in err.path)}: {err.message}")
        return (len(errors) == 0), errors
    for k in _SCHEMA_REQUIRED:
        if k not in output_json or output_json.get(k) in (None, ""):
            errors.append(f"missing/empty required key: {k}")
    return (len(errors) == 0), errors


def _component_present(output_json: Dict[str, Any], key: str) -> bool:
    val = output_json.get(key)
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, dict):
        return any(bool(str(v).strip()) for v in val.values())
    if isinstance(val, list):
        return len(val) > 0
    return bool(val)


def _ooi_kind(val: Any) -> str:
    """Classify a hasObjectOfInterest value: 'asymmetric' | 'symmetric' | 'simple' | 'empty'."""
    if isinstance(val, dict):
        if "AsymmetricSystem" in val:
            return "asymmetric"
        if "SymmetricSystem" in val:
            return "symmetric"
        return "simple"
    if isinstance(val, str) and val.strip():
        return "simple"
    return "empty"


# Pattern-id families that imply a structured Object of Interest.
_ASYM_OOI_PATTERNS = {"VDP27", "VDP17", "VDP31", "VDP14", "VDP06", "VDP07", "VDP08", "VDP09"}
_SYM_OOI_PATTERNS = {"VDP30"}


def validate_against_selected_pattern(
    output_json: Dict[str, Any],
    components: Dict[str, Any],
    vdps: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Deterministic, rule-based validation against ALL selected patterns.

    Lenient by design: structural mismatches that we cannot robustly verify are
    warnings, not hard failures.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # (a) involved_components present
    for comp, vdp in vdps.items():
        for ic in (s.lower() for s in involved_components_of(vdp)):
            if "property" in ic and not _component_present(output_json, "hasProperty"):
                errors.append(f"{comp} pattern requires property but hasProperty empty")
            elif ("object of interest" in ic or ic == "ooi") and not _component_present(
                output_json, "hasObjectOfInterest"
            ):
                errors.append(f"{comp} pattern requires OoI but hasObjectOfInterest empty")
            elif "matrix" in ic and not _component_present(output_json, "hasMatrix"):
                warnings.append(f"{comp} pattern mentions matrix but hasMatrix empty")

    # (b) Object-of-Interest structure implied by the OoI pattern family
    ooi_pid = (components.get("object_of_interest") or {}).get("selected_pattern_id")
    prop_pid = (components.get("property") or {}).get("selected_pattern_id")
    kind = _ooi_kind(output_json.get("hasObjectOfInterest"))
    implied = {p for p in (ooi_pid, prop_pid) if p}
    if implied & _ASYM_OOI_PATTERNS and kind != "asymmetric":
        warnings.append(f"pattern(s) {sorted(implied & _ASYM_OOI_PATTERNS)} imply an AsymmetricSystem OoI, got {kind}")
    if implied & _SYM_OOI_PATTERNS and kind != "symmetric":
        warnings.append(f"pattern(s) {sorted(implied & _SYM_OOI_PATTERNS)} imply a SymmetricSystem OoI, got {kind}")

    return {
        "pattern_valid": len(errors) == 0,
        "pattern_validation_errors": errors,
        "pattern_validation_warnings": warnings,
    }


# --- gold-signature pattern confusion (rule based, no gold pattern label) --- #
def gold_pattern_signature(gt: Dict[str, Any]) -> set:
    """Derive expected pattern *families* from the gold JSON structure.

    Families (deterministic tokens): "property", "ooi-single", "ooi-asym",
    "ooi-sym", "matrix", "constraint".
    """
    sig = set()
    if _component_present(gt, "hasProperty"):
        sig.add("property")
    kind = _ooi_kind(gt.get("hasObjectOfInterest"))
    if kind == "asymmetric":
        sig.add("ooi-asym")
    elif kind == "symmetric":
        sig.add("ooi-sym")
    elif kind == "simple":
        sig.add("ooi-single")
    if _component_present(gt, "hasMatrix"):
        sig.add("matrix")
    if gt.get("hasConstraint"):
        sig.add("constraint")
    return sig


def predicted_pattern_signature(components: Dict[str, Any]) -> set:
    """Derive the same family tokens from the selected patterns."""
    sig = set()
    prop = (components.get("property") or {}).get("selected_pattern_id")
    if prop:
        sig.add("property")
    ooi = (components.get("object_of_interest") or {}).get("selected_pattern_id")
    if ooi:
        if ooi in _ASYM_OOI_PATTERNS:
            sig.add("ooi-asym")
        elif ooi in _SYM_OOI_PATTERNS:
            sig.add("ooi-sym")
        else:
            sig.add("ooi-single")
    if (components.get("matrix") or {}).get("selected_pattern_id"):
        sig.add("matrix")
    return sig


# The fixed universe of comparable pattern families. TN counts the families
# that are (correctly) absent from BOTH gold and prediction.
_PATTERN_FAMILY_UNIVERSE = {"property", "ooi-single", "ooi-asym", "ooi-sym", "matrix"}


def pattern_selection_confusion(gt: Dict[str, Any], components: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based TP/FP/FN/TN over pattern-family signatures (gold vs selected).

    Same confusion-count style as the baseline (TP/FP/FN/TN) rather than P/R/F,
    so it is directly comparable. Constraints are not traversed here, so the
    'constraint' family is excluded from the comparison.
    """
    g = gold_pattern_signature(gt)
    g_cmp = {x for x in g if x != "constraint"}  # only families traversal attempts
    p = predicted_pattern_signature(components)
    tp = len(g_cmp & p)
    fp = len(p - g_cmp)
    fn = len(g_cmp - p)
    tn = len(_PATTERN_FAMILY_UNIVERSE - (g_cmp | p))
    return {
        "gold_signature": sorted(g),
        "pred_signature": sorted(p),
        "pattern_tp": tp,
        "pattern_fp": fp,
        "pattern_fn": fn,
        "pattern_tn": tn,
    }


def shacl_validate_placeholder(output_json: Dict[str, Any], repo: PatternRepository) -> Dict[str, Any]:
    """Documented placeholder for future SHACL validation (pyshacl).

    Only two shapes ship today; the baseline does no RDF/SHACL validation. Kept
    as a no-op so the benchmark never breaks. Wire in pyshacl here later.
    """
    return {
        "shacl_checked": False,
        "shacl_available_shapes": [p.name for p in repo.shacl_files()],
        "shacl_note": "SHACL validation not implemented (placeholder).",
    }


# --------------------------------------------------------------------------- #
# 7. Per-variable worker
# --------------------------------------------------------------------------- #
def _load_vdps(repo: PatternRepository, components: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    vdps: Dict[str, Optional[Dict[str, Any]]] = {}
    for comp, c in components.items():
        pid = c.get("selected_pattern_id")
        vdp = repo.load_variable_design_pattern(pid) if pid else None
        if vdp:
            c["selected_pattern_name"] = _vdp_field(vdp, "name")
        vdps[comp] = vdp
    return vdps


def _emit_variable_log(rec: Dict[str, Any]) -> None:
    """Write ONE concise, ordered block per variable to the log.

    Order requested: complete prompt -> decision-tree path -> ground truth ->
    predicted (labels only) -> wikidata -> evaluations.
    """
    sep = "═" * 100
    lines = [
        sep,
        f"VARIABLE #{rec['var_index']} | {rec['variable']} | path={rec['path']} | model={rec['model']}",
        sep,
        "── PROMPT ──",
        rec["pattern_prompt"],
        "── DECISION-TREE PATH ──",
        rec["decision_tree_path"],
        f"selected patterns: {rec['selected_pattern_id']}",
        "── GROUND TRUTH (JSON) ──",
        json.dumps(rec["ground_truth_json"], indent=2, ensure_ascii=False),
        "── PREDICTED (labels only) ──",
        json.dumps(rec["predicted_json"], indent=2, ensure_ascii=False),
        "── WIKIDATA (predicted + URIs) ──",
        f"status: {rec['wikidata_linking_status']}",
        json.dumps(rec["predicted_json_with_uris"], indent=2, ensure_ascii=False),
        "── EVALUATIONS ──",
        "EXACT CONFUSION | TP={tp:.3f} FP={fp:.3f} FN={fn:.3f} TN={tn:.3f}".format(**rec["confusion"]["exact_totals"]),
        "CLOSE CONFUSION | TP={tp:.3f} FP={fp:.3f} FN={fn:.3f} TN={tn:.3f}".format(**rec["confusion"]["close_totals"]),
        "jaccard both/concept/text = {:.3f}/{:.3f}/{:.3f}".format(rec["j_both"], rec["j_concept"], rec["j_text"]),
        "URI: total={} correct={} acc={:.3f} coverage={:.3f}".format(
            rec["uris_total"], rec["uris_correct"], rec["uris_acc"], rec["uris_coverage"]
        ),
        "PATTERN-SELECTION CONFUSION | TP={} FP={} FN={} TN={} (gold={} pred={})".format(
            rec["pattern_tp"],
            rec["pattern_fp"],
            rec["pattern_fn"],
            rec["pattern_tn"],
            rec["gold_signature"],
            rec["pred_signature"],
        ),
        "json_schema_valid={} | pattern_valid={}".format(rec["json_schema_valid"], rec["pattern_valid"]),
        f"pattern_validation_errors={rec['pattern_validation_errors']}",
        f"pattern_validation_warnings={rec['pattern_validation_warnings']}",
        f"system_repairs={rec.get('system_repairs', '[]')}",
        sep,
    ]
    with _log_lock:
        logger.info("\n" + "\n".join(lines))


def _run_one_pattern(
    repo: PatternRepository,
    model: str,
    temperature: float,
    tree_temp: float,
    approach: str,
    link_model_name: str,
    threshold: float,
    shot: int,
    examples: List[Dict[str, Any]],
    gt: Dict[str, Any],
    gt_path: str,
    var_index: int,
    include_matrix: bool,
) -> Optional[Dict[str, Any]]:
    """Process one variable end-to-end. Never raises: failures are recorded."""
    label = gt.get("label", "")
    definition = gt.get("definition") or gt.get("comment") or ""

    try:
        # --- multi-component traversal -> select patterns ---------------- #
        components = traverse_all(repo, definition, model, tree_temp, include_matrix=include_matrix)
        vdps = _load_vdps(repo, components)
        sel_ids = selected_pattern_ids(components)
        logger.info("VAR #%d %s | patterns=%s", var_index, label, sel_ids)

        # --- pattern-guided decomposition -------------------------------- #
        prompt = build_pattern_prompt(definition, examples, components, vdps)
        pred, raw_resp, repair_notes = pattern_guided_decompose(model, prompt, label, definition, temperature)
        if repair_notes:
            logger.warning("VAR #%d %s | system repairs: %s", var_index, label, repair_notes)

        # --- JSON schema validation (existing) --------------------------- #
        schema_valid, schema_errors = validate_against_json_schema(pred)

        # --- pattern-level validation (rule based) ----------------------- #
        pattern_val = validate_against_selected_pattern(pred, components, vdps)
        _ = shacl_validate_placeholder(pred, repo)  # placeholder
        psel = pattern_selection_confusion(gt, components)

        # --- Wikidata linking (unchanged from baseline) ------------------ #
        wikidata_status = "ok"
        try:
            pred_enriched = enrich_with_uris(pred, approach=approach, model_name=link_model_name, threshold=threshold)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("VAR #%d %s | Wikidata linking failed: %r", var_index, label, e)
            pred_enriched = pred
            wikidata_status = f"error: {e!r}"

        # --- scoring (unchanged) ----------------------------------------- #
        confusion_data = compute_confusion_for_pair(gt, pred)
        j_both = jaccard(atoms(gt, "both"), atoms(pred, "both"))
        j_concept = jaccard(atoms(gt, "concept"), atoms(pred, "concept"))
        j_text = jaccard(atoms(gt, "text"), atoms(pred, "text"))
        uris_total, uris_correct, uris_acc, uris_coverage, uris_predicted, uri_flags = compare_uris(gt, pred_enriched)

        rec = {
            # baseline-compatible fields (so compute_summary_metrics works)
            "variable": label,
            "path": gt_path,
            "model": model,
            "temperature": temperature,
            "prompt_version": "patternGuided",
            "shot": shot,
            "link_approach": approach,
            "link_model_name": link_model_name,
            "link_threshold": threshold,
            "prompt": prompt,
            "ground_truth_json": gt,
            "predicted_json": pred,
            "predicted_json_with_uris": pred_enriched,
            "confusion": confusion_data,
            "j_both": j_both,
            "j_concept": j_concept,
            "j_text": j_text,
            "uris_total": uris_total,
            "uris_correct": uris_correct,
            "uris_acc": uris_acc,
            "uris_coverage": uris_coverage,
            "uris_predicted": uris_predicted,
            "uri_flags": uri_flags,
            # pattern-guided specific fields
            "var_index": var_index,
            "components": components,
            "selected_pattern_id": ", ".join(sel_ids),
            "selected_pattern_name": "; ".join(
                f"{comp}:{c.get('selected_pattern_name')}"
                for comp, c in components.items()
                if c.get("selected_pattern_name")
            ),
            "selected_pattern_path": "; ".join(
                str(c.get("selected_pattern_path")) for c in components.values() if c.get("selected_pattern_path")
            ),
            "decision_tree_path": decision_path_str(components),
            "decision_tree_answers": json.dumps(
                {comp: c.get("decision_path", []) for comp, c in components.items()},
                ensure_ascii=False,
            ),
            "traversal_errors": json.dumps(
                {comp: c.get("errors", []) for comp, c in components.items()}, ensure_ascii=False
            ),
            "pattern_valid": pattern_val["pattern_valid"],
            "pattern_validation_errors": json.dumps(pattern_val["pattern_validation_errors"], ensure_ascii=False),
            "pattern_validation_warnings": json.dumps(pattern_val["pattern_validation_warnings"], ensure_ascii=False),
            "system_repairs": json.dumps(repair_notes, ensure_ascii=False),
            "pattern_prompt": prompt,
            "pattern_guided_raw_llm_response": raw_resp,
            "json_schema_valid": schema_valid,
            "json_schema_errors": json.dumps(schema_errors, ensure_ascii=False),
            "wikidata_linking_status": wikidata_status,
            # gold-signature pattern-selection confusion (TP/FP/FN/TN)
            "pattern_tp": psel["pattern_tp"],
            "pattern_fp": psel["pattern_fp"],
            "pattern_fn": psel["pattern_fn"],
            "pattern_tn": psel["pattern_tn"],
            "gold_signature": psel["gold_signature"],
            "pred_signature": psel["pred_signature"],
        }

        _emit_variable_log(rec)
        return rec

    except Exception as e:  # one variable must never stop the run
        logger.exception("VAR #%d %s | FAILED, continuing: %r", var_index, label, e)
        return {
            "variable": label,
            "path": gt_path,
            "model": model,
            "temperature": temperature,
            "prompt_version": "patternGuided",
            "shot": shot,
            "link_approach": approach,
            "link_model_name": link_model_name,
            "link_threshold": threshold,
            "prompt": "",
            "ground_truth_json": gt,
            "predicted_json": {},
            "predicted_json_with_uris": {},
            "confusion": compute_confusion_for_pair(gt, {}),
            "j_both": 0.0,
            "j_concept": 0.0,
            "j_text": 0.0,
            "uris_total": 0,
            "uris_correct": 0,
            "uris_acc": 0.0,
            "uris_coverage": 0.0,
            "uris_predicted": 0,
            "uri_flags": {},
            "var_index": var_index,
            "components": {},
            "selected_pattern_id": "",
            "selected_pattern_name": "",
            "selected_pattern_path": "",
            "decision_tree_path": "",
            "decision_tree_answers": "{}",
            "traversal_errors": json.dumps({"worker": [f"{e!r}"]}, ensure_ascii=False),
            "pattern_valid": False,
            "pattern_validation_errors": json.dumps([f"worker exception: {e!r}"]),
            "pattern_validation_warnings": "[]",
            "system_repairs": "[]",
            "pattern_prompt": "",
            "pattern_guided_raw_llm_response": "",
            "json_schema_valid": False,
            "json_schema_errors": "[]",
            "wikidata_linking_status": "skipped (worker failed)",
            "pattern_tp": 0,
            "pattern_fp": 0,
            "pattern_fn": 0,
            "pattern_tn": 0,
            "gold_signature": [],
            "pred_signature": [],
        }


# --------------------------------------------------------------------------- #
# 8. Evaluation loop
# --------------------------------------------------------------------------- #
def evaluate_pattern_guided(
    repo: PatternRepository,
    data_dir: pathlib.Path,
    models: List[str],
    temps: List[float],
    tree_temp: float,
    shot: int,
    approach: str,
    link_model_name: str,
    threshold: float,
    workers: int,
    limit: int,
    start_index: int,
    include_matrix: bool,
) -> List[Dict[str, Any]]:
    examples = load_examples(shot)
    example_labels = {ex.get("label") for ex in examples if ex.get("label")}

    gt_items = load_gt_files_recursive(data_dir, max_vars=0)
    if start_index:
        gt_items = gt_items[start_index:]
    if limit:
        gt_items = gt_items[:limit]

    tasks = []
    idx = start_index
    for p, gt in gt_items:
        idx += 1
        if gt.get("label") in example_labels:
            logger.info("Skip %s (in-prompt example)", gt.get("label"))
            continue
        try:
            rel_path = str(p.relative_to(data_dir.parent))
        except Exception:
            rel_path = str(p)
        for model in models:
            for temp in temps:
                tasks.append((model, temp, gt, rel_path, idx))

    results: List[Dict[str, Any]] = []
    max_workers = min(os.cpu_count() or 4, workers) if approach == "cross-encoder" else workers

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [
            pool.submit(
                _run_one_pattern,
                repo,
                model,
                temp,
                tree_temp,
                approach,
                link_model_name,
                threshold,
                shot,
                examples,
                gt,
                rel_path,
                vidx,
                include_matrix,
            )
            for (model, temp, gt, rel_path, vidx) in tasks
        ]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                logger.exception("Worker failed but run continues")
                continue
            if r:
                results.append(r)

    results.sort(key=lambda r: r.get("var_index", 0))
    return results


def dry_run_pattern_selection(
    repo: PatternRepository,
    data_dir: pathlib.Path,
    model: str,
    tree_temp: float,
    limit: int,
    start_index: int,
    include_matrix: bool,
) -> List[Dict[str, Any]]:
    """Traverse + select patterns per variable; no decomposition/Wikidata."""
    gt_items = load_gt_files_recursive(data_dir, max_vars=0)
    if start_index:
        gt_items = gt_items[start_index:]
    if limit:
        gt_items = gt_items[:limit]

    rows = []
    for i, (p, gt) in enumerate(gt_items, start=start_index + 1):
        definition = gt.get("definition") or gt.get("comment") or ""
        components = traverse_all(repo, definition, model, tree_temp, include_matrix=include_matrix)
        psel = pattern_selection_confusion(gt, components)
        logger.info(
            "DRY #%d | %s | patterns=%s | %s | TP/FP/FN/TN=%d/%d/%d/%d",
            i,
            gt.get("label"),
            selected_pattern_ids(components),
            decision_path_str(components),
            psel["pattern_tp"],
            psel["pattern_fp"],
            psel["pattern_fn"],
            psel["pattern_tn"],
        )
        rows.append(
            {
                "variable": gt.get("label"),
                "selected_pattern_ids": ", ".join(selected_pattern_ids(components)),
                "decision_tree_path": decision_path_str(components),
                "gold_signature": ", ".join(psel["gold_signature"]),
                "pred_signature": ", ".join(psel["pred_signature"]),
                "pattern_tp": psel["pattern_tp"],
                "pattern_fp": psel["pattern_fp"],
                "pattern_fn": psel["pattern_fn"],
                "pattern_tn": psel["pattern_tn"],
                "errors": json.dumps({c: comp.get("errors", []) for c, comp in components.items()}, ensure_ascii=False),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 9. Excel output (baseline-comparable + pattern columns + pattern-eval sheet)
# --------------------------------------------------------------------------- #
def timestamped_output(user_output: Optional[pathlib.Path], prefix: str) -> pathlib.Path:
    """Always append a date+time stamp to the output filename."""
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    if user_output is None:
        return OUTBOOK_DIR / f"{prefix}_{stamp}.xlsx"
    p = pathlib.Path(user_output)
    return p.with_name(f"{p.stem}_{stamp}{p.suffix or '.xlsx'}")


def build_excel_pattern(results: List[Dict[str, Any]], out_path: pathlib.Path) -> pathlib.Path:
    if not results:
        raise RuntimeError("No results to write.")
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as wr:
        # 1) one sheet per ONTO_KEY (same style as baseline)
        for key in ONTO_KEYS:
            rows = []
            for r in results:
                gt = r["ground_truth_json"]
                pred = r["predicted_json"]
                gt_val = strip_all_uri_fields(gt.get(key, [] if key == "hasConstraint" else ""))
                pred_val = strip_all_uri_fields(pred.get(key, [] if key == "hasConstraint" else ""))
                rows.append(
                    {
                        "variable": r["variable"],
                        "path": r["path"],
                        "model": r["model"],
                        "temperature": r["temperature"],
                        "prompt_version": r["prompt_version"],
                        "shot": r["shot"],
                        "link_approach": r["link_approach"],
                        "selected_pattern_id": r.get("selected_pattern_id"),
                        "ground_truth": json.dumps(gt_val, ensure_ascii=False, indent=2),
                        "predicted": json.dumps(pred_val, ensure_ascii=False, indent=2),
                    }
                )
            pd.DataFrame(rows).to_excel(wr, sheet_name=f"{key} concepts"[:31], index=False)

        # 2) LLM outputs (baseline columns + pattern-guided columns)
        json_rows = []
        for r in results:
            json_rows.append(
                {
                    "variable": r["variable"],
                    "path": r["path"],
                    "model": r["model"],
                    "temperature": r["temperature"],
                    "prompt_version": r["prompt_version"],
                    "shot": r["shot"],
                    "link_approach": r["link_approach"],
                    "link_model_name": r["link_model_name"],
                    "link_threshold": r["link_threshold"],
                    # pattern-guided columns
                    "selected_pattern_id": r.get("selected_pattern_id"),
                    "selected_pattern_name": r.get("selected_pattern_name"),
                    "selected_pattern_path": r.get("selected_pattern_path"),
                    "decision_tree_path": r.get("decision_tree_path"),
                    "decision_tree_answers": r.get("decision_tree_answers"),
                    "traversal_errors": r.get("traversal_errors"),
                    "json_schema_valid": r.get("json_schema_valid"),
                    "json_schema_errors": r.get("json_schema_errors"),
                    "pattern_valid": r.get("pattern_valid"),
                    "pattern_validation_errors": r.get("pattern_validation_errors"),
                    "pattern_validation_warnings": r.get("pattern_validation_warnings"),
                    "system_repairs": r.get("system_repairs"),
                    "pattern_tp": r.get("pattern_tp"),
                    "pattern_fp": r.get("pattern_fp"),
                    "pattern_fn": r.get("pattern_fn"),
                    "pattern_tn": r.get("pattern_tn"),
                    "gold_signature": ", ".join(r.get("gold_signature", [])),
                    "pred_signature": ", ".join(r.get("pred_signature", [])),
                    "wikidata_linking_status": r.get("wikidata_linking_status"),
                    # URI eval summary (baseline)
                    "uris_total": r["uris_total"],
                    "uris_correct": r["uris_correct"],
                    "uris_acc": round(r["uris_acc"], 3),
                    "uris_coverage": round(r["uris_coverage"], 3),
                    "uris_predicted": r["uris_predicted"],
                    # prompts + raw responses + JSONs
                    "pattern_prompt": r.get("pattern_prompt"),
                    "pattern_guided_raw_llm_response": r.get("pattern_guided_raw_llm_response"),
                    "ground_truth_json": json.dumps(r["ground_truth_json"], ensure_ascii=False, indent=2),
                    "predicted_json": json.dumps(r["predicted_json"], ensure_ascii=False, indent=2),
                    "predicted_json_with_uris": json.dumps(r["predicted_json_with_uris"], ensure_ascii=False, indent=2),
                }
            )
        pd.DataFrame(json_rows).to_excel(wr, sheet_name="LLM outputs", index=False)

        # 3) Pattern evaluation sheet (rule-based TP/FP/FN/TN, concise)
        peval = []
        for r in results:
            peval.append(
                {
                    "variable": r["variable"],
                    "selected_patterns": r.get("selected_pattern_id"),
                    "decision_tree_path": r.get("decision_tree_path"),
                    "gold_signature": ", ".join(r.get("gold_signature", [])),
                    "pred_signature": ", ".join(r.get("pred_signature", [])),
                    "pattern_tp": r.get("pattern_tp"),
                    "pattern_fp": r.get("pattern_fp"),
                    "pattern_fn": r.get("pattern_fn"),
                    "pattern_tn": r.get("pattern_tn"),
                    "pattern_valid": r.get("pattern_valid"),
                    "pattern_validation_errors": r.get("pattern_validation_errors"),
                    "pattern_validation_warnings": r.get("pattern_validation_warnings"),
                    "json_schema_valid": r.get("json_schema_valid"),
                }
            )
        df_peval = pd.DataFrame(peval)
        if not df_peval.empty:
            total_row = {
                "variable": "TOTAL",
                "pattern_tp": int(df_peval["pattern_tp"].sum()),
                "pattern_fp": int(df_peval["pattern_fp"].sum()),
                "pattern_fn": int(df_peval["pattern_fn"].sum()),
                "pattern_tn": int(df_peval["pattern_tn"].sum()),
            }
            df_peval = pd.concat([df_peval, pd.DataFrame([total_row])], ignore_index=True)
        df_peval.to_excel(wr, sheet_name="Pattern eval", index=False)

        # 4) Summary (reuse baseline scoring; format unchanged for comparability)
        summary = compute_summary_metrics(results)
        summary.to_excel(wr, sheet_name="Summary", index=False)

    logger.info("✓ Pattern-guided results saved → %s", out_path.resolve())
    return out_path


# --------------------------------------------------------------------------- #
# Self-tests (no LLM / network)
# --------------------------------------------------------------------------- #
def _run_self_tests(repo_root: pathlib.Path) -> int:
    failures = 0

    def check(name: str, cond: bool):
        nonlocal failures
        if not cond:
            failures += 1
        logger.info("SELF-TEST %s: %s", "PASS" if cond else "FAIL", name)

    # 1) URL -> local path + id extraction
    url = "https://github.com/mabablue/I-ADOPT-patterns-playground/blob/main/pattern/VDP11.yaml"
    check(
        "blob URL -> pattern/VDP11.yaml",
        resolve_github_blob_url_to_local_path(url, repo_root) == repo_root / "pattern" / "VDP11.yaml",
    )
    check("extract_pattern_id(VDP11)", extract_pattern_id(url) == "VDP11")
    check("extract_tree_id(DT4.mmd)", extract_tree_id("blah/DT4.mmd") == "DT4")

    repo = PatternRepository(repo_root)

    # 2) load + normalise DT4 (ratio, named edges)
    dt4 = repo.load_decision_tree("DT4")
    check("load DT4 graph has root+nodes", bool(dt4) and dt4.get("root") and bool(dt4.get("nodes")))
    if dt4:
        kinds = {n["kind"] for n in dt4["nodes"].values()}
        check("DT4 has pattern + question nodes", {"pattern", "question"} <= kinds)

    # 3) load a VDP by bare id
    vdp = repo.load_variable_design_pattern("VDP17")
    check("load VDP17 by id has name", bool(vdp) and bool(_vdp_field(vdp, "name")))

    # 4) traverse a tiny mock graph (monkeypatch the batched answer)
    mock_graph = {
        "id": "MOCK",
        "title": "mock",
        "root": "A",
        "nodes": {
            "A": {"kind": "question", "label": "denominator?", "ref": None},
            "C": {"kind": "pattern", "label": "AREAL VDP14", "ref": "VDP14"},
            "E": {"kind": "pattern", "label": "MASSIC VDP06", "ref": "VDP06"},
        },
        "edges": {"A": [{"label": "Area", "to": "C"}, {"label": "Mass", "to": "E"}]},
    }
    mock_repo = PatternRepository(repo_root)
    mock_repo._tree_cache["MOCK"] = mock_graph
    orig = globals()["answer_tree"]
    try:
        globals()["answer_tree"] = lambda *a, **k: {"A": "Area"}
        trav = traverse_component(mock_repo, "MOCK", "x", "m", 0.0, "property")
    finally:
        globals()["answer_tree"] = orig
    check("mock traverse selects VDP14 via 'Area'", trav.get("selected_pattern_id") == "VDP14")

    # 5) rule-based validation + gold signature
    fake_vdps = {"property": {"involved_components": ["property", "object of interest"]}}
    fake_comps = {"property": {"selected_pattern_id": "VDP13"}, "object_of_interest": {"selected_pattern_id": "VDP28"}}
    good = validate_against_selected_pattern(
        {"hasProperty": "temp", "hasObjectOfInterest": "air"}, fake_comps, fake_vdps
    )
    check("rule validation passes when components present", good["pattern_valid"] is True)
    bad = validate_against_selected_pattern({"hasProperty": "", "hasObjectOfInterest": ""}, fake_comps, fake_vdps)
    check("rule validation fails when components missing", bad["pattern_valid"] is False)

    gt_asym = {
        "hasProperty": "concentration",
        "hasObjectOfInterest": {"AsymmetricSystem": "x", "hasSource": "a", "hasTarget": "b"},
        "hasMatrix": "water",
    }
    comps = {
        "property": {"selected_pattern_id": "VDP17"},
        "object_of_interest": {"selected_pattern_id": "VDP27"},
        "matrix": {"selected_pattern_id": "VDP37"},
    }
    psel = pattern_selection_confusion(gt_asym, comps)
    check(
        "gold-signature confusion TP=3 FP=0 FN=0 on aligned case",
        psel["pattern_tp"] == 3 and psel["pattern_fp"] == 0 and psel["pattern_fn"] == 0,
    )

    # 6) degenerate-system repair: bare type word + empty system are blanked
    rep1, notes1 = _repair_degenerate_systems({"hasObjectOfInterest": "AsymmetricSystem"})
    check("bare 'AsymmetricSystem' string is blanked", rep1["hasObjectOfInterest"] == "" and bool(notes1))
    rep2, notes2 = _repair_degenerate_systems(
        {"hasObjectOfInterest": {"AsymmetricSystem": "ratio", "hasSource": "", "hasTarget": ""}}
    )
    check("empty AsymmetricSystem object is blanked", rep2["hasObjectOfInterest"] == "" and bool(notes2))
    rep3, notes3 = _repair_degenerate_systems(
        {"hasObjectOfInterest": {"AsymmetricSystem": "ozone per area", "hasSource": "ozone", "hasTarget": "area"}}
    )
    check("valid AsymmetricSystem is kept", isinstance(rep3["hasObjectOfInterest"], dict) and not notes3)

    logger.info("SELF-TESTS COMPLETE: %d failure(s)", failures)
    return failures


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    global CROSS_ENCODER_MODEL
    parser = argparse.ArgumentParser(
        description="I-ADOPT pattern-guided benchmark (multi-component decision-tree traversal -> VDPs -> decomposition)."
    )
    # pattern-guided specific
    parser.add_argument(
        "--patterns-repo",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("IADOPT_PATTERNS_REPO", str(DEFAULT_PATTERNS_REPO))),
        help="Path to the I-ADOPT patterns repo (env: IADOPT_PATTERNS_REPO).",
    )
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--limit", type=int, default=102, help="Max variables (default 102).")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--no-matrix", action="store_true", help="Skip the MATRIX tree traversal.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run-pattern-selection", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tree-temp", type=float, default=0.0, help="Temperature for decision-tree questions.")

    # baseline-compatible grid options
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--only-model", action="append")
    parser.add_argument("--temps", type=float, nargs="+", default=None)
    parser.add_argument("--shot", type=int, choices=[0, 1, 3, 5], default=5)
    parser.add_argument(
        "--approach", type=str, choices=["none", "naive", "embedding", "cross-encoder"], default="cross-encoder"
    )
    parser.add_argument("--model_name", type=str, default=EMBED_MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=16)

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    repo_root = args.patterns_repo.resolve()
    logger.info("Patterns repo: %s", repo_root)

    if args.self_test:
        sys.exit(1 if _run_self_tests(repo_root) else 0)

    repo = PatternRepository(repo_root)
    models = args.only_model or [MODEL_NAMES[0]]
    temps = args.temps or [0.5]
    include_matrix = not args.no_matrix

    if args.dry_run_pattern_selection:
        rows = dry_run_pattern_selection(
            repo,
            args.data_dir,
            models[0],
            args.tree_temp,
            limit=args.limit,
            start_index=args.start_index,
            include_matrix=include_matrix,
        )
        out = timestamped_output(args.output, "patternGuided_dryrun")
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_excel(out, index=False)
        logger.info("✓ Dry-run pattern selection saved → %s", pathlib.Path(out).resolve())
        return

    if args.approach == "cross-encoder" and CROSS_ENCODER_MODEL is None:
        logger.info("Pre-loading cross encoder model...")
        from sentence_transformers import CrossEncoder

        CROSS_ENCODER_MODEL = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", device="cpu")

    logger.info(
        "RUN | models=%s | temps=%s | tree_temp=%.2f | shot=%d | approach=%s | matrix=%s | limit=%d",
        models,
        temps,
        args.tree_temp,
        args.shot,
        args.approach,
        include_matrix,
        args.limit,
    )

    results = evaluate_pattern_guided(
        repo=repo,
        data_dir=args.data_dir,
        models=models,
        temps=temps,
        tree_temp=args.tree_temp,
        shot=args.shot,
        approach=args.approach,
        link_model_name=args.model_name,
        threshold=args.threshold,
        workers=args.workers,
        limit=args.limit,
        start_index=args.start_index,
        include_matrix=include_matrix,
    )

    out = timestamped_output(args.output, "patternGuided")
    build_excel_pattern(results, out)


# =========================================================================== #
# >>> PATTERN-GUIDED LOGIC ENDS <<<
# =========================================================================== #

if __name__ == "__main__":
    main()
