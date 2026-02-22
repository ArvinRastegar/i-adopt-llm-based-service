#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
# I-ADOPT Benchmark – Phase 1 Decomposition + Phase 3 Wikidata Linking
# Refactor to "randomShotsPhaseOne.py" style (recursive loading, prompt files,
# example formatting with definition-only + expected output without URIs,
# atomic logging, Excel outputs in the same style).
# --------------------------------------------------------------------------- #

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple
import textwrap
import urllib.parse

import httpx
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer, CrossEncoder, util

try:
    import requests_cache

    _CACHE_SESSION = requests_cache.CachedSession("wikidata_cache", backend="sqllite", expire_after=None)
    _REQUESTS = _CACHE_SESSION
except Exception:
    _REQUESTS = requests


# --------------------------------------------------------------------------- #
# Static config
# --------------------------------------------------------------------------- #
load_dotenv()

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

DEFAULT_DATA_DIR = pathlib.Path(
    "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/benchmarking_example/data/Json_preferred/test_set"
)

SCHEMA_PATH = SCRIPT_DIR / "data" / "Json_schema.json"
PROMPT_DIR = SCRIPT_DIR / "data" / "prompts"

ONE_SHOT_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "one_shot"
THREE_SHOT_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "three_shot"
FIVE_SHOT_DIR = SCRIPT_DIR / "data" / "Json_preferred" / "five_shot"

LOG_DIR = SCRIPT_DIR / "benchmarking_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"phaseOneThreeMerged{datetime.now():%Y%m%d_%H%M%S}.log"

OUTBOOK_DIR = pathlib.Path("benchmarking_outputs")
OUTBOOK_DIR.mkdir(exist_ok=True)

MODEL_NAMES = [
    "qwen/qwen3-32b",
    "qwen/qwen3.5-397b-a17b",
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

# --------------------------------------------------------------------------- #
# Logging
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

_log_lock = Lock()
_RERANK_LOCK = Lock()
# --------------------------------------------------------------------------- #
# OpenAI client (OpenRouter)
# --------------------------------------------------------------------------- #
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

# --------------------------------------------------------------------------- #
# Prompt helpers (programmatic, like randomShotsPhaseOne.py)
# --------------------------------------------------------------------------- #
_SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8").strip()

_EXAMPLE_HDR = "\n\n### Examples (valid against the same schema)\n"
_USER_HDR = "\n\n### Variable's definition to decompose\n"
_EXPECTED_HDR = "\n\n### Expected output\n*(only the JSON object)*"


@lru_cache(maxsize=None)
def list_prompt_versions() -> List[str]:
    if not PROMPT_DIR.exists():
        logging.warning("PROMPT_DIR %s does not exist", PROMPT_DIR)
        return []
    return sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))


@lru_cache(maxsize=None)
def load_prompt_instructions(prompt_version: str) -> str:
    available = list_prompt_versions()
    if not available:
        raise RuntimeError(f"No prompt templates found in {PROMPT_DIR}")

    if not prompt_version:
        prompt_version = available[0]

    if prompt_version not in available:
        logging.warning("Prompt version '%s' not found. Falling back to '%s'.", prompt_version, available[0])
        prompt_version = available[0]

    return (PROMPT_DIR / f"{prompt_version}.txt").read_text(encoding="utf-8").strip()


def strip_all_uri_fields(obj: Any) -> Any:
    """
    Remove ANY dict key containing 'URI' (so ...URI and ...URIs) recursively.
    This is used ONLY for examples inside the prompt, so the model doesn't think
    it must output URIs.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if "URI" in k:
                continue
            if k.startswith("__"):  # defensive: remove path metadata if present
                continue
            out[k] = strip_all_uri_fields(v)
        return out
    if isinstance(obj, list):
        return [strip_all_uri_fields(x) for x in obj]
    return obj


def format_example_block(ex: Dict[str, Any], idx: int) -> str:
    """
    Example format:
      Variable's definition to decompose: <definition only>
      Expected output:
      <json without URI fields>
    """
    definition = ex.get("definition") or ex.get("comment") or ""
    ex_no_uris = strip_all_uri_fields(ex)

    # Keep expected output as JSON object (no markdown fences)
    return (
        f"\n\n#### Example {idx}\n"
        f"Variable's definition to decompose: {definition}\n\n"
        f"Expected output:\n{json.dumps(ex_no_uris, indent=2, ensure_ascii=False)}"
    )


def build_prompt(definition: str, examples: List[Dict[str, Any]] | None, prompt_version: str) -> str:
    examples = examples or []
    instructions = load_prompt_instructions(prompt_version)

    ex_block = ""
    if examples:
        blocks = [format_example_block(ex, i + 1) for i, ex in enumerate(examples)]
        ex_block = _EXAMPLE_HDR + "".join(blocks)

    return (
        f"{instructions}\n\n"
        f"### JSON-Schema\n{_SCHEMA_TEXT}\n"
        f"{ex_block}"
        f"{_USER_HDR}Variable's definition to decompose: {definition}"
        f"{_EXPECTED_HDR}"
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
# LLM invocation (robust) + coercion
# --------------------------------------------------------------------------- #
_JSON_FENCE_RE = re.compile(r"```(?:json)?", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_model(model: str, prompt: str, temperature: float) -> str:
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
                logging.warning("%s: HTML error response on attempt %d", model, attempt)
                continue
            if not stripped:
                logging.warning("%s: empty response on attempt %d", model, attempt)
                continue
            return text

        except APIStatusError as e:
            logging.warning(
                "%s: APIStatusError attempt %d – %s – %s", model, attempt, e.status_code, getattr(e, "body", "")
            )
        except (OpenAIError, httpx.HTTPError) as e:
            logging.warning("%s: transport error attempt %d – %r", model, attempt, e)
        except Exception as e:
            logging.warning("%s: unexpected error attempt %d – %r", model, attempt, e)

    logging.error("%s: failed after 3 attempts", model)
    return ""


def coerce_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    pred = dict(pred or {})
    for k in ONTO_KEYS:
        if k not in pred or pred[k] is None:
            pred[k] = [] if k == "hasConstraint" else ""
        elif k == "hasConstraint" and not isinstance(pred[k], list):
            pred[k] = []
    # Some models sometimes return dict for hasProperty; normalize it.
    if isinstance(pred.get("hasProperty"), dict):
        pred["hasProperty"] = pred["hasProperty"].get("label", "") or ""
    return pred


def call_llm_loose(model: str, prompt: str, gt_label: str, definition: str, temperature: float) -> Dict[str, Any]:
    for attempt in range(1, 4):
        raw = call_model(model, prompt, temperature)
        if not raw.strip():
            continue

        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            logging.warning("%s: no JSON block found on attempt %d", model, attempt)
            continue

        try:
            data = json.loads(m.group(0))
        except Exception as e:
            logging.warning("%s: JSON decode failure on attempt %d – %r", model, attempt, e)
            continue

        # Force label + definition (definition-only prompting, but evaluation needs stable label)
        data["label"] = gt_label
        data["definition"] = definition

        return coerce_prediction(data)

    logging.error("%s: could not extract JSON after 3 attempts", model)
    return {}


# --------------------------------------------------------------------------- #
# Similarity & confusion helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


# @lru_cache(maxsize=2)
# def load_crossencoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> CrossEncoder:
#     return CrossEncoder(model_name)


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
# Phase 3: Wikidata linking + URI evaluation
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
            logging.warning("Wikidata API HTTP %s for %r", resp.status_code, term)
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
        logging.warning("Wikidata API error for %r: %r", term, e)
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

    # top-level
    for p in ["hasProperty", "hasMatrix", "hasObjectOfInterest", "hasContextObject"]:
        if p in out and isinstance(out[p], str):
            add_uri_field(out, p, out[p])

    # nested systems
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


# --------------------------------------------------------------------------- #
# Worker (atomic logging) + evaluation loop (recursive)
# --------------------------------------------------------------------------- #
def _run_one(
    model: str,
    temperature: float,
    prompt_version: str,
    approach: str,
    link_model_name: str,
    threshold: float,
    gt: Dict[str, Any],
    gt_path: str,
    prompt: str,
    shot: int,
) -> Dict[str, Any]:

    logs: List[str] = []
    logs.append(
        "MODEL | {model} | shot={shot} | T={temp:.2f} | prompt={pv} | approach={ap} | link_model={lm} | thr={thr:.2f} | var={var} | path={path}".format(
            model=model,
            shot=shot,
            temp=temperature,
            pv=prompt_version,
            ap=approach,
            lm=link_model_name,
            thr=threshold,
            var=gt.get("label", ""),
            path=gt_path,
        )
    )

    logs.append(f"PROMPT:\n{prompt}")
    logs.append("GROUND-TRUTH JSON (GT, as loaded):\n" + json.dumps(gt, indent=2, ensure_ascii=False))

    definition = gt.get("definition") or gt.get("comment") or ""
    pred = call_llm_loose(model, prompt, gt.get("label", ""), definition, temperature=temperature)
    pred_enriched = enrich_with_uris(pred, approach=approach, model_name=link_model_name, threshold=threshold)

    logs.append("PREDICTED JSON (labels only):\n" + json.dumps(pred, indent=2, ensure_ascii=False))
    logs.append("PREDICTED JSON (with URIs):\n" + json.dumps(pred_enriched, indent=2, ensure_ascii=False))

    confusion_data = compute_confusion_for_pair(gt, pred)
    exact = confusion_data["exact_totals"]
    close = confusion_data["close_totals"]

    # Explicit constraint result (your requested “result of the constraints”)
    # c_tp_e, c_fp_e, c_fn_e, c_tn_e = confusion_data["per_key_exact"]["hasConstraint"]
    # c_tp_c, c_fp_c, c_fn_c, c_tn_c = confusion_data["per_key_close"]["hasConstraint"]

    logs.append(
        "EXACT CONFUSION | TP={tp:.3f} FP={fp:.3f} FN={fn:.3f} TN={tn:.3f} | per-key={pk}".format(
            tp=exact["tp"], fp=exact["fp"], fn=exact["fn"], tn=exact["tn"], pk=confusion_data["per_key_exact"]
        )
    )
    logs.append(
        "CLOSE CONFUSION | TP={tp:.3f} FP={fp:.3f} FN={fn:.3f} TN={tn:.3f} | per-key={pk}".format(
            tp=close["tp"], fp=close["fp"], fn=close["fn"], tn=close["tn"], pk=confusion_data["per_key_close"]
        )
    )

    # logs.append(
    #     "CONSTRAINT RESULT | EXACT: TP={tp:.3f} FP={fp:.3f} FN={fn:.3f} TN={tn:.3f} | CLOSE: TP={tp2:.3f} FP={fp2:.3f} FN={fn2:.3f} TN={tn2:.3f}".format(
    #         tp=c_tp_e, fp=c_fp_e, fn=c_fn_e, tn=c_tn_e, tp2=c_tp_c, fp2=c_fp_c, fn2=c_fn_c, tn2=c_tn_c
    #     )
    # )

    # Jaccards (like original script)
    j_both = jaccard(atoms(gt, "both"), atoms(pred, "both"))
    j_concept = jaccard(atoms(gt, "concept"), atoms(pred, "concept"))
    j_text = jaccard(atoms(gt, "text"), atoms(pred, "text"))

    # URI evaluation
    uris_total, uris_correct, uris_acc, uris_coverage, uris_predicted, uri_flags = compare_uris(gt, pred_enriched)
    logs.append(
        "URI EVAL | total={t} correct={c} acc={a:.3f} coverage={cov:.3f} predicted_non_null={pn}".format(
            t=uris_total, c=uris_correct, a=uris_acc, cov=uris_coverage, pn=uris_predicted
        )
    )

    with _log_lock:
        logging.info("\n" + "\n".join(logs) + "\n" + ("─" * 120))

    # Return “randomShotsPhaseOne-like” result package
    return {
        "variable": gt.get("label", ""),
        "path": gt_path,
        "model": model,
        "temperature": temperature,
        "prompt_version": prompt_version,
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
    }


def load_gt_files_recursive(data_dir: pathlib.Path, max_vars: int) -> List[Tuple[pathlib.Path, Dict[str, Any]]]:
    paths = sorted(data_dir.rglob("*.json"))
    if max_vars:
        paths = paths[:max_vars]

    out = []
    for p in paths:
        obj = json.load(open(p, "r", encoding="utf-8"))
        out.append((p, obj))
    return out


def evaluate(
    data_dir: pathlib.Path,
    shot_mode: int,
    prompt_version: str,
    approach: str,
    models: List[str],
    temps: List[float],
    workers: int,
    max_vars: int,
    link_model_name: str,
    threshold: float,
) -> List[Dict[str, Any]]:

    examples = load_examples(shot_mode)

    # For skipping (if the test set accidentally contains an example variable)
    example_labels = {ex.get("label") for ex in examples if ex.get("label")}

    gt_items = load_gt_files_recursive(data_dir, max_vars=max_vars)

    tasks = []
    for p, gt in gt_items:
        if gt.get("label") in example_labels:
            logging.info("Skip %s (in-prompt example)", gt.get("label"))
            continue

        definition = gt.get("definition") or gt.get("comment") or ""
        prompt = build_prompt(definition, examples, prompt_version)

        # keep a stable relative path for logging/excel
        try:
            rel_path = str(p.relative_to(data_dir.parent))  # e.g. Json_preferred/test_set/sub/x.json
        except Exception:
            rel_path = str(p)

        for model in models:
            for temp in temps:
                tasks.append(
                    (
                        model,
                        temp,
                        prompt_version,
                        approach,
                        link_model_name,
                        threshold,
                        gt,
                        rel_path,
                        prompt,
                        shot_mode,
                    )
                )

    results: List[Dict[str, Any]] = []

    if approach == "cross-encoder":
        logging.info("Cross-encoder detected → using reduced threading.")
        max_workers = min(os.cpu_count(), workers)
    else:
        max_workers = workers

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_run_one, *task) for task in tasks]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                logging.exception("Worker failed but run continues")
                continue
            if r:
                results.append(r)

    return results


# --------------------------------------------------------------------------- #
# Excel outputs (same style as randomShotsPhaseOne.py)
# --------------------------------------------------------------------------- #
def compute_summary_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
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

        # aggregate confusion totals
        exact = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        close = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

        per_key = {
            k: {
                "exact": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
                "close": {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
            }
            for k in ONTO_KEYS
        }

        j_both = []
        j_concept = []
        j_text = []
        uri_acc = []
        uri_cov = []
        uri_pred = []

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

        # per-key P/R/F (exact + close)
        for k in ONTO_KEYS:
            for tag in ("exact", "close"):
                tp = per_key[k][tag]["tp"]
                fp = per_key[k][tag]["fp"]
                fn = per_key[k][tag]["fn"]
                p, r_, f = prf(tp, fp, fn)
                suf = "exact" if tag == "exact" else "close"
                out[f"{k}_P_{suf}"] = round(p, 3)
                out[f"{k}_R_{suf}"] = round(r_, 3)
                out[f"{k}_F_{suf}"] = round(f, 3)

        rows.append(out)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["F_exact", "URI_acc_mean"], ascending=[False, False])
    return df


def build_excel(results: List[Dict[str, Any]]) -> pathlib.Path:
    if not results:
        raise RuntimeError("No results to write.")

    out_xlsx = OUTBOOK_DIR / f"phaseOneThreeMerged_{datetime.now():%Y%m%d_%H%M%S}.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
        # 1) ONE SHEET PER ONTO_KEY (same as randomShotsPhaseOne)
        for key in ONTO_KEYS:
            rows = []
            for r in results:
                gt = r["ground_truth_json"]
                pred = r["predicted_json"]

                # Make concept sheets readable (remove URI fields from display only)
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
                        "ground_truth": json.dumps(gt_val, ensure_ascii=False, indent=2),
                        "predicted": json.dumps(pred_val, ensure_ascii=False, indent=2),
                    }
                )

            pd.DataFrame(rows).to_excel(wr, sheet_name=f"{key} concepts"[:31], index=False)

        # 2) LLM outputs (same sheet name as randomShotsPhaseOne)
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
                    # NEW: include the actual prompt text used
                    "prompt": r["prompt"],
                    # URI eval summary
                    "uris_total": r["uris_total"],
                    "uris_correct": r["uris_correct"],
                    "uris_acc": round(r["uris_acc"], 3),
                    "uris_coverage": round(r["uris_coverage"], 3),
                    "uris_predicted": r["uris_predicted"],
                    # Full JSONs (GT includes URIs, predicted includes URIs after enrichment)
                    "ground_truth_json": json.dumps(r["ground_truth_json"], ensure_ascii=False, indent=2),
                    "predicted_json": json.dumps(r["predicted_json"], ensure_ascii=False, indent=2),
                    "predicted_json_with_uris": json.dumps(r["predicted_json_with_uris"], ensure_ascii=False, indent=2),
                }
            )

        pd.DataFrame(json_rows).to_excel(wr, sheet_name="LLM outputs", index=False)

        # 3) Summary (same sheet name as randomShotsPhaseOne)
        summary = compute_summary_metrics(results)
        summary.to_excel(wr, sheet_name="Summary", index=False)

    logging.info("✓ Results saved → %s", out_xlsx.resolve())
    return out_xlsx


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="I-ADOPT benchmark (Phase 1 + Phase 3) – refactored output style")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-vars", type=int, default=0, help="0 = all")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--only-model", action="append")
    parser.add_argument("--temps", type=float, nargs="+")
    parser.add_argument("--shot", type=int, choices=[0, 1, 3, 5], default=None)
    parser.add_argument("--prompt-version", type=str, nargs="+", default=None)
    parser.add_argument("--approach", type=str, choices=["none", "naive", "embedding", "cross-encoder"], default=None)
    parser.add_argument("--model_name", type=str, default=EMBED_MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # prompt versions (programmatic)
    prompt_versions = args.prompt_version if args.prompt_version else (list_prompt_versions() or [])
    if not prompt_versions:
        raise RuntimeError(f"No prompt templates found in {PROMPT_DIR}")

    shots = [args.shot] if args.shot is not None else [0, 1, 3, 5]
    approaches = [args.approach] if args.approach else ["none", "naive", "embedding", "cross-encoder"]

    if "cross-encoder" in approaches:
        logging.info("Pre-loading cross encoder model...")
        global CROSS_ENCODER_MODEL
        CROSS_ENCODER_MODEL = CrossEncoder(
            "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
            device="cpu",
            # max_length=256,
        )

    models = args.only_model or MODEL_NAMES
    temps = args.temps or TEMPERATURES
    max_vars = args.max_vars if args.max_vars and args.max_vars > 0 else 0

    all_results: List[Dict[str, Any]] = []
    for pv in prompt_versions:
        for approach in approaches:
            for shot in shots:
                logging.info("=== RUN | prompt=%s | approach=%s | shot=%d ===", pv, approach, shot)
                res = evaluate(
                    data_dir=args.data_dir,
                    shot_mode=shot,
                    prompt_version=pv,
                    approach=approach,
                    models=models,
                    temps=temps,
                    workers=args.workers,
                    max_vars=max_vars,
                    link_model_name=args.model_name,
                    threshold=args.threshold,
                )
                all_results.extend(res)

    build_excel(all_results)


if __name__ == "__main__":
    main()
