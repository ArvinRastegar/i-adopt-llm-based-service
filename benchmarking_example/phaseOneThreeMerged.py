#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
# I-ADOPT LLM Benchmark (Phase 1 + Phase 3 Linking & URI Evaluation)
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
from typing import Any, Dict, List, Set, Tuple, Optional

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
import urllib.parse

import requests

try:
    import requests_cache

    _CACHE_SESSION = requests_cache.CachedSession("wikidata_cache", backend="sqlite", expire_after=None)
    _REQUESTS = _CACHE_SESSION
except Exception:
    _REQUESTS = requests  # fallback without cache

# ----- static config -------------------------------------------------------- #
load_dotenv()

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
    "qwen/qwen3-32b",
    # "qwen/qwen3-max",
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

# --------------------------------------------------------------------------- #
# 1 ▪ Logging & external clients
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

PREPROC_LOG_FILE = LOG_DIR / f"iadopt_preprocess_{datetime.now():%Y%m%d_%H%M%S}.log"
_preproc_logger = logging.getLogger("preprocess")
_preproc_logger.setLevel(logging.INFO)
_preproc_logger.addHandler(logging.FileHandler(PREPROC_LOG_FILE, mode="w", encoding="utf-8"))

logging.info(f"Logging to {LOG_FILE.resolve()}")
logging.info(f"Pre-processing log → {PREPROC_LOG_FILE.resolve()}")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

# --------------------------------------------------------------------------- #
# 2 ▪ Prompt helpers
# --------------------------------------------------------------------------- #
_SCHEMA_TEXT = SCHEMA_PATH.read_text(encoding="utf-8").strip()

_SYSTEM_RULES = textwrap.dedent(
    """
    You are an ontology engineer.
    Your task is to output **one** JSON object that satisfies the
    JSON-Schema provided below.

    ▸ Copy *comment* verbatim from the user section.
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
# 3 ▪ LLM invocation with schema validation / retry
# --------------------------------------------------------------------------- #
_JSON_FENCE_RE = re.compile(r"```(?:json)?", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        raise json.JSONDecodeError("No JSON block found", cleaned, 0)
    return json.loads(match.group(0))


def call_model(model: str, prompt: str, temperature: float) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        return resp.choices[0].message.content
    except APIStatusError as e:
        logging.warning(f"{model}: HTTP {e.status_code} – {e.body!s}")
    except (OpenAIError, httpx.HTTPError) as e:
        logging.warning(f"{model}: transport error – {e!r}")
    except json.JSONDecodeError as e:
        logging.warning(f"{model}: invalid JSON payload – {e!r}")
    return ""


def coerce_for_eval(rec: Dict[str, Any], fixes: dict) -> Dict[str, Any]:
    rec = dict(rec)
    fixes["missing_keys"] = [k for k in ONTO_KEYS if k not in rec]
    fixes["extra_keys"] = [k for k in rec.keys() if k not in {"label", "comment", *ONTO_KEYS}]
    for k in fixes["missing_keys"]:
        rec[k] = [] if k == "hasConstraint" else ""
    if isinstance(rec.get("hasProperty"), dict):
        fixes["coerced_property_dict"] = True
        fixes["orig_hasProperty"] = rec["hasProperty"]
        rec["hasProperty"] = rec["hasProperty"].get("label", "")
    else:
        fixes["coerced_property_dict"] = False
    return rec


def call_llm_loose(model: str, prompt: str, orig_label: str, orig_comment: str, temperature: float) -> Dict[str, Any]:
    attempts, data, fixes, raw = 0, {}, {}, ""
    while attempts < 3:
        attempts += 1
        raw = call_model(model, prompt, temperature)
        fixes = {
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
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        m = _JSON_BLOCK_RE.search(cleaned)
        if not m:
            fixes["unparsable_json"] = True
            fixes["raw_llm_output"] = raw
            _preproc_logger.info(json.dumps(fixes, ensure_ascii=False))
            if attempts < 3:
                continue
            return {}
        try:
            data = json.loads(m.group(0))
            break
        except json.JSONDecodeError:
            fixes["unparsable_json"] = True
            fixes["raw_llm_output"] = raw
            _preproc_logger.info(json.dumps(fixes, ensure_ascii=False))
            if attempts < 3:
                continue
            return {}
    fixes["retry_count"] = attempts
    if raw:
        pre = raw.split("{", 1)[0]
        post = raw.rsplit("}", 1)[-1] if "}" in raw else ""
        if pre.strip():
            fixes["non_json_prefix"] = True
            fixes["prefix_text"] = pre.strip()[:200]
        if post.strip():
            fixes["non_json_suffix"] = True
            fixes["suffix_text"] = post.strip()[:200]
    pred_label = data.get("label")
    pred_comment = data.get("comment")
    if pred_label != orig_label:
        fixes["label_overwritten"] = True
        fixes["orig_label"] = orig_label
    if pred_comment != orig_comment:
        fixes["comment_overwritten"] = True
        fixes["orig_comment"] = (orig_comment or "")[:400]
    data["label"] = orig_label
    data["comment"] = orig_comment
    data = coerce_for_eval(data, fixes=fixes)
    _preproc_logger.info("%s", json.dumps(fixes, ensure_ascii=False))
    return data


# --------------------------------------------------------------------------- #
# 4 ▪ Similarity helpers (existing metrics)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def load_crossencoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> CrossEncoder:
    return CrossEncoder(model_name)


def _cosine(a: str, b: str, model_name: str) -> float:
    embedder = load_embedder(model_name)
    emb1 = embedder.encode(a, convert_to_tensor=True)
    emb2 = embedder.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def sim_string(a: str, b: str, close: bool, model_name: str = "all-MiniLM-L6-v2") -> float:
    if not a or not b:
        return 0.0
    norm_a, norm_b = a.lower().strip(), b.lower().strip()
    if norm_a == norm_b:
        return 1.0
    return _cosine(norm_a, norm_b, model_name) if close else 0.0


def sim_asym(a: Dict[str, str], b: Dict[str, str], close: bool) -> float:
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
    lbl_sim = sim_string(a.get("label", ""), b.get("label", ""), close)
    on_a = canonical_on(a.get("on", ""))
    on_b = canonical_on(b.get("on", ""))
    on_sim = sim_string(on_a, on_b, close)
    return (lbl_sim + on_sim) / 2


_ON_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+)$")


def canonical_on(text: str) -> str:
    if not text:
        return ""
    m = _ON_PREFIX_RE.match(text)
    if m and m.group(1) in ONTO_KEYS:
        return m.group(2).strip()
    return text.strip()


# --------------------------------------------------------------------------- #
# 5 ▪ Confusion-matrix helpers (existing metrics)
# --------------------------------------------------------------------------- #
def confusion(gt, pred, close: bool) -> Tuple[int, int, int, int]:
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
        return (0, 0, 0, 1) if not pred else (0, 1, 0, 0)


def confusion_constraints(
    gt_list: List[Dict[str, str]],
    pred_list: List[Dict[str, str]],
    close: bool,
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[float, float, float, float]:
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
        i, j = divmod(int(np.argmax(S)), S.shape[1])
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
                canonical_on(gt_list[i].get("on", "")), canonical_on(pred_list[j].get("on", "")), close, model_name
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


# --------------------------------------------------------------------------- #
# 6 ▪ URI linking to Wikidata (Phase 3)
# --------------------------------------------------------------------------- #
def _qid_from_uri_or_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"(Q\d+)", s)
    return m.group(1) if m else None


def canonicalize_uri_for_compare(uri: Optional[str]) -> Optional[str]:
    """Make http/https equivalent and wiki/entity equivalent by canonicalizing to https://www.wikidata.org/wiki/Qxxxx."""
    if not uri:
        return None
    q = _qid_from_uri_or_text(uri)
    if q:
        return f"https://www.wikidata.org/wiki/{q}"
    # Fallback: normalize scheme & strip trailing slash
    u = uri.strip().replace("http://", "https://")
    return u[:-1] if u.endswith("/") else u


def _to_wiki_url(uri: Optional[str]) -> Optional[str]:
    """Convert any Q-id or wikidata URI to canonical https://www.wikidata.org/wiki/Qxxxx for storage in ...URI."""
    if not uri:
        return None
    q = _qid_from_uri_or_text(uri)
    return f"https://www.wikidata.org/wiki/{q}" if q else canonicalize_uri_for_compare(uri)


def format_queries(query, instruction=None):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


def get_wikidata_entity(
    term: str, approach: str = "naive", context: str = "", model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.0
) -> Optional[str]:
    """
    Returns a Wikidata URI for *term* using the chosen approach.
    Output is canonicalized to https://www.wikidata.org/wiki/Qxxxx for consistency.
    """
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
            qid = search[0]["id"]
            return _to_wiki_url(qid)

        if approach == "embedding":
            embedder = load_embedder(model_name)
            query_vec = embedder.encode(f'Definition of "{term}" in context: "{context}"')
            docs = [f'label: "{s.get("label","")}", description: "{s.get("description","")}"' for s in search]
            doc_vecs = embedder.encode(docs)
            sims = util.cos_sim(query_vec, doc_vecs).cpu().numpy().ravel()
            idx = int(sims.argmax())
            return _to_wiki_url(search[idx]["id"])

        if approach == "cross-encoder":
            model = load_crossencoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
            task = "Given a web search query, retrieve relevant passages that answer the query"
            queries = [f'Definition of "{term}" in context: "{context}"'] * len(search)
            documents = [
                f"label: \"{search_entry['label']}\", description: \"{search_entry['description'] if 'description' in search_entry else ""}\""
                for search_entry in search
            ]
            pairs = [[format_queries(query, task), format_document(doc)] for query, doc in zip(queries, documents)]
            scores = model.predict(pairs)
            # Sort results by score (descending)
            ranked = sorted(zip(search, scores), key=lambda x: x[1], reverse=True)

            # Log neatly: original term, candidate label, score
            logging.info("Cross-encoder ranking | term=%r | context=%r", term, context)
            for s, score in ranked:
                logging.info(
                    "  term=%r | candidate=%r | score=%.4f | id=%s", term, s.get("label"), float(score), s.get("id")
                )

            # Pick top candidate if above threshold
            best_s, best_score = ranked[0]
            if float(best_score) >= float(threshold):
                return _to_wiki_url(best_s["id"])
            return None

        # default to naive
        qid = search[0]["id"]
        return _to_wiki_url(qid)

    except Exception as e:
        logging.warning("Wikidata API error for %r: %r", term, e)
        return None


def enrich_with_uris(
    pred: Dict[str, Any], approach: str = "naive", model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Return a copy of *pred* enriched with ...URI fields (top-level and nested systems).
    """
    if approach == "none":
        # Skip Phase 3 entirely – return Phase 1 output unchanged
        return pred
    out = json.loads(json.dumps(pred))  # deep copy

    def add_uri_field(container: Dict[str, Any], key: str, label_value: Any):
        if isinstance(label_value, str) and label_value.strip():
            uri = get_wikidata_entity(
                label_value,
                approach=approach,
                context=pred.get("label", ""),
                model_name=model_name,
                threshold=threshold,
            )
            if uri:
                container[f"{key}URI"] = _to_wiki_url(uri)

    # Top-level simple keys
    for p, key in [
        ("hasProperty", "hasProperty"),
        ("hasMatrix", "hasMatrix"),
        ("hasObjectOfInterest", "hasObjectOfInterest"),
        ("hasContextObject", "hasContextObject"),
    ]:
        if p in out and isinstance(out[p], str):
            add_uri_field(out, key, out[p])

    # Systems (nested)
    for p in ["hasMatrix", "hasObjectOfInterest", "hasContextObject"]:
        val = out.get(p)
        if isinstance(val, dict):
            # Asymmetric
            if "AsymmetricSystem" in val:
                sys_lbl = val.get("AsymmetricSystem")
                src_lbl = val.get("hasSource")
                tgt_lbl = val.get("hasTarget")
                if sys_lbl:
                    uri = get_wikidata_entity(
                        sys_lbl,
                        approach=approach,
                        context=pred.get("label", ""),
                        model_name=model_name,
                        threshold=threshold,
                    )
                    if uri:
                        val["AsymmetricSystemURI"] = _to_wiki_url(uri)
                if src_lbl:
                    uri = get_wikidata_entity(
                        src_lbl,
                        approach=approach,
                        context=pred.get("label", ""),
                        model_name=model_name,
                        threshold=threshold,
                    )
                    if uri:
                        val["hasSourceURI"] = _to_wiki_url(uri)
                if tgt_lbl:
                    uri = get_wikidata_entity(
                        tgt_lbl,
                        approach=approach,
                        context=pred.get("label", ""),
                        model_name=model_name,
                        threshold=threshold,
                    )
                    if uri:
                        val["hasTargetURI"] = _to_wiki_url(uri)

            # Symmetric
            if "SymmetricSystem" in val:
                sys_lbl = val.get("SymmetricSystem")
                parts = val.get("hasPart", [])
                if sys_lbl:
                    uri = get_wikidata_entity(
                        sys_lbl,
                        approach=approach,
                        context=pred.get("label", ""),
                        model_name=model_name,
                        threshold=threshold,
                    )
                    if uri:
                        val["SymmetricSystemURI"] = _to_wiki_url(uri)
                if isinstance(parts, list) and parts:
                    part_uris: List[Optional[str]] = []
                    for part in parts:
                        if isinstance(part, str) and part.strip():
                            uri = get_wikidata_entity(
                                part,
                                approach=approach,
                                context=pred.get("label", ""),
                                model_name=model_name,
                                threshold=threshold,
                            )
                            part_uris.append(_to_wiki_url(uri) if uri else None)
                        else:
                            part_uris.append(None)
                    if any(part_uris):
                        val["hasPartURIs"] = part_uris
    return out


# --------------------------------------------------------------------------- #
# 7 ▪ URI evaluation helpers
# --------------------------------------------------------------------------- #
def _iter_uri_assertions(gt: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """
    Return list of (path, expected) for all URI-bearing fields present in gt.
    Path uses dotted notation like 'hasPropertyURI' or 'hasMatrix.hasSourceURI'.
    """
    out: List[Tuple[str, Any]] = []
    # top-level
    for key in ["hasPropertyURI", "hasMatrixURI", "hasObjectOfInterestURI", "hasContextObjectURI"]:
        if key in gt and gt[key]:
            out.append((key, gt[key]))
    # nested: hasMatrix / hasObjectOfInterest / hasContextObject
    for root in ["hasMatrix", "hasObjectOfInterest", "hasContextObject"]:
        node = gt.get(root)
        if isinstance(node, dict):
            for k in ["AsymmetricSystemURI", "SymmetricSystemURI", "hasSourceURI", "hasTargetURI"]:
                if k in node and node[k]:
                    out.append((f"{root}.{k}", node[k]))
            if "hasPartURIs" in node and isinstance(node["hasPartURIs"], list):
                out.append((f"{root}.hasPartURIs", node["hasPartURIs"]))
    return out


def _get_pred_uri_at_path(pred_enriched: Dict[str, Any], path: str) -> Any:
    cur: Any = pred_enriched
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def compare_uris(gt: Dict[str, Any], pred_enriched: Dict[str, Any]) -> Tuple[int, int, float, Dict[str, bool]]:
    """
    Compare all URI fields present in GT with predicted enriched URIs.
    Returns (total, correct, acc, per_field_ok).
    """
    assertions = _iter_uri_assertions(gt)
    total = 0
    correct = 0
    per_field_ok: Dict[str, bool] = {}
    for path, expected in assertions:
        total += 1
        pred_val = _get_pred_uri_at_path(pred_enriched, path)
        ok = False
        if isinstance(expected, list):
            # lists must match length and order (exact by QID)
            if isinstance(pred_val, list) and len(pred_val) == len(expected):
                ok = all(
                    canonicalize_uri_for_compare(p) == canonicalize_uri_for_compare(g)
                    for p, g in zip(pred_val, expected)
                )
            else:
                ok = False
        else:
            ok = canonicalize_uri_for_compare(pred_val) == canonicalize_uri_for_compare(expected)
        per_field_ok[path.replace(".", "_")] = bool(ok)
        correct += 1 if ok else 0
    acc = (correct / total) if total else 1.0
    return total, correct, acc, per_field_ok


# --------------------------------------------------------------------------- #
# 8 ▪ Flatten for Jaccard (existing)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 9 ▪ Prompt-deduplication helpers
# --------------------------------------------------------------------------- #
_PRINTED_PROMPTS: set[tuple[int, str]] = set()
_PROMPT_LOCK = Lock()


# --------------------------------------------------------------------------- #
# 10 ▪ Evaluation worker  (returns {"_rows": [...]})
# --------------------------------------------------------------------------- #

# ---- Phase 1 decomposition cache (per model/shot/temp/variable) ----
_DECOMP_CACHE: dict[tuple, Dict[str, Any]] = {}
_DECOMP_LOCK = Lock()


def _run_one(
    model: str,
    gt: Dict[str, Any],
    prompt: str,
    shot: int,
    temperature: float,
    approach: str,
    model_name: str,
    threshold: float,
) -> Dict[str, Any]:

    try:
        # ---------------- Phase 1: call LLM to decompose (labels only) ----------------
        # Use a cache so we do NOT recompute the LLM decomposition when only the
        # linking 'approach' changes.
        cache_key = (model, shot, temperature, gt["label"])
        with _DECOMP_LOCK:
            pred = _DECOMP_CACHE.get(cache_key)

        if pred is None:
            logging.info("LLM cache MISS | model=%s | shot=%d | T=%.2f | var=%r", model, shot, temperature, gt["label"])
            pred = call_llm_loose(model, prompt, gt["label"], gt["comment"], temperature=temperature)
            with _DECOMP_LOCK:
                _DECOMP_CACHE[cache_key] = pred
        else:
            logging.info("LLM cache HIT  | model=%s | shot=%d | T=%.2f | var=%r", model, shot, temperature, gt["label"])

        # Phase 3: link predicted labels to Wikidata and add ...URI fields
        pred_enriched = enrich_with_uris(pred, approach=approach, model_name=model_name, threshold=threshold)

        # ---------- human-readable logs -----------------------------------
        logging.info("MODEL | %-35s | shot=%d | T=%.2f | %s", model, shot, temperature, gt["label"])
        logging.info("GROUND-TRUTH JSON (GT):\n%s", json.dumps(gt, indent=2, ensure_ascii=False))
        logging.info("PREDICTED JSON (labels only):\n%s", json.dumps(pred, indent=2, ensure_ascii=False))
        logging.info("PREDICTED JSON (with URIs):\n%s", json.dumps(pred_enriched, indent=2, ensure_ascii=False))

        rows: list[dict] = []

        # ---------- metrics for 'exact' and 'close' on ONTO_KEYS ----------
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

            # Jaccards (same as before)
            j_both = jaccard(atoms(gt, "both"), atoms(pred, "both"))
            j_concept = jaccard(atoms(gt, "concept"), atoms(pred, "concept"))
            j_text = jaccard(atoms(gt, "text"), atoms(pred, "text"))

            per_key_unwrapped = {
                key + "_" + metric: per_key[key][i]
                for key in per_key
                for i, metric in enumerate(["TP", "FP", "FN", "TN"])
            }

            # ---------- NEW: URI evaluation (exact by QID; http/https equal) ----------
            uris_total, uris_correct, uris_acc, uri_flags = compare_uris(gt, pred_enriched)

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
                    "URIs_total": uris_total,
                    "URIs_correct": uris_correct,
                    "URIs_acc": round(uris_acc, 3),
                    **uri_flags,
                    **per_key_unwrapped,
                }
            )

        return {"_rows": rows}

    except Exception as e:
        logging.error("%s | %s: worker crashed – %r", model, gt["label"], e)
        return {"_rows": []}


# --------------------------------------------------------------------------- #
# 11 ▪ Evaluation loop
# --------------------------------------------------------------------------- #
def evaluate(
    data_dir: pathlib.Path,
    shot_mode: int,
    max_vars: int = 30,
    models: List[str] | None = None,
    temps: List[float] | None = None,
    workers: int = 8,
    approach: str = "naive",
    model_name: str = EMBED_MODEL_NAME,
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:

    temps = temps or TEMPERATURES
    models = models or MODEL_NAMES
    examples = load_examples(shot_mode)
    tasks: list[tuple] = []

    logging.info("LINKING approach=%s | model_name=%s | threshold=%s", approach, model_name, threshold)

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
                tasks.append((model, gt, prompt, shot_mode, temp, approach, model_name, threshold))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, *task) for task in tasks]
        for f in as_completed(futs):
            res = f.result()
            if res and "_rows" in res:
                rows.extend(res["_rows"])
    return rows


# --------------------------------------------------------------------------- #
# 12 ▪ Metric utilities + summaries
# --------------------------------------------------------------------------- #
def prf(tp, fp, fn) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


# --------------------------------------------------------------------------- #
# 13 ▪ CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="I-ADOPT LLM benchmark – shot × temperature (+ Wikidata URI eval)")
    parser.add_argument(
        "--data-dir", type=pathlib.Path, default=DATA_DIR, help="Folder with URI-enriched ground-truth JSON files"
    )
    parser.add_argument(
        "--shot",
        type=int,
        choices=[0, 1, 3, 5],
        default=None,
        help="Prompting mode (0 / 1 / 3 / 5). If omitted, all four modes are executed.",
    )
    parser.add_argument("--max-vars", type=int, default=30, help="Debug: limit number of variables")
    parser.add_argument("--only-model", action="append", help="Debug: restrict to one or more models")
    parser.add_argument("--workers", type=int, default=96, help="Parallel requests")
    parser.add_argument("--temps", type=float, nargs="+", help="Override the default temperature grid")
    parser.add_argument(
        "--approach",
        type=str,
        choices=["none", "naive", "embedding", "cross-encoder"],
        help="Wikidata linking approach",
    )
    parser.add_argument(
        "--model_name", type=str, default=EMBED_MODEL_NAME, help="Sentence Transformer / CrossEncoder model for linking"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold (used by cross-encoder)")
    args = parser.parse_args()

    temps = args.temps or TEMPERATURES
    shots = [args.shot] if args.shot is not None else [0, 1, 3, 5]
    # NEW: if approach not specified → run all three
    approaches = [args.approach] if args.approach else ["none", "naive", "embedding", "cross-encoder"]
    all_rows: list[dict] = []

    for approach in approaches:
        for s in shots:
            rows = evaluate(
                args.data_dir,
                shot_mode=s,
                max_vars=args.max_vars,
                models=args.only_model,
                workers=args.workers,
                approach=approach,
                model_name=args.model_name,
                threshold=args.threshold,
            )
            # Tag each row with the approach used
            for r in rows:
                r["LinkApproach"] = approach
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # per-variable sheet: use exact rows only, highest-F first
    df_exact = df[df["Metric"] == "exact"]
    df_sorted = df_exact.sort_values(by="F", ascending=False)

    # --------- wide summary with exact & close side-by-side ------------------
    def pick(rowset: pd.DataFrame, col: str, metric: str):
        sub = rowset.loc[rowset["Metric"] == metric, col]
        return sub.mean() if not sub.empty else float("nan")

    summary_rows = []
    for (model, shot, temp, approach), grp in df.groupby(["Model", "Shot", "Temp", "LinkApproach"]):

        per_key_metrics = {}
        # aggregate per ontology key
        for key in ONTO_KEYS:
            # exact
            tp_exact = grp.loc[grp["Metric"] == "exact", key + "_TP"].sum()
            fp_exact = grp.loc[grp["Metric"] == "exact", key + "_FP"].sum()
            fn_exact = grp.loc[grp["Metric"] == "exact", key + "_FN"].sum()
            prec_exact, rec_exact, f1_exact = prf(tp_exact, fp_exact, fn_exact)

            # close
            tp_close = grp.loc[grp["Metric"] == "close", key + "_TP"].sum()
            fp_close = grp.loc[grp["Metric"] == "close", key + "_FP"].sum()
            fn_close = grp.loc[grp["Metric"] == "close", key + "_FN"].sum()
            prec_close, rec_close, f1_close = prf(tp_close, fp_close, fn_close)

            per_key_metrics[key + "_F_exact"] = f1_exact
            per_key_metrics[key + "_F_close"] = f1_close
            per_key_metrics[key + "_P_exact"] = prec_exact
            per_key_metrics[key + "_P_close"] = prec_close
            per_key_metrics[key + "_R_exact"] = rec_exact
            per_key_metrics[key + "_R_close"] = rec_close

        # overall averages (just take mean across variables)
        def avg(col: str, metric: str):
            sub = grp.loc[grp["Metric"] == metric, col]
            return sub.mean() if not sub.empty else float("nan")

        summary_rows.append(
            {
                "Model": model,
                "Shot": shot,
                "Temp": temp,
                "Approach": approach,
                "F_exact": avg("F", "exact"),
                "F_close": avg("F", "close"),
                "P_exact": avg("P", "exact"),
                "P_close": avg("P", "close"),
                "R_exact": avg("R", "exact"),
                "R_close": avg("R", "close"),
                "J_both": avg("J_both", "exact"),
                "J_concept": avg("J_concept", "exact"),
                "J_text": avg("J_text", "exact"),
                "URI_acc_mean": grp["URIs_acc"].mean() if "URIs_acc" in grp else float("nan"),
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

    # ---------- Optional: F-exact matrix and ranking (unchanged) ----------
    df_exact = df[df["Metric"] == "exact"].copy()
    df_exact["Prompt"] = df_exact["Shot"].astype(str) + "-shot | " + df_exact["Variable"]
    prompt_matrix = df_exact.pivot_table(index="Prompt", columns="Model", values="F", aggfunc="first").round(3)
    shot_mean = df_exact.pivot_table(index="Shot", columns="Model", values="F", aggfunc="mean").round(3)
    shot_mean.index = shot_mean.index.map(lambda s: f"{int(s)}-shot | MEAN")
    shot_mean = shot_mean.sort_index(key=lambda s: s.str.extract(r"(\d+)").astype(int)[0])
    overall_means = pd.concat([shot_mean, prompt_matrix]).mean(axis=0)
    col_order = overall_means.sort_values(ascending=False).index.tolist()
    shot_mean = shot_mean[col_order]
    prompt_matrix = prompt_matrix[col_order]
    f_matrix = pd.concat([shot_mean, prompt_matrix])
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
        f_matrix.to_excel(wr, sheet_name="F_exact_matrix")
        best_pairs.to_excel(wr, sheet_name="best_pairs", index=False)

    logging.info("✓ F-exact matrix & ranking saved → %s", out_matrix.resolve())

    # ---------- Append a quick preprocess summary ----------
    counter = Counter()
    with PREPROC_LOG_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("===="):
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
