#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import httpx
from openai import OpenAI, OpenAIError, APIStatusError
from dotenv import load_dotenv
import logging
from datetime import datetime

# Optional deps for Wikidata enrichment
try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# sentence_transformers is needed only if you use embedding/cross-encoder
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
except Exception:  # pragma: no cover
    SentenceTransformer = None
    CrossEncoder = None
    util = None


def make_run_id(script_path: str) -> str:
    script_name = pathlib.Path(script_path).stem
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{script_name}_{ts}"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("iadopt-workflow")

load_dotenv()

ONTO_KEYS = [
    "hasStatisticalModifier",
    "hasProperty",
    "hasObjectOfInterest",
    "hasMatrix",
    "hasContextObject",
    "hasConstraint",
]

_JSON_FENCE_RE = re.compile(r"```(?:json)?", re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# ----------------------------
# Utilities: file IO
# ----------------------------
def _read_text(path: str | pathlib.Path) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8").strip()


def _write_text(path: str | pathlib.Path, text: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _write_json(path: str | pathlib.Path, obj: Any) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_run_log(
    out_dir: str,
    run_id: str,
    cfg: Dict[str, Any],
    ctx: Dict[str, Any],
):
    path = pathlib.Path(out_dir) / f"{run_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    # --- METADATA ---
    sections.append("# RUN METADATA")
    sections.append(f"run_id: {run_id}")
    sections.append(f"script: {pathlib.Path(__file__).name}")

    # --- PARAMETERS ---
    sections.append("\n# PARAMETERS (config used)")
    sections.append(json.dumps(cfg, indent=2, ensure_ascii=False))

    # --- PROMPT ---
    if ctx.get("prompt"):
        sections.append("\n# PROMPT")
        sections.append(ctx["prompt"])

    # --- GROUND TRUTH ---
    if ctx.get("gt"):
        sections.append("\n# GROUND TRUTH")
        sections.append(json.dumps(ctx["gt"], indent=2, ensure_ascii=False))

    # --- PREDICTED JSON ---
    pred = ctx.get("pred_enriched") or ctx.get("pred_json")
    if pred:
        sections.append("\n# PREDICTED JSON")
        sections.append(json.dumps(pred, indent=2, ensure_ascii=False))

    path.write_text("\n\n".join(sections), encoding="utf-8")

    log.info("📝 Run log written to %s", path)


# ----------------------------
# Prompt loading (from data/prompts/*.txt)
# ----------------------------
@lru_cache(maxsize=None)
def list_prompt_versions(prompts_dir: str) -> List[str]:
    d = pathlib.Path(prompts_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.txt"))


@lru_cache(maxsize=None)
def load_prompt_instructions(prompts_dir: str, prompt_version: str) -> str:
    available = list_prompt_versions(prompts_dir)
    if not available:
        raise RuntimeError(f"No prompt templates found in {prompts_dir}")

    if not prompt_version:
        prompt_version = "strict_minimal"
    if prompt_version not in available:
        # fallback
        prompt_version = "strict_minimal"
        if prompt_version not in available:
            # fallback to first available
            prompt_version = available[0]

    path = pathlib.Path(prompts_dir) / f"{prompt_version}.txt"
    return _read_text(path)


def load_examples(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    shot = int(cfg["prompting"]["shot"])
    override = cfg["prompting"].get("examples_paths_override") or []
    if override:
        ex = []
        for p in override:
            ex.append(json.loads(_read_text(p)))
        return ex[:shot] if shot else []

    if shot == 0:
        return []

    examples_dir_map = cfg["prompting"].get("examples_dir_map") or {}
    folder = examples_dir_map.get(str(shot))
    if not folder:
        raise ValueError(
            f"No examples_dir_map entry for shot={shot}. "
            f"Provide prompting.examples_dir_map['{shot}'] or examples_paths_override."
        )

    d = pathlib.Path(folder)
    paths = sorted(d.glob("*.json"))
    if len(paths) < shot:
        raise ValueError(f"Not enough examples in {folder}: need {shot}, found {len(paths)}")
    return [json.loads(_read_text(p)) for p in paths[:shot]]


def build_prompt(definition: str, examples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> str:
    schema_text = _read_text(cfg["prompting"]["schema_path"])
    instructions = load_prompt_instructions(
        cfg["prompting"]["prompts_dir"],
        cfg["prompting"]["prompt_version"],
    )

    ex_block = ""
    if examples:
        ex_block = "\n\n### Examples (valid against the same schema)\n"
        ex_block += "\n\n".join(json.dumps(e, indent=2, ensure_ascii=False) for e in examples)

    return (
        f"{instructions}\n\n"
        f"### JSON-Schema\n{schema_text}\n"
        f"{ex_block}"
        f"\n\n### Variable's definition to decompose\ndefinition: {definition}"
        f"\n\n### Expected output\n*(only the JSON object)*"
    )


# ----------------------------
# LLM call + JSON extraction
# ----------------------------
def extract_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    m = _JSON_BLOCK_RE.search(cleaned)
    if not m:
        raise json.JSONDecodeError("No JSON block found", cleaned, 0)
    return json.loads(m.group(0))


def call_model(client: OpenAI, model: str, prompt: str, temperature: float, timeout_sec: int, max_retries: int) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_sec,
            )
            text = resp.choices[0].message.content or ""
            stripped = text.strip()
            if stripped.startswith("<!DOCTYPE html") or stripped.startswith("<html"):
                continue
            if not stripped:
                continue
            return text
        except APIStatusError:
            continue
        except (OpenAIError, httpx.HTTPError):
            continue
    return ""


def normalize_prediction(pred: Dict[str, Any], definition: str) -> Dict[str, Any]:
    # keep only expected keys + allow label/comment if present
    out = dict(pred)

    # Attach definition (you said this is all you need)
    out["definition"] = definition

    # Ensure ONTO keys exist and types are sane
    for k in ONTO_KEYS:
        if k not in out or out[k] is None:
            out[k] = [] if k == "hasConstraint" else ""
        if k == "hasConstraint":
            if not isinstance(out[k], list):
                out[k] = []
        else:
            if isinstance(out[k], (dict, list)):
                # force to string for now; later you might support systems here
                out[k] = str(out[k])

    return out


def llm_decompose_step(ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    log.info("   🤖 Starting LLM decomposition")

    client = OpenAI(
        base_url=cfg["llm"]["base_url"],
        api_key=os.getenv(cfg["llm"]["api_key_env"]),
    )

    if not client.api_key:
        raise RuntimeError(f"Missing API key env var: {cfg['llm']['api_key_env']}")

    definition = ctx["definition"]
    log.info("   📝 Definition length: %d chars", len(definition))

    examples = load_examples(cfg)
    log.info("   📚 Loaded %d in-context examples", len(examples))

    prompt = build_prompt(definition, examples, cfg)
    log.info("   🧾 Prompt length: %d chars", len(prompt))

    log.info(
        "   🚀 Calling model=%s temp=%s",
        cfg["llm"]["model"],
        cfg["llm"]["temperature"],
    )

    raw = call_model(
        client=client,
        model=cfg["llm"]["model"],
        prompt=prompt,
        temperature=float(cfg["llm"]["temperature"]),
        timeout_sec=int(cfg["llm"]["timeout_sec"]),
        max_retries=int(cfg["llm"]["max_retries"]),
    )

    if not raw:
        log.warning("   ⚠️ LLM returned EMPTY output")
        ctx["llm_raw"] = ""
        ctx["pred_json"] = {}
        ctx["prompt"] = prompt
        return ctx

    log.info("   📥 LLM raw output length: %d chars", len(raw))

    try:
        pred = extract_json_from_text(raw)
        log.info("   ✅ JSON successfully extracted")
    except Exception as e:
        log.error("   ❌ Failed to extract JSON: %s", e)
        pred = {}

    if cfg.get("postprocess", {}).get("normalize_output", False) and pred:
        log.info("   🧹 Normalizing JSON output")
        pred = normalize_prediction(pred, definition)
    elif pred:
        pred["definition"] = definition

    ctx["prompt"] = prompt
    ctx["llm_raw"] = raw
    ctx["pred_json"] = pred

    log.info("   📦 pred_json keys: %s", sorted(pred.keys()))
    return ctx


# ----------------------------
# Input step: definition from config or GT json
# ----------------------------
def extract_definition_step(ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    log.info("   🔍 Extracting definition")

    definition = cfg["input"].get("definition")
    if definition and str(definition).strip():
        ctx["definition"] = str(definition).strip()
        log.info("   📌 Using definition from config (%d chars)", len(ctx["definition"]))
        return ctx

    gt_path = cfg["input"].get("gt_path")
    if not gt_path:
        raise ValueError("Provide either input.definition or input.gt_path in config.")

    log.info("   📂 Loading GT file: %s", gt_path)
    gt = json.loads(_read_text(gt_path))

    definition = gt.get("definition") or gt.get("comment")
    if not definition:
        raise ValueError(f"GT file {gt_path} has no 'definition' or 'comment' field.")

    ctx["gt"] = gt
    ctx["definition"] = definition
    log.info("   📌 Extracted definition from GT (%d chars)", len(definition))
    return ctx


# ----------------------------
# Placeholder step: JSON Schema validation (to implement later)
# ----------------------------
def validate_json_schema_step(ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Intentionally empty shell:
    # Later: validate ctx["pred_json"] against cfg["prompting"]["schema_path"] using jsonschema.
    ctx.setdefault("notes", []).append("validate_json_schema_step is a placeholder (not implemented yet).")
    return ctx


# ----------------------------
# Wikidata enrichment (all options, including none)
# ----------------------------
def _qid_from_uri_or_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"(Q\d+)", s)
    return m.group(1) if m else None


def _to_wiki_url(uri_or_qid: Optional[str]) -> Optional[str]:
    if not uri_or_qid:
        return None
    q = _qid_from_uri_or_text(uri_or_qid)
    if q:
        return f"https://www.wikidata.org/wiki/{q}"
    u = uri_or_qid.strip().replace("http://", "https://")
    return u[:-1] if u.endswith("/") else u


def _get_requests_session(use_cache: bool):
    if requests is None:
        raise RuntimeError("requests is required for wikidata enrichment")
    if not use_cache:
        return requests
    try:
        import requests_cache

        return requests_cache.CachedSession("wikidata_cache", backend="sqlite", expire_after=None)
    except Exception:
        return requests


@lru_cache(maxsize=4)
def _load_embedder(model_name: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError("sentence_transformers is required for embedding approach")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _load_crossencoder(model_name: str) -> CrossEncoder:
    if CrossEncoder is None:
        raise RuntimeError("sentence_transformers is required for cross-encoder approach")
    return CrossEncoder(model_name)


def _format_queries(query: str, instruction: Optional[str] = None) -> str:
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def _format_document(document: str) -> str:
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"


def get_wikidata_entity(
    term: str,
    approach: str,
    context: str,
    cfg: Dict[str, Any],
) -> Optional[str]:
    if not term or not term.strip():
        return None

    session = _get_requests_session(bool(cfg["wikidata"].get("use_requests_cache", True)))
    encoded = term.replace(" ", "+")
    headers = {"User-Agent": "IADOPT-Linker/1.0 (+workflow)"}

    resp = session.get(
        f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={encoded}&language=en&format=json",
        headers=headers,
        timeout=20,
    )
    if resp.status_code != 200:
        return None
    search = resp.json().get("search", [])
    if not search:
        return None

    # none
    if approach == "none":
        return None

    # naive: first hit
    if approach == "naive":
        return _to_wiki_url(search[0]["id"])

    # embedding: cosine over label+description
    if approach == "embedding":
        embed_model_name = cfg["wikidata"].get("embed_model_name", "all-MiniLM-L6-v2")
        embedder = _load_embedder(embed_model_name)
        query_vec = embedder.encode(f'Definition of "{term}" in context: "{context}"')
        docs = [f'label: "{s.get("label","")}", description: "{s.get("description","")}"' for s in search]
        doc_vecs = embedder.encode(docs)
        sims = util.cos_sim(query_vec, doc_vecs).cpu().numpy().ravel()
        idx = int(sims.argmax())
        return _to_wiki_url(search[idx]["id"])

    # cross-encoder rerank
    if approach == "cross-encoder":
        ce_name = cfg["wikidata"].get("cross_encoder_model_name", "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
        threshold = float(cfg["wikidata"].get("threshold", 0.0))
        model = _load_crossencoder(ce_name)

        task = "Given a web search query, retrieve relevant passages that answer the query"
        q = f'Definition of "{term}" in context: "{context}"'
        documents = [f'label: "{s.get("label","")}", description: "{s.get("description","")}"' for s in search]
        pairs = [[_format_queries(q, task), _format_document(doc)] for doc in documents]
        scores = model.predict(pairs)

        ranked = sorted(zip(search, scores), key=lambda x: float(x[1]), reverse=True)
        best_s, best_score = ranked[0]
        if float(best_score) >= threshold:
            return _to_wiki_url(best_s["id"])
        return None

    # fallback
    return _to_wiki_url(search[0]["id"])


def enrich_with_uris(pred: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    approach = cfg["wikidata"]["approach"]
    if approach == "none":
        return pred

    out = json.loads(json.dumps(pred))  # deep copy
    context = pred.get("label", "") or pred.get("definition", "") or ""

    def add_uri_field(container: Dict[str, Any], key: str, label_value: Any):
        if isinstance(label_value, str) and label_value.strip():
            uri = get_wikidata_entity(label_value, approach=approach, context=context, cfg=cfg)
            if uri:
                container[f"{key}URI"] = _to_wiki_url(uri)

    for key in ["hasProperty", "hasMatrix", "hasObjectOfInterest", "hasContextObject"]:
        if key in out and isinstance(out[key], str):
            add_uri_field(out, key, out[key])

    # (If later you output nested systems, extend here. Kept minimal for now.)
    return out


def wikidata_enrich_step(ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    approach = cfg["wikidata"]["approach"]
    log.info("   🌍 Wikidata enrichment approach: %s", approach)

    pred = ctx.get("pred_json") or {}
    if not pred:
        log.warning("   ⚠️ No pred_json → skipping enrichment")
        ctx["pred_enriched"] = {}
        return ctx

    ctx["pred_enriched"] = enrich_with_uris(pred, cfg)
    log.info("   🔗 Wikidata enrichment finished")
    return ctx


# ----------------------------
# Placeholder step: JSON -> RDF export (to implement later)
# ----------------------------
def export_rdf_step(ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Intentionally empty shell:
    # Later: use rdflib to map ctx["pred_enriched"] (or ctx["pred_json"]) into triples and serialize Turtle.
    ctx.setdefault("notes", []).append("export_rdf_step is a placeholder (not implemented yet).")
    return ctx


# ----------------------------
# Workflow runner
# ----------------------------
StepFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

STEP_REGISTRY: Dict[str, StepFn] = {
    "extract_definition": extract_definition_step,
    "llm_decompose": llm_decompose_step,
    "validate_json_schema": validate_json_schema_step,
    "wikidata_enrich": wikidata_enrich_step,
    "export_rdf": export_rdf_step,
}


def run_workflow(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    steps = cfg.get("workflow", {}).get("steps") or []

    log.info("🚀 Starting workflow")
    log.info("Configured steps: %s", " → ".join(steps))

    for i, name in enumerate(steps, start=1):
        log.info("▶️  Step %d/%d: %s", i, len(steps), name)

        step_fn = STEP_REGISTRY.get(name)
        if not step_fn:
            raise ValueError(f"Unknown step: {name}")

        try:
            ctx = step_fn(ctx, cfg)
        except Exception:
            log.exception("❌ Step '%s' failed", name)
            raise

        log.info("✅ Finished step: %s", name)
        log.info("   Context keys now: %s", sorted(ctx.keys()))

    log.info("🏁 Workflow finished")
    return ctx


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to workflow config JSON")
    args = ap.parse_args()

    cfg = json.loads(_read_text(args.config))
    ctx = run_workflow(cfg)

    out_cfg = cfg.get("output", {})
    if out_cfg.get("write_prompt_path") and ctx.get("prompt"):
        _write_text(out_cfg["write_prompt_path"], ctx["prompt"])
    if out_cfg.get("write_raw_llm_path") and ctx.get("llm_raw") is not None:
        _write_text(out_cfg["write_raw_llm_path"], ctx["llm_raw"])
    if out_cfg.get("write_json_path"):
        # prefer enriched if available
        payload = ctx.get("pred_enriched") or ctx.get("pred_json") or {}
        _write_json(out_cfg["write_json_path"], payload)

    # print final JSON to stdout
    payload = ctx.get("pred_enriched") or ctx.get("pred_json") or {}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    # write full run log
    run_id = make_run_id(__file__)
    out_dir = cfg.get("output", {}).get("run_log_dir", "out")
    write_run_log(out_dir, run_id, cfg, ctx)


if __name__ == "__main__":
    main()
