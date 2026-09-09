"""One-user-message protocol with exact schema, definition and retry evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from iadopt_lab.corpus.ingestion import DEMONSTRATION_PATHS
from iadopt_lab.validation import validate_prediction

RENDERER_VERSION = "one-user-message-v1"
CORRECTION_VERSION = "immediate-predecessor-v1"


def _json(value: Any) -> str:
    """Encode deterministic readable UTF-8 JSON with preserved array order.

    Args: JSON primitives with finite numeric values.
    Returns: compact sorted-key JSON string without ASCII folding.
    Raises: TypeError/ValueError for non-JSON input.
    Side effects: none.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _hash(raw: bytes) -> str:
    """Calculate a raw artifact hash.

    Args: exact bytes, never an implicit file path.
    Returns: SHA-256 hexadecimal digest.
    Raises: TypeError for non-byte input.
    Side effects: none.
    """
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PromptVersion:
    """Frozen template bytes bound to historical-reference compatibility evidence."""

    prompt_id: str
    version: str
    display_name: str
    text: str
    sha256: str
    historical_sha256: str
    review_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return serializable template evidence.

        Args: this immutable prompt version.
        Returns: independent primitive dictionary.
        Raises: no content errors.
        Side effects: none.
        """
        return asdict(self)


@dataclass(frozen=True)
class RenderedPrompt:
    """Exact provider-neutral messages and their reconstructable provenance."""

    messages: tuple[dict[str, str], ...]
    content: str
    metadata: dict[str, Any]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible prompt evidence without changing model-visible bytes.

        Args: this complete rendering.
        Returns: dictionary whose messages are an ordered list.
        Raises: no content errors.
        Side effects: none.
        """
        result = asdict(self)
        result["messages"] = list(result["messages"])
        return result


def load_prompt_version(prompt_id: str, project_root: Path | str | None = None) -> PromptVersion:
    """Load and verify one exact registered prompt artifact and historical lineage.

    Args: stable family ID, optionally suffixed ':v1', and optional iadopt-lab root.
    Returns: PromptVersion with verified bytes/hash and accepted compatibility review.
    Raises: ValueError for unknown ID, moved artifact, or invalid historical evidence;
    OSError for missing files. Historical aliases do not resolve as active families.
    Side effects: read-only template and registry filesystem access.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    registry = json.loads((root / "data/manifests/prompt-registry-v1.json").read_bytes())
    family = prompt_id.removesuffix(":v1")
    entries = [entry for entry in registry["prompts"] if entry["prompt_id"] == family]
    if len(entries) != 1:
        raise ValueError("Unknown frozen prompt ID: " + prompt_id)
    entry = entries[0]
    raw = (root / entry["template_path"]).read_bytes()
    historical_raw = entry["historical_text"].encode("utf-8")
    if _hash(raw) != entry["template_sha256"] or _hash(historical_raw) != entry["historical_sha256"]:
        raise ValueError("Prompt template or historical source hash mismatch")
    if entry["review_status"] != "approved-compatibility-policy-D021":
        raise ValueError("Prompt compatibility policy has not been accepted")
    text = raw.decode("utf-8")
    for placeholder in ("{{schema}}", "{{demonstrations}}", "{{target_definition}}"):
        if text.count(placeholder) != 1:
            raise ValueError("Template requires each protocol placeholder exactly once")
    return PromptVersion(family, "v1", entry["display_name"], text, _hash(raw),
                         entry["historical_sha256"], entry["review_status"])


def select_demonstrations(pool: Iterable[dict[str, Any]], shot_count: int) -> tuple[dict[str, Any], ...]:
    """Choose a fixed prefix after verifying the complete approved source pool.

    Args: five ordered canonical records and shot count in 0,1,3,5.
    Returns: immutable tuple of the requested prefix in unchanged order.
    Raises: ValueError for wrong paths, duplicate IDs, invalid gold or shot count.
    Side effects: reads schema during validation; no randomization or target-aware choice.
    """
    examples = tuple(pool)
    if type(shot_count) is not int or shot_count not in (0, 1, 3, 5):
        raise ValueError("Shot count must be 0, 1, 3 or 5")
    if tuple(record["source_path"] for record in examples) != DEMONSTRATION_PATHS:
        raise ValueError("Demonstration pool must match all five approved paths in order")
    if len({record["variable_id"] for record in examples}) != 5:
        raise ValueError("Demonstration variable IDs must be unique")
    for example in examples:
        if not validate_prediction(example["gold"]).valid:
            raise ValueError("Invalid demonstration gold")
    return examples[:shot_count]


def _rendered(content: str, metadata: dict[str, Any]) -> RenderedPrompt:
    """Construct exact message evidence without adding provider-specific instructions.

    Args: complete one-user-message content and scientific lineage metadata.
    Returns: rendering with message-array hash and exact text/byte/character counts.
    Raises: UnicodeError for content that cannot be represented as UTF-8.
    Side effects: none.
    """
    raw = content.encode("utf-8")
    messages = ({"role": "user", "content": content},)
    final_hash = _hash(_json(list(messages)).encode("utf-8"))
    return RenderedPrompt(messages, content, {**metadata, "content_sha256": _hash(raw),
                          "byte_length": len(raw), "character_length": len(content),
                          "token_estimate": None}, final_hash)


def render_base_prompt(template: PromptVersion, target_definition: str, schema: bytes,
                       demonstrations: Iterable[dict[str, Any]] = (), *,
                       target_id: str | None = None) -> RenderedPrompt:
    """Render the exact frozen schema, approved examples and one target definition.

    Args: verified template, exact target definition, exact runtime schema bytes,
    zero/one/three/five ordered example records, optional non-model-visible target ID.
    Actions: verify artifacts and prefix membership, insert each placeholder once,
    retaining target text byte-for-byte and keeping all target gold/provenance out.
    Returns: RenderedPrompt with one user message and component/final hashes.
    Raises: ValueError for mismatch, invalid examples, overlap or unsupported shots;
    UnicodeError for invalid text. Context capacity is enforced by caller preflight.
    Side effects: none; no target record, model, tokenizer or database is consulted.
    """
    if not isinstance(template, PromptVersion) or _hash(template.text.encode("utf-8")) != template.sha256:
        raise ValueError("Renderer requires an intact frozen PromptVersion")
    if not isinstance(target_definition, str) or not isinstance(schema, bytes):
        raise TypeError("Target definition must be text and schema must be exact bytes")
    schema_text = schema.decode("utf-8")
    examples = tuple(demonstrations)
    if len(examples) not in (0, 1, 3, 5) or tuple(example["source_path"] for example in examples) != DEMONSTRATION_PATHS[:len(examples)]:
        raise ValueError("Examples must be an approved 0/1/3/5 ordered prefix")
    demo_payload, demo_evidence = [], []
    for position, example in enumerate(examples, 1):
        if target_id is not None and target_id == example["variable_id"]:
            raise ValueError("Target must not be a demonstration")
        if not validate_prediction(example["gold"], schema).valid:
            raise ValueError("Demonstration fails the runtime schema")
        if _hash(_json(example["gold"]).encode("utf-8")) != example["gold_sha256"]:
            raise ValueError("Demonstration gold hash mismatch")
        demo_payload.append({"demonstration": position, "definition": example["definition"], "decomposition": example["gold"]})
        demo_evidence.append({"position": position, "variable_id": example["variable_id"],
                              "source_sha256": example["source_sha256"], "gold_sha256": example["gold_sha256"]})
    # Split placeholders in the template only: braces in inserted data are inert.
    values = {"schema": schema_text, "demonstrations": _json(demo_payload), "target_definition": target_definition}
    import re
    content = re.sub(r"\{\{(schema|demonstrations|target_definition)\}\}", lambda match: values[match.group(1)], template.text)
    metadata = {"renderer_version": RENDERER_VERSION, "prompt_id": template.prompt_id,
                "prompt_version": template.version, "template_sha256": template.sha256,
                "historical_sha256": template.historical_sha256, "schema_sha256": _hash(schema),
                "target_definition_sha256": _hash(target_definition.encode("utf-8")),
                "demonstrations": demo_evidence, "demonstrations_sha256": _hash(_json(demo_payload).encode("utf-8")),
                "correction": None}
    return _rendered(content, metadata)


def render_correction(base: RenderedPrompt, raw_response: str, errors: Iterable[dict[str, Any]],
                      previous_attempt_number: int = 1) -> RenderedPrompt:
    """Append only the immediately previous content-invalid response and full errors.

    Args: untouched base rendering, exact assistant-visible raw response, ordered
    extraction/schema/semantic errors and predecessor attempt number 1 or 2.
    Returns: one-user-message rendering with base/correction/final hash lineage.
    Raises: ValueError for modified/corrected base, invalid predecessor or empty errors.
    Side effects: none; does not insert hidden reasoning or older failures.
    """
    if base.metadata.get("correction") is not None:
        raise ValueError("Corrections must be derived from the original base, not a prior correction")
    if _rendered(base.content, {}).sha256 != base.sha256:
        raise ValueError("Base prompt hash mismatch")
    if type(previous_attempt_number) is not int or previous_attempt_number not in (1, 2):
        raise ValueError("Correction predecessor must be attempt 1 or 2")
    ordered_errors = list(errors)
    if not ordered_errors:
        raise ValueError("A correction requires the complete nonempty content error bundle")
    raw_bytes = raw_response.encode("utf-8")
    feedback = {"protocol_version": CORRECTION_VERSION, "previous_attempt_number": previous_attempt_number,
                "previous_response": raw_response, "previous_response_sha256": _hash(raw_bytes),
                "previous_response_byte_length": len(raw_bytes), "errors": ordered_errors,
                "instruction": "Return exactly one corrected JSON object matching the schema. Do not add unstated information."}
    feedback_text = _json(feedback)
    content = base.content + "\nVALIDATION CORRECTION\n" + feedback_text + "\n"
    metadata = {**base.metadata, "base_sha256": base.sha256,
                "correction": {"version": CORRECTION_VERSION, "previous_attempt_number": previous_attempt_number,
                               "previous_response_sha256": _hash(raw_bytes), "sha256": _hash(feedback_text.encode("utf-8"))}}
    return _rendered(content, metadata)


def render_correction_prompt(base: RenderedPrompt, previous_attempt: dict[str, Any]) -> RenderedPrompt:
    """Adapt a typed previous-attempt evidence mapping to the correction renderer.

    Args: original base and mapping with raw_response, errors and attempt_number.
    Returns: exact immediate-predecessor correction rendering.
    Raises: KeyError for incomplete evidence or renderer validation errors.
    Side effects: none; unknown provider fields and hidden reasoning are ignored.
    """
    return render_correction(base, previous_attempt["raw_response"], previous_attempt["errors"], previous_attempt["attempt_number"])
