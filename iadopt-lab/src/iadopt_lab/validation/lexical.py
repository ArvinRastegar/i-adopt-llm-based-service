"""Gold-independent representation validation with deterministic evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ENTITY_FIELDS = ("hasObjectOfInterest", "hasMatrix", "hasContextObject")
FIELDS = ("hasStatisticalModifier", "hasProperty", *ENTITY_FIELDS, "hasConstraint")
POLICY_VERSION = "lexical-cross-reference-v1"


def _bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible candidate deterministically.

    Args: finite JSON primitives with ordered arrays.
    Returns: sorted-key compact UTF-8 bytes.
    Raises: TypeError or ValueError for unsupported/non-finite values.
    Side effects: none.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _key(text: str) -> str:
    """Build the frozen lexical cross-reference key.

    Args: original lexical text.
    Returns: NFC-normalized, stripped, case-folded comparison string.
    Raises: TypeError for nontext input.
    Side effects: none; retained scoreable text is not changed.
    """
    return unicodedata.normalize("NFC", text).strip().casefold()


def lexical_order(text: str) -> tuple[str, str]:
    """Build a stable comparison key for set-like lexical collections.

    Args: preserved lexical text.
    Returns: NFC/casefold key and original-text tie-break tuple.
    Raises: TypeError for nontext input.
    Side effects: none; asymmetric roles are never ordered with this function.
    """
    return unicodedata.normalize("NFC", text).casefold(), text


def load_schema_bytes(project_root: Path | str | None = None) -> bytes:
    """Read the exact shared prompt/runtime lexical schema.

    Args: optional iadopt-lab root; defaults to this installed source tree's root.
    Returns: original schema bytes without reserialization.
    Raises: OSError for missing or unreadable schema.
    Side effects: read-only filesystem access.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    return (root / "schemas/lexical-decomposition.schema.json").read_bytes()


def empty_prediction() -> dict[str, Any]:
    """Create the explicit six-field empty representation.

    Args: none.
    Returns: new dictionary with empty scalar/entity strings and empty constraints.
    Raises: no content-validation exceptions.
    Side effects: none. The caller separately records valid-empty or failure provenance.
    """
    return {field: [] if field == "hasConstraint" else "" for field in FIELDS}


def readable_prediction(prediction: dict[str, Any], *, label: str, definition: str) -> dict[str, Any]:
    """Build the human-readable view of a scored prediction (D-031).

    The label and definition are copied from the target's canonical corpus record.
    The model is never asked for them: it has not seen the label, so a generated one
    would be invented, and re-emitting the definition costs output tokens on every
    task for a field that carries no score. Neither value is evaluated.

    Args: prediction: canonical six-field prediction; label/definition: exact corpus values.
    Returns: new mapping with label, definition and the six fields in canonical order.
    Raises: ValueError when the prediction is not exactly the six lexical fields, which
    keeps this presentation helper from silently accepting evaluator-invalid input.
    Side effects: none. The supplied prediction is copied, never modified, and this
    result must not reach the evaluator, a prediction hash, or any metric.
    """
    if set(prediction) != set(FIELDS):
        raise ValueError("A readable prediction is built from exactly the six lexical fields")
    if not isinstance(label, str) or not isinstance(definition, str):
        raise ValueError("Readable label and definition must be corpus strings")
    return {"label": label, "definition": definition,
            **{field: copy.deepcopy(prediction[field]) for field in FIELDS}}


def system_display_label(system: dict[str, Any]) -> str:
    """Derive an unscored stable display label from system members.

    Args: schema-valid symmetric or asymmetric system mapping.
    Returns: sorted-part, source/target or numerator/denominator display string.
    Raises: KeyError or TypeError for programmer-supplied invalid shape.
    Side effects: none; original member text and asymmetric roles are preserved.
    """
    if "SymmetricSystem" in system:
        return " + ".join(sorted(system["hasPart"], key=lexical_order))
    if "hasNumerator" in system:
        return system["hasNumerator"] + " / " + system["hasDenominator"]
    return system["hasSource"] + " → " + system["hasTarget"]


@dataclass(frozen=True)
class ValidationResult:
    """Serializable validation outcome; invalid predictions carry errors, never a repaired value."""

    valid: bool
    errors: tuple[dict[str, Any], ...]
    canonical_prediction: dict[str, Any] | None
    candidate_sha256: str | None
    schema_sha256: str
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Expose complete independent validation evidence.

        Args: this completed validation result.
        Returns: JSON-compatible deep dictionary with list-valued errors.
        Raises: no content-validation exceptions.
        Side effects: none; does not mutate the original candidate/result.
        """
        result = asdict(self)
        result["errors"] = list(result["errors"])
        return result


def _pointer(parts: Any) -> str:
    """Convert an error's path to a JSON Pointer.

    Args: iterable string or integer path components.
    Returns: RFC6901 escaped pointer string; root is the empty string.
    Raises: TypeError for noniterable programmer input.
    Side effects: none.
    """
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _targets(candidate: dict[str, Any]) -> dict[str, set[str]]:
    """Index all allowed component and whole-system Constraint targets.

    Args: a schema-valid six-field candidate.
    Returns: normalized alias-to-distinct-preserved-label set mapping.
    Raises: KeyError for malformed programmer input.
    Side effects: none; container strings are aliases, never scoreable system members.
    """
    result: dict[str, set[str]] = {}
    for field in FIELDS[:-1]:
        value = candidate[field]
        if isinstance(value, str):
            entries = [(value, value)] if value.strip() else []
        else:
            display = system_display_label(value)
            container = "SymmetricSystem" if "SymmetricSystem" in value else "AsymmetricSystem"
            entries = [(display, display)]
            if value[container].strip():
                entries.append((value[container], display))
            if container == "SymmetricSystem":
                entries.extend((part, part) for part in value["hasPart"])
            else:
                entries.extend((part, part) for role, part in value.items() if role != container)
        for alias, resolved in entries:
            result.setdefault(_key(alias), set()).add(resolved)
    return result


def _canonicalize(candidate: dict[str, Any], targets: dict[str, set[str]]) -> dict[str, Any]:
    """Derive canonical metadata only after representation validation.

    Args: valid candidate and its unambiguous target index.
    Returns: deep copy with derived system labels/sorted parts and resolved targets.
    Raises: KeyError for an internal post-validation invariant violation.
    Side effects: none; original candidate and Constraint array order remain unchanged.
    """
    result = copy.deepcopy(candidate)
    for field in ENTITY_FIELDS:
        value = result[field]
        if not isinstance(value, dict):
            continue
        if "SymmetricSystem" in value:
            value["hasPart"] = sorted(value["hasPart"], key=lexical_order)
            value["SymmetricSystem"] = system_display_label(value)
        else:
            value["AsymmetricSystem"] = system_display_label(value)
    # Array order remains model order: January's greedy Constraint scorer depends on ties.
    for constraint in result["hasConstraint"]:
        constraint["on"] = next(iter(targets[_key(constraint["on"])]))
    return result


def validate_prediction(candidate: Any, schema_bytes: bytes | None = None) -> ValidationResult:
    """Validate JSON candidate without gold or repair and return complete ordered evidence.

    Args: arbitrary JSON value and optional exact lexical schema bytes.
    Actions: enforce schema, unique symmetric parts and resolvable Constraint targets;
    derive unscored system metadata only after success.
    Returns: ValidationResult with canonical prediction only after validation succeeds.
    Raises: schema corruption or filesystem errors; ordinary invalid content is returned.
    Side effects: read packaged schema when not supplied; never mutates the candidate.
    """
    schema_raw = load_schema_bytes() if schema_bytes is None else schema_bytes
    schema = json.loads(schema_raw)
    Draft202012Validator.check_schema(schema)
    schema_hash = hashlib.sha256(schema_raw).hexdigest()
    try:
        candidate_hash = hashlib.sha256(_bytes(candidate)).hexdigest()
    except (ValueError, TypeError, UnicodeError):
        return ValidationResult(False, ({"stage": "schema", "code": "not_json_value", "pointer": "", "message": "Candidate is not a finite UTF-8 JSON value."},), None, None, schema_hash)
    errors: list[dict[str, Any]] = []
    for error in Draft202012Validator(schema).iter_errors(candidate):
        errors.append({"stage": "schema", "code": "schema_" + str(error.validator),
                       "pointer": _pointer(error.absolute_path), "message": error.message})
    if errors:
        errors.sort(key=lambda item: (item["pointer"], item["code"], item["message"]))
        return ValidationResult(False, tuple(errors), None, candidate_hash, schema_hash)
    for field in ENTITY_FIELDS:
        value = candidate[field]
        if isinstance(value, dict) and "hasPart" in value:
            keys = [_key(part) for part in value["hasPart"]]
            if len(set(keys)) != len(keys):
                errors.append({"stage": "semantic", "code": "duplicate_system_part", "pointer": "/" + field + "/hasPart", "message": "Symmetric members must be unique under lexical target normalization."})
    targets = _targets(candidate)
    for index, constraint in enumerate(candidate["hasConstraint"]):
        matches = targets.get(_key(constraint["on"]), set())
        if len(matches) != 1:
            code = "unknown_constraint_target" if not matches else "ambiguous_constraint_target"
            errors.append({"stage": "semantic", "code": code, "pointer": f"/hasConstraint/{index}/on", "message": "Constraint target must resolve to exactly one emitted lexical value."})
    errors.sort(key=lambda item: (item["pointer"], item["code"], item["message"]))
    return ValidationResult(not errors, tuple(errors), None if errors else _canonicalize(candidate, targets), candidate_hash, schema_hash)


def validate_candidate(candidate: Any, schema_bytes: bytes | None = None) -> ValidationResult:
    """Provide the component-contract alias for prediction validation.

    Args: JSON candidate and optional exact schema bytes.
    Returns: ValidationResult from the same gold-independent validation path.
    Raises: schema-artifact errors, not ordinary model-content failures.
    Side effects: reads packaged schema only when bytes are not supplied.
    """
    return validate_prediction(candidate, schema_bytes)


def canonicalize_prediction(candidate: Any, schema_bytes: bytes | None = None) -> dict[str, Any]:
    """Require successful validation before exposing evaluator input.

    Args: candidate JSON value and optional exact schema bytes.
    Returns: independent canonical six-field prediction.
    Raises: ValueError for invalid model content; schema/filesystem errors propagate.
    Side effects: reads packaged schema only when not supplied; no content repair.
    """
    result = validate_prediction(candidate, schema_bytes)
    if not result.valid:
        raise ValueError("Cannot canonicalize invalid prediction: " + json.dumps(result.errors))
    assert result.canonical_prediction is not None
    return result.canonical_prediction
