"""Deterministic, non-repairing JSON extraction from assistant-visible output."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

EXTRACTOR_VERSION = "json-extractor-v1"


@dataclass(frozen=True)
class ExtractionResult:
    """Pure extraction evidence; raw provider text remains independently authoritative."""

    success: bool
    candidate: dict[str, Any] | None
    errors: tuple[dict[str, Any], ...]
    strategy: str | None
    json_string_unwrapped: bool
    candidates: tuple[dict[str, Any], ...]
    coordinate_space: str
    raw_sha256: str
    selected_sha256: str | None
    safety_limits: dict[str, int]
    version: str = EXTRACTOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result without exposing references to mutable evidence.

        Args: self, containing a completed extraction outcome.
        Returns: JSON-compatible dictionary with list-valued errors/candidates.
        Raises: no content-validation exceptions.
        Side effects: none; the original result is unchanged.
        """
        value = asdict(self)
        value["errors"] = list(value["errors"])
        value["candidates"] = list(value["candidates"])
        return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object without silently accepting duplicate JSON keys.

    Args: ordered key/value pairs supplied by the JSON decoder.
    Returns: the decoded object when all keys are unique.
    Raises: ValueError for duplicate keys.
    Side effects: none.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject the decoder's non-standard NaN and infinity constants.

    Args: decoder constant text.
    Returns: never returns normally.
    Raises: ValueError for every supplied constant.
    Side effects: none.
    """
    raise ValueError("Non-finite JSON number is forbidden")


def _check_depth(text: str, limit: int) -> None:
    """Bound container nesting without counting braces inside JSON strings.

    Args: candidate text and positive maximum nesting depth.
    Returns: None when nesting does not exceed the limit; parsing checks syntax.
    Raises: ValueError when the resource limit is exceeded.
    Side effects: none.
    """
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > limit:
                raise ValueError("JSON nesting safety limit exceeded")
        elif char in "]}":
            depth -= 1


def _strict_parse(text: str, max_depth: int) -> Any:
    """Parse one strict JSON value after checking resource bounds.

    Args: candidate text and nesting limit.
    Returns: parsed JSON primitives with unique object keys and finite numbers.
    Raises: ValueError, JSONDecodeError, or RecursionError for invalid content.
    Side effects: none; no syntax or field repair occurs.
    """
    _check_depth(text, max_depth)
    result = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    # Very large exponent notation can decode to infinity without parse_constant.
    json.dumps(result, allow_nan=False, ensure_ascii=False).encode("utf-8")
    return result


def _balanced_spans(text: str, limit: int) -> list[tuple[int, int]]:
    """Find complete outermost object spans using quote/escape-aware scanning.

    Args: full response text and positive maximum candidate count.
    Returns: character-offset half-open spans in source order.
    Raises: ValueError if the candidate resource limit is exceeded.
    Side effects: none. Unclosed spans are not repaired or returned.
    """
    spans, start, depth, quoted, escaped = [], None, 0, False, False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, index + 1))
                if len(spans) > limit:
                    raise ValueError("Object candidate safety limit exceeded")
    return spans


def extract_json(raw_text: str, *, max_bytes: int = 1_048_576, max_depth: int = 64,
                 max_fences: int = 64, max_candidates: int = 64) -> ExtractionResult:
    """Extract a unique JSON object using the frozen ordered protocol.

    Args: untouched assistant text; explicit byte, depth, fence and candidate limits.
    Actions: parse whole response, unwrap one JSON string, try complete eligible
    fences, then balanced outermost objects. Record every inspected candidate.
    Returns: ExtractionResult; malformed output, ambiguity and limits are failures
    represented as data, never permission to select the first answer or repair it.
    Raises: TypeError for a non-string input; ValueError for invalid safety limits.
    Side effects: none. Candidate offsets are UTF-8 bytes in the named coordinate space.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    limits = {"max_bytes": max_bytes, "max_depth": max_depth,
              "max_fences": max_fences, "max_candidates": max_candidates}
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("Extraction limits must be positive integers")
    raw_bytes = raw_text.encode("utf-8", errors="surrogatepass")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    evidence: list[dict[str, Any]] = []
    unwrapped = False
    space = "raw_response_utf8"
    working = raw_text

    def finish(code: str | None, candidate: Any = None, strategy: str | None = None,
               selected: str | None = None) -> ExtractionResult:
        """Build one deterministic outcome from the current scan evidence.

        Args: optional failure code, accepted object, method and exact selected text.
        Returns: immutable extraction result.
        Raises: no parser errors.
        Side effects: none; captured evidence is copied to an immutable tuple.
        """
        errors = () if code is None else ({"stage": "extraction", "code": code,
                                          "pointer": "", "message": code.replace("_", " ")},)
        digest = hashlib.sha256(selected.encode("utf-8")).hexdigest() if selected is not None else None
        return ExtractionResult(code is None, candidate, errors, strategy, unwrapped,
                                tuple(evidence), space, raw_hash, digest, limits)

    def inspect(start: int, end: int, strategy: str) -> tuple[bool, Any, str]:
        """Strictly parse a span and append its complete diagnostic evidence.

        Args: character start/end offsets and extraction strategy in current text.
        Returns: (parsed-successfully, parsed-value-or-None, exact-candidate-text).
        Raises: no model-controlled parser exceptions.
        Side effects: appends one diagnostic record to this extraction's local list.
        """
        text = working[start:end]
        item = {"strategy": strategy, "coordinate_space": space,
                "start_byte": len(working[:start].encode("utf-8", errors="surrogatepass")),
                "end_byte": len(working[:end].encode("utf-8", errors="surrogatepass")),
                "sha256": hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()}
        try:
            value = _strict_parse(text, max_depth)
            item.update({"parsed": True, "json_type": type(value).__name__})
            evidence.append(item)
            return True, value, text
        except (ValueError, RecursionError, UnicodeError) as error:
            item.update({"parsed": False, "error_type": type(error).__name__, "message": str(error)})
            evidence.append(item)
            return False, None, text

    if len(raw_bytes) > max_bytes:
        return finish("response_byte_limit")
    try:
        raw_text.encode("utf-8")
    except UnicodeError:
        return finish("invalid_utf8_text")
    while True:
        start = len(working) - len(working.lstrip())
        end = len(working.rstrip())
        parsed, value, selected = inspect(start, end, "whole_response")
        if parsed and isinstance(value, dict):
            return finish(None, value, "whole_response", selected)
        if parsed and isinstance(value, str):
            if unwrapped:
                return finish("multiple_json_string_wrappers")
            unwrapped, working, space = True, value, "decoded_json_string_utf8"
            continue
        if parsed:
            return finish("top_level_not_object")
        break
    # Complete fenced bodies only. A language other than json is not privileged.
    fence_pattern = re.compile(r"(?m)^[ \t]*```(?:json)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)", re.DOTALL | re.IGNORECASE)
    fences = list(fence_pattern.finditer(working))
    if len(fences) > max_fences:
        return finish("fence_count_limit")
    successes = []
    for match in fences:
        parsed, value, selected = inspect(match.start(1), match.end(1), "markdown_fence")
        if parsed:
            successes.append((value, selected))
    if successes:
        if len(successes) != 1:
            return finish("ambiguous_json")
        value, selected = successes[0]
        if not isinstance(value, dict):
            return finish("top_level_not_object")
        return finish(None, value, "markdown_fence", selected)
    try:
        spans = _balanced_spans(working, max_candidates)
    except ValueError:
        return finish("candidate_count_limit")
    successes = []
    for start, end in spans:
        parsed, value, selected = inspect(start, end, "balanced_object")
        if parsed and isinstance(value, dict):
            successes.append((value, selected))
    if len(successes) > 1:
        return finish("ambiguous_json")
    if not successes:
        return finish("no_valid_json_object")
    value, selected = successes[0]
    return finish(None, value, "balanced_object", selected)
