"""Response extraction protocol regressions with no network or syntax repair."""

import json

import pytest

from iadopt_lab.generation.extractor import extract_json


@pytest.mark.parametrize("raw,strategy", [
    ('  {"a":1}\n', "whole_response"),
    ('```json\n{"a":1}\n```', "markdown_fence"),
    ('```\n{"a":1}\n```', "markdown_fence"),
    ('Here is JSON: {"a":1} done', "balanced_object"),
    ('```python\n{"a":1}\n```', "balanced_object"),
    (json.dumps('{"a":1}'), "whole_response"),
])
def test_supported_wrappers(raw, strategy):
    """Accept unique object responses using only the documented ordered strategies.

    Args: wrapped input and expected selected strategy. Returns: None.
    Raises: AssertionError on extraction regression. Side effects: none.
    """
    result = extract_json(raw)
    assert result.success and result.candidate == {"a": 1}
    assert result.strategy == strategy and result.selected_sha256
    assert result.to_dict() == extract_json(raw).to_dict()


@pytest.mark.parametrize("raw,code", [
    ('[{"a":1}]', "top_level_not_object"),
    ('null', "top_level_not_object"), ('true', "top_level_not_object"),
    ('4', "top_level_not_object"),
    (json.dumps(json.dumps('{"a":1}')), "multiple_json_string_wrappers"),
    ('{"a":1}{"b":2}', "ambiguous_json"),
    ('```json\n{}\n```\n```json\n[]\n```', "ambiguous_json"),
    ('```json\n[]\n```\nprose {}', "top_level_not_object"),
    ('{"a":1,}', "no_valid_json_object"),
    ('{"a":1,"a":2}', "no_valid_json_object"),
    ('{"a":NaN}', "no_valid_json_object"),
    ('{"a":1e999}', "no_valid_json_object"),
    ('no JSON', "no_valid_json_object"),
])
def test_rejected_responses(raw, code):
    """Reject scalars, ambiguity and malformed JSON without mining or repair.

    Args: invalid input and expected stable failure code. Returns: None.
    Raises: AssertionError on accidental acceptance. Side effects: none.
    """
    result = extract_json(raw)
    assert not result.success and result.candidate is None
    assert result.errors[0]["code"] == code


def test_quotes_braces_escapes_and_utf8_offsets():
    """Preserve escaped literal braces and report source-space UTF-8 byte offsets.

    Args: none. Returns: None.
    Raises: AssertionError on scanner/offset mutation. Side effects: none.
    """
    value = {"label": 'water } { "quoted" \\ text', "nested": {"value": "é"}}
    raw = "α β " + json.dumps(value, ensure_ascii=False) + " tail"
    result = extract_json(raw)
    assert result.success and result.candidate == value
    selected = [item for item in result.candidates if item["strategy"] == "balanced_object" and item["parsed"]][0]
    span = raw.encode("utf-8")[selected["start_byte"]:selected["end_byte"]]
    assert json.loads(span) == value
    assert result.coordinate_space == "raw_response_utf8"


def test_safety_limits_and_bad_api_inputs():
    """Return bounded content failures and reject invalid programmer arguments.

    Args: none. Returns: None.
    Raises: AssertionError on unsafe acceptance. Side effects: none.
    """
    assert extract_json("{}", max_bytes=1).errors[0]["code"] == "response_byte_limit"
    assert extract_json("{} {}", max_candidates=1).errors[0]["code"] == "candidate_count_limit"
    assert not extract_json('{"a":{"b":1}}', max_depth=1).success
    assert extract_json("\ud800").errors[0]["code"] == "invalid_utf8_text"
    with pytest.raises(TypeError):
        extract_json(None)
    with pytest.raises(ValueError):
        extract_json("{}", max_depth=0)


def test_decoded_wrapper_coordinate_space():
    """Identify offsets as decoded-string bytes after exactly one wrapper is removed.

    Args: none. Returns: None.
    Raises: AssertionError on incorrect wrapper evidence. Side effects: none.
    """
    result = extract_json(json.dumps('text {"a":1}'))
    assert result.success and result.json_string_unwrapped
    assert result.coordinate_space == "decoded_json_string_utf8"
