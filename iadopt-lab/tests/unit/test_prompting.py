"""Historical-compatible prompt snapshots, leakage and immediate correction tests."""

import hashlib
import json
from pathlib import Path

import pytest

from iadopt_lab.corpus import DEMONSTRATION_PATHS, load_canonical_records
from iadopt_lab.prompting import (
    load_prompt_version,
    render_base_prompt,
    render_correction,
    select_demonstrations,
)
from iadopt_lab.validation import load_schema_bytes

ROOT = Path(__file__).resolve().parents[2]
# Corpus v2.0.1 (D-030) changed the gold of PersWelfare, demonstration position 2.
# It appears only in the 3-shot and 5-shot prefixes, so exactly those two hashes moved
# per family; the 0-shot and 1-shot values are byte-identical to the v2.0.0 snapshots.
SNAPSHOTS = {
    "strict-minimal": ["0a5aeba3594aa5d0a20558f306ae48edf1b2954aba09a71be9f6a0dff5bbf0e0", "eca39932e775aac060cbcdd0e7ac582b3046c24b8b9f345ee345cbb97b319c51", "e021fe61f1571d579448c5674245668bcba4509b1f88ecd200bafef05472e0ff", "794c751aafa6f77e2cff9c937cd40d1d258d165bcc5fe3ab0963ff554c10b2e4"],
    "constraint-decomposition": ["a353b76d20ed1b0ed705a364288c47f3759bb7caae960aaf96e9135e0b069809", "7180af0f9795e57cde309b79d065b71ec1fe63f825b60a0fedc51d2946656fbd", "c32bc36a1952f05da00c74ef626f0699c10205edf590fc76fc14be6632e814e0", "e8d59631d8ff965b32ee95f6bda0c52ea170799d652c14dcfc0b3392f03ffa7b"],
    "matrix-decomposition": ["4f2f0f7b80114ff7006cec75fc9312fd3cf7202517c8c88f906ac35d92785cb9", "046e3b0d1b2adfbd7e1ae036133401920487fdfadcc744e3c030d09046774ebe", "d624fe5fff4acb037c6dbfa779668aeb7df0f732c52a8a8266c996e25d7a2060", "4c35f23a8167c292ce1890df234afff828473bff1701470590733e48b84837f3"],
}


@pytest.fixture(scope="module")
def prompt_inputs():
    """Load immutable examples and a deterministic non-demo target once per module.

    Args: none. Returns: (approved demo pool, first non-demo target) tuple.
    Raises: corpus integrity errors. Side effects: read-only corpus access.
    """
    records = load_canonical_records(ROOT)
    pool = tuple(next(record for record in records if record["source_path"] == path) for path in DEMONSTRATION_PATHS)
    target = next(record for record in records if record["source_path"] not in DEMONSTRATION_PATHS)
    return pool, target


@pytest.mark.parametrize("family", list(SNAPSHOTS))
@pytest.mark.parametrize("shots", [0, 1, 3, 5])
def test_all_prompt_shot_snapshots(prompt_inputs, family, shots):
    """Prove exact schema, example order, target-only rendering and all12 frozen hashes.

    Args: corpus fixture, family and shot count. Returns: None.
    Raises: AssertionError on protocol byte drift or leakage. Side effects: artifact reads.
    """
    pool, target = prompt_inputs
    schema = load_schema_bytes(ROOT)
    result = render_base_prompt(load_prompt_version(family, ROOT), target["definition"], schema,
                                select_demonstrations(pool, shots), target_id=target["variable_id"])
    assert result.sha256 == SNAPSHOTS[family][[0, 1, 3, 5].index(shots)]
    assert len(result.messages) == 1 and result.messages[0]["role"] == "user"
    assert schema.decode("utf-8") in result.content
    assert result.metadata["schema_sha256"] == hashlib.sha256(schema).hexdigest()
    assert target["definition"] in result.content
    for key in ("variable_id", "source_iri", "source_path", "category", "subcategory"):
        assert target[key] not in result.content
    assert [entry["variable_id"] for entry in result.metadata["demonstrations"]] == [entry["variable_id"] for entry in pool[:shots]]


def test_retained_prompt_limitations_and_historical_evidence(prompt_inputs):
    """Keep accepted no-inference rules and unchanged non-literal/geographic demo gold.

    Args: approved prompt fixtures. Returns: None.
    Raises: AssertionError if historical policy or demonstration gold is repaired.
    Side effects: template/registry reads only.
    """
    pool, _ = prompt_inputs
    expected_matrix = {0: "atmosphere", 2: "person", 3: "sewer line", 4: "urban area"}
    for index, expected in expected_matrix.items():
        assert pool[index]["gold"]["hasMatrix"] == expected
        assert expected not in pool[index]["definition"]
    assert pool[3]["gold"]["hasObjectOfInterest"] == "water circulation"
    registry = json.loads((ROOT / "data/manifests/prompt-registry-v1.json").read_bytes())
    for entry in registry["prompts"]:
        template = load_prompt_version(entry["prompt_id"], ROOT)
        assert "Do not infer or invent new concepts." in template.text
        assert hashlib.sha256(entry["historical_text"].encode()).hexdigest() == entry["historical_sha256"]
        assert entry["unified_diff"] and "definition must be exactly the same string" in entry["historical_text"]
    matrix = load_prompt_version("matrix-decomposition", ROOT)
    assert "Never use methods, units, instruments, or locations." in matrix.text


def test_demonstration_guards(prompt_inputs):
    """Reject reordered examples, invalid shot counts and target/demo overlap.

    Args: verified corpus fixture. Returns: None.
    Raises: AssertionError if leakage guards fail. Side effects: artifact reads.
    """
    pool, _ = prompt_inputs
    with pytest.raises(ValueError):
        select_demonstrations(reversed(pool), 3)
    with pytest.raises(ValueError):
        select_demonstrations(pool, 2)
    with pytest.raises(ValueError):
        render_base_prompt(load_prompt_version("strict-minimal", ROOT), pool[0]["definition"],
                           load_schema_bytes(ROOT), pool[:1], target_id=pool[0]["variable_id"])
    with pytest.raises(ValueError):
        load_prompt_version("constraint_tree", ROOT)


def test_correction_round_trip_and_predecessor_isolation(prompt_inputs):
    """Retain only immediate prior visible output/errors with unchanged base science.

    Args: verified prompt fixture. Returns: None.
    Raises: AssertionError on history accumulation, encoding loss or lineage mismatch.
    Side effects: template/schema reads only.
    """
    _, target = prompt_inputs
    base = render_base_prompt(load_prompt_version("strict-minimal", ROOT), target["definition"], load_schema_bytes(ROOT))
    raw1 = 'attempt one ```\n{"text":"é \\" } {{target_definition}}"}\n'
    raw2 = "attempt two's own output"
    errors = [{"stage": "schema", "code": "schema_required", "pointer": "/x", "message": "missing"}]
    attempt2 = render_correction(base, raw1, errors, 1)
    attempt3 = render_correction(base, raw2, errors, 2)
    assert attempt2.content.startswith(base.content)
    feedback = json.loads(attempt2.content[len(base.content):].split("VALIDATION CORRECTION\n", 1)[1])
    assert feedback["previous_response"] == raw1 and feedback["errors"] == errors
    assert raw1 not in attempt3.content and raw2 in attempt3.content
    assert attempt3.metadata["base_sha256"] == base.sha256
    with pytest.raises(ValueError):
        render_correction(attempt2, raw2, errors, 2)
    with pytest.raises(ValueError):
        render_correction(base, raw2, [], 2)


def test_target_placeholder_text_is_not_recursively_rendered():
    """Preserve braces, quotes, Unicode and template-looking target text exactly once.

    Args: none. Returns: None.
    Raises: AssertionError on recursive substitution. Side effects: artifact reads.
    """
    target = 'A definition with {{schema}}, "quotes", braces {}, ``` and é.\n'
    result = render_base_prompt(load_prompt_version("strict-minimal", ROOT), target, load_schema_bytes(ROOT))
    assert target in result.content
    assert result.metadata["target_definition_sha256"] == hashlib.sha256(target.encode()).hexdigest()
