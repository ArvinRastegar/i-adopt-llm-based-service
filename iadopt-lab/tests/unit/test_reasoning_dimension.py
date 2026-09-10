"""Reasoning as an experimental dimension, and the identity safety it requires.

The danger in adding `reasoning = enabled` is not that the new tasks are wrong. It is that
introducing the dimension might change the identity of the ~21,000 tasks already completed
and paid for, so a continuation silently re-runs them. These tests pin the property that
makes that impossible: adding a second reasoning profile ADDS configurations and leaves
every existing one byte-identical.

They drive the real configuration and planner rather than a hand-built fixture, because the
identity being protected is the one the real planner computes.
"""

import copy
from pathlib import Path

import pytest

from iadopt_lab.configuration import load_parameters, resolve_configuration
from iadopt_lab.planning import expand_campaign

ROOT = Path(__file__).resolve().parents[2]

DISABLED = {"mode": "disabled",
            "request_fields": {"chat_template_kwargs": {"enable_thinking": False}}}
ENABLED = {"mode": "enabled",
           "request_fields": {"chat_template_kwargs": {"enable_thinking": True}}}
SUFFIX_OFF = {"mode": "disabled", "request_fields": {}, "prompt_suffix": "\n/no_think"}
NOT_APPLICABLE = {"mode": "not_applicable", "request_fields": {}}

TARGETS = [{"variable_id": "urn:test:v1", "source_sha256": "a", "gold_sha256": "b",
            "demonstration_position": None}]
ARTIFACTS = {"corpus": "c", "population": "p", "prompts": "pr", "schema": "s",
             "scorer": "sc", "runtime": "r", "implementation": "i"}


@pytest.fixture(scope="module")
def base():
    """The real parameters, parsed. `Configuration` is byte-backed and re-parses `.data`
    on every access, so a variant must be written as bytes rather than mutated in place."""
    import yaml
    return yaml.safe_load((ROOT / "parameters.yml").read_text(encoding="utf-8"))


def _expand(base, profiles, tmp_path):
    """Expand a one-model, one-cell grid carrying exactly the given reasoning profiles."""
    import yaml

    data = copy.deepcopy(base)
    provider = data["campaign"]["providers"][0]
    models = [m for m in data["providers"][provider]["models"] if m["enabled"]]
    model = copy.deepcopy(models[0])
    model["reasoning_profiles"] = copy.deepcopy(profiles)
    model["capabilities"]["reasoning_control"] = profiles[0]["mode"] != "not_applicable"
    data["providers"][provider]["models"] = [model]
    grid = data["parameter_grid"]
    grid["prompt_variants"], grid["shot_counts"], grid["temperatures"] = ["strict-minimal"], [5], [0.5]

    target = tmp_path / "parameters.yml"
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    resolved = resolve_configuration(load_parameters(target))
    assert not resolved.issues, resolved.issues
    return expand_campaign(resolved, TARGETS, artifact_identities=ARTIFACTS, mode="synthetic")


def test_reasoning_mode_participates_in_configuration_identity(base, tmp_path):
    off = {c["configuration_id"] for c in _expand(base, [DISABLED], tmp_path)["configurations"]}
    on = {c["configuration_id"] for c in _expand(base, [ENABLED], tmp_path)["configurations"]}
    assert off and on
    assert off.isdisjoint(on), "reasoning mode does not change configuration identity"


def test_adding_the_enabled_profile_leaves_existing_identities_untouched(base, tmp_path):
    """The property protecting ~21,000 completed, paid tasks from being re-run."""
    before = {c["configuration_id"] for c in _expand(base, [DISABLED], tmp_path)["configurations"]}
    after = {c["configuration_id"] for c in _expand(base, [DISABLED, ENABLED], tmp_path)["configurations"]}
    assert before < after, "existing disabled configurations changed identity"
    assert len(after) == 2 * len(before), "adding one profile should exactly double the grid"


def test_the_off_mechanism_is_part_of_identity_too(base, tmp_path):
    """qwen3-32b disables reasoning by prompt suffix, qwen3-8b by request field.

    Treating those as one configuration would merge results produced by different requests.
    """
    field_off = {c["configuration_id"] for c in _expand(base, [DISABLED], tmp_path)["configurations"]}
    prompt_off = {c["configuration_id"] for c in _expand(base, [SUFFIX_OFF], tmp_path)["configurations"]}
    assert field_off.isdisjoint(prompt_off)


def test_a_model_that_cannot_reason_is_not_duplicated(base, tmp_path):
    plan = _expand(base, [NOT_APPLICABLE], tmp_path)
    assert len(plan["configurations"]) == 1
    assert plan["configurations"][0]["reasoning_mode"] == "not_applicable"


def test_every_run_carries_its_reasoning_mode_and_suffix(base, tmp_path):
    """Reporting can only separate the two arms if the run records the mode."""
    plan = _expand(base, [DISABLED, ENABLED], tmp_path)
    assert {run["reasoning_mode"] for run in plan["runs"]} == {"disabled", "enabled"}
    assert all("reasoning_prompt_suffix" in run for run in plan["runs"])


def test_task_count_scales_exactly_with_the_reasoning_dimension(base, tmp_path):
    """Guards against the accidental multiplication this continuation is most at risk of."""
    one = _expand(base, [DISABLED], tmp_path)
    two = _expand(base, [DISABLED, ENABLED], tmp_path)
    assert two["counts"]["tasks"] == 2 * one["counts"]["tasks"]
