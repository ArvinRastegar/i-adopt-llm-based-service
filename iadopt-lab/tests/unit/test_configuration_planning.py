"""Strict YAML, private credentials and provider-owned deterministic grid tests."""

import copy
import pickle
from pathlib import Path

import pytest
import yaml

from iadopt_lab.canonical import canonical_json_bytes
from iadopt_lab.configuration import (
    Configuration,
    ConfigurationError,
    load_parameters,
    load_runtime_secrets,
    resolve_configuration,
    validate_live_readiness,
)
from iadopt_lab.planning import expand_campaign, synthetic_configuration

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = {key: key + "-fixture-sha256" for key in ("corpus", "population", "prompts", "schema", "scorer", "runtime")}


def _snapshot(data):
    """Build an independent immutable in-memory test configuration.

    Args: already schema-shaped fixture dictionary. Returns: Configuration.
    Raises: canonicalization errors. Side effects: none.
    """
    raw = canonical_json_bytes(data)
    return Configuration(raw, raw)


def _targets(count=97):
    """Generate unique explicitly artificial population identifiers.

    Args: fixture population size. Returns: list of non-demo variable dictionaries.
    Raises: no content errors. Side effects: none.
    """
    return [{"variable_id": f"fixture-variable-{number:03d}", "demonstration_position": None} for number in range(count)]


def _full_grid_configuration(providers, controlled):
    """Apply declared mock capabilities while preserving the user's scientific grid.

    Args: selected providers and whether fixture models expose enabled/disabled control.
    Returns: resolved six-model-or-selected-subset fixture configuration.
    Raises: configuration validation errors. Side effects: reads public parameters only.
    """
    original = load_parameters(ROOT / "parameters.yml")
    data = original.data
    fixture = synthetic_configuration(original).data
    template = next(iter(fixture["providers"].values()))["models"][0]
    data["campaign"]["providers"] = list(providers)
    data["parameter_grid"]["top_p"] = 1.0
    data["parameter_grid"]["max_output_tokens"] = 512
    # These tests assert the documented D-029 campaign grid, so they declare it here
    # rather than inheriting whatever scope parameters.yml currently carries. A live
    # campaign may legitimately be narrowed to one configuration between runs.
    data["parameter_grid"]["prompt_variants"] = [
        "strict-minimal", "constraint-decomposition", "matrix-decomposition"]
    data["parameter_grid"]["shot_counts"] = [0, 1, 3, 5]
    data["parameter_grid"]["temperatures"] = [0.0, 0.5, 1.0, 2.0]
    for name in providers:
        for model in data["providers"][name]["models"]:
            model["enabled"] = True
            model.update(capabilities=copy.deepcopy(template["capabilities"]),
                         context_window_tokens=100000, revision="fixture-revision-v1",
                         reasoning_profiles=copy.deepcopy(template["reasoning_profiles"]))
            model["capabilities"]["reasoning_control"] = controlled
            if controlled:
                model["reasoning_profiles"] = [
                    {"mode": "disabled", "request_fields": {"enable_thinking": False}},
                    {"mode": "enabled", "request_fields": {"enable_thinking": True}},
                ]
    return resolve_configuration(_snapshot(data))


# 3 prompts x 4 shot counts x 4 temperatures = 48 configurations per model and reasoning
# profile, over 97 variables. The catalog now holds 3 PSNC and 5 OpenRouter models, so the
# counts below are 48 x 97 x models, doubled where every model exposes controlled reasoning.
# OpenRouter holds 6 since z-ai/glm-5.2 was added to carry GLM's reasoning arm (D-045).
# Hard-coded on purpose: adding a model should force a deliberate update here rather than
# quietly changing the size of the documented grid.
_PSNC_MODELS, _OPENROUTER_MODELS, _CELL = 3, 6, 48 * 97


@pytest.mark.parametrize("providers,controlled,expected_calls", [
    (["psnc"], False, _CELL * _PSNC_MODELS),
    (["openrouter"], False, _CELL * _OPENROUTER_MODELS),
    (["psnc", "openrouter"], False, _CELL * (_PSNC_MODELS + _OPENROUTER_MODELS)),
    (["psnc"], True, _CELL * _PSNC_MODELS * 2),
    (["openrouter"], True, _CELL * _OPENROUTER_MODELS * 2),
    (["psnc", "openrouter"], True, _CELL * (_PSNC_MODELS + _OPENROUTER_MODELS) * 2),
])
def test_current_one_repetition_grid_counts(providers, controlled, expected_calls):
    """Count each selected provider's own models with singleton runs at every temperature.

    Args: provider selection, controlled reasoning fixture and hand-calculated calls.
    Returns: None. Raises: AssertionError on grid/call multiplication regression.
    Side effects: reads non-secret configuration; no model or database requests.
    """
    resolved = _full_grid_configuration(providers, controlled)
    plan = expand_campaign(resolved, _targets(), artifact_identities=ARTIFACTS)
    assert plan["counts"]["initial_calls"] == expected_calls
    assert plan["counts"]["maximum_calls"] == expected_calls * 3
    assert {run["repetition"] for run in plan["runs"]} == {1}
    assert {run["temperature"] for run in plan["runs"]} == {0, 0.5, 1, 2}
    assert len(plan["configurations"]) == len(plan["runs"])
    assert {task["provider"] for task in plan["tasks"]} == set(providers)
    for task in plan["tasks"]:
        owned = {model["id"] for model in resolved.data["providers"][task["provider"]]["models"]}
        assert task["model_id"] in owned


def test_draft_keeps_all_models_and_blocks_plan_without_capabilities():
    """Retain unresolved selected models without inventing reasoning support or dropping work.

    Args: none. Returns: None.
    Raises: AssertionError on accidental readiness. Side effects: public YAML read only.
    """
    original = load_parameters(ROOT / "parameters.yml")
    # Build a draft rather than assuming the live file is one: a frozen campaign
    # configuration is a legitimate state between runs, and this test is about how an
    # unresolved draft behaves, not about what parameters.yml currently selects.
    data = original.data
    data["campaign"]["providers"] = ["psnc", "openrouter"]
    for profile in data["providers"].values():
        for model in profile["models"]:
            model.update(enabled=True, capabilities=None, reasoning_profiles=[])
    resolved = resolve_configuration(_snapshot(data))
    # 3 PSNC + 5 OpenRouter. Every catalog entry survives resolution when enabled, and
    # each one lacking capabilities contributes at least one blocking issue.
    catalog = sum(len(profile["models"]) for profile in resolved.data["providers"].values())
    assert catalog == _PSNC_MODELS + _OPENROUTER_MODELS
    assert len(resolved.issues) >= catalog
    assert original.data["parameter_grid"]["repetitions"] == {"temperature_zero": 1, "nonzero_temperature": 1}
    with pytest.raises(ConfigurationError, match="draft"):
        expand_campaign(resolved, _targets(), artifact_identities=ARTIFACTS)


def test_provider_order_and_inactive_catalog_do_not_change_selected_identity():
    """Canonicalize selection order and ignore inactive catalog changes in scientific identity.

    Args: none. Returns: None.
    Raises: AssertionError on order-sensitive hashes. Side effects: configuration reads only.
    """
    both = _full_grid_configuration(["psnc", "openrouter"], False)
    reversed_data = both.data
    reversed_data["campaign"]["providers"].reverse()
    assert resolve_configuration(_snapshot(reversed_data)).sha256 == both.sha256
    original = load_parameters(ROOT / "parameters.yml").data
    original["campaign"]["providers"] = ["psnc"]
    selected = resolve_configuration(_snapshot(original))
    original["providers"]["openrouter"]["models"][0]["id"] = "inactive-catalog-edit"
    assert resolve_configuration(_snapshot(original)).sha256 == selected.sha256


def test_same_model_name_at_two_providers_is_two_owned_runs():
    """Allow a native model name reused across providers without merging execution identity.

    Args: none. Returns: None.
    Raises: AssertionError on provider identity collision. Side effects: YAML read only.
    """
    config = synthetic_configuration(load_parameters(ROOT / "parameters.yml"))
    plan = expand_campaign(config, _targets(3), artifact_identities=ARTIFACTS, mode="synthetic")
    assert len(plan["runs"]) == 2 and len(plan["tasks"]) == 6
    assert {run["model_id"] for run in plan["runs"]} == {"synthetic-fixture-v1"}
    assert len({run["configuration_id"] for run in plan["runs"]}) == 2
    assert plan["sha256"] == expand_campaign(config, list(reversed(_targets(3))), artifact_identities=ARTIFACTS, mode="synthetic")["sha256"]


@pytest.mark.parametrize("modification", ["no_modes", "foreign_mode", "identical_fields", "wrong_native", "not_applicable_fields", "duplicate_model"])
def test_invalid_reasoning_or_model_ownership(modification):
    """Reject inconsistent mode mappings and duplicate IDs before provider dispatch.

    Args: invalid fixture mutation name. Returns: None.
    Raises: AssertionError on accepted contradiction. Side effects: YAML read only.
    """
    data = _full_grid_configuration(["psnc"], True).data
    model = data["providers"]["psnc"]["models"][0]
    if modification == "no_modes":
        # D-033 permits a declared subset of the capable modes, but never an empty one.
        model["reasoning_profiles"].clear()
    elif modification == "foreign_mode":
        # A mode the declared capability cannot support is still a contradiction.
        model["reasoning_profiles"] = [{"mode": "not_applicable", "request_fields": {}}]
    elif modification == "identical_fields":
        model["reasoning_profiles"][1]["request_fields"] = copy.deepcopy(model["reasoning_profiles"][0]["request_fields"])
    elif modification == "wrong_native":
        model["reasoning_profiles"][0]["request_fields"] = {"model": "override"}
    elif modification == "not_applicable_fields":
        model["capabilities"]["reasoning_control"] = False
        model["reasoning_profiles"] = [{"mode": "not_applicable", "request_fields": {"enable_thinking": False}}]
    else:
        data["providers"]["psnc"]["models"].append(copy.deepcopy(model))
    with pytest.raises(ConfigurationError):
        resolve_configuration(_snapshot(data))


@pytest.mark.parametrize("payload", [
    "a: 1\na: 2\n", "a: &a [1]\nb: *a\n", "a: !!python/object:builtins.str {}\n",
    "a: !!custom value\n", "[not, a, mapping]\n", "a: .nan\n",
])
def test_unsafe_and_duplicate_yaml_rejected(tmp_path, payload):
    """Reject unsafe YAML constructs before creating a usable configuration.

    Args: isolated temporary directory and invalid YAML text. Returns: None.
    Raises: AssertionError on accidental acceptance. Side effects: temporary test file only.
    """
    path = tmp_path / "invalid.yml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_parameters(path)


@pytest.mark.parametrize("selection", [[], ["psnc", "psnc"], ["other"]])
def test_invalid_provider_selection_schema(tmp_path, selection):
    """Reject empty, repeated and unknown provider selections through the public loader.

    Args: temporary directory and invalid provider list. Returns: None.
    Raises: AssertionError on schema regression. Side effects: temporary fixture file only.
    """
    data = load_parameters(ROOT / "parameters.yml").data
    data["campaign"]["providers"] = selection
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_parameters(path)


def test_unknown_parameter_and_secret_values_are_not_echoed(tmp_path):
    """Reject unknown secret-bearing configuration without reproducing the rejected value.

    Args: isolated temporary directory. Returns: None.
    Raises: AssertionError if rejection leaks fake secret text. Side effects: temporary fixture only.
    """
    data = load_parameters(ROOT / "parameters.yml").data
    marker = "not-a-real-secret-do-not-echo"
    data["api_key"] = marker
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError) as error:
        load_parameters(path)
    assert marker not in str(error.value)


def test_literal_secret_loading_allowlist_precedence_and_nonserialization(tmp_path):
    """Load only selected literal settings without interpolation, shell execution or serialization.

    Args: isolated temporary directory. Returns: None.
    Raises: AssertionError on literal/allowlist/privacy regression. Side effects: temporary fake env file.
    """
    path = tmp_path / "fixture.env"
    path.write_text("OPENROUTER_API_KEY='${UNEXPANDED}'\nPSNC_API_KEY='$(echo not-executed)'\nUNRELATED='not-imported'\n", encoding="utf-8")
    environment = {"OPENROUTER_API_KEY": "process-wins", "UNRELATED": "ignored"}
    secrets = load_runtime_secrets(path, ["OPENROUTER_API_KEY", "PSNC_API_KEY"], environment)
    assert secrets.get("OPENROUTER_API_KEY") == "process-wins"
    assert secrets.get("PSNC_API_KEY") == "$(echo not-executed)"
    assert secrets.get("UNRELATED") is None
    assert "process-wins" not in repr(secrets)
    assert environment == {"OPENROUTER_API_KEY": "process-wins", "UNRELATED": "ignored"}
    with pytest.raises(TypeError):
        pickle.dumps(secrets)


def test_secret_loader_errors_are_private(tmp_path):
    """Reject malformed/duplicate selected literals with value-free diagnostics.

    Args: temporary fixture directory. Returns: None.
    Raises: AssertionError on privacy regression. Side effects: fake env fixture writes.
    """
    path = tmp_path / "fixture.env"
    for text in ("PSNC_API_KEY='PRIVATE_UNCLOSED", "PSNC_API_KEY=ONE\nPSNC_API_KEY=TWO\n"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigurationError) as error:
            load_runtime_secrets(path, ["PSNC_API_KEY"], {})
        assert "PRIVATE_UNCLOSED" not in str(error.value)
        assert "ONE" not in str(error.value) and "TWO" not in str(error.value)


def test_live_readiness_requires_separate_disclosure_and_authorization():
    """Keep cost disclosure and execution authority separate even with complete injected facts.

    Args: none. Returns: None.
    Raises: AssertionError for a missing live gate. Side effects: YAML read; no dispatch.
    """
    data = synthetic_configuration(load_parameters(ROOT / "parameters.yml")).data
    data["campaign"]["live_calls_enabled"] = True
    data["cost_accounting"]["price_card_manifest"] = "fixture-not-a-real-live-price-card"
    resolved = resolve_configuration(_snapshot(data))
    gates = {key: True for key in ("artifacts_verified", "database_verified", "credentials_present", "estimate_disclosed", "live_authorized")}
    for missing in ("estimate_disclosed", "live_authorized"):
        facts = {**gates, missing: False}
        report = validate_live_readiness(resolved, facts)
        assert not report["ready"] and any(missing in issue for issue in report["issues"])
    assert validate_live_readiness(resolved, gates)["ready"]


def test_population_and_artifact_guards():
    """Reject incomplete/duplicate/demo-containing live populations and missing artifact evidence.

    Args: none. Returns: None.
    Raises: AssertionError if invalid plan scope is accepted. Side effects: YAML read only.
    """
    config = _full_grid_configuration(["psnc"], False)
    cases = [_targets(96), _targets(96) + [_targets(1)[0]], [{"variable_id": "demo", "demonstration_position": 1}] + _targets(96)]
    for targets in cases:
        with pytest.raises(ConfigurationError):
            expand_campaign(config, targets, artifact_identities=ARTIFACTS)
    with pytest.raises(ConfigurationError):
        expand_campaign(config, _targets(), artifact_identities={})
