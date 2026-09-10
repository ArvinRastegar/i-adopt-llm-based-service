"""Pure, provider-owned grid expansion and explicitly labelled offline fixtures."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

from .canonical import canonical_json_bytes, content_hash
from .configuration import Configuration, ConfigurationError, resolve_configuration


def expand_campaign(configuration: Configuration, targets: list[dict] | tuple[dict, ...],
                    *, artifact_identities: dict, mode: str = "live") -> dict[str, Any]:
    """Expand the selected-provider union into deterministic configurations, runs and tasks.

    Args:
        configuration: Resolved, plan-ready snapshot; live authorization is not required.
        targets: Canonical non-demonstration records. Real plans require exactly 97.
        artifact_identities: Verified corpus, population, prompts, schema, scorer and runtime hashes.
        mode: ``live`` for the real population or explicitly labelled ``synthetic`` fixtures.
    Returns:
        JSON-compatible plan with every owned run/task, exact call counts and SHA-256 identity.
    Raises:
        ConfigurationError: Unresolved settings, wrong/duplicate population or missing artifacts.
    Side Effects:
        None. Input order is canonicalized; no credential, database or network access occurs.
    """
    if configuration.issues:
        raise ConfigurationError("Cannot freeze a draft plan: " + "; ".join(configuration.issues))
    if mode not in {"live", "synthetic"}:
        raise ConfigurationError("Unknown execution mode")
    data = configuration.data
    if not targets or (mode == "live" and len(targets) != data["evaluation_population"]["expected_variable_count"]):
        raise ConfigurationError("Plan population does not match its declared execution mode")
    ids = [row["variable_id"] for row in targets]
    if len(set(ids)) != len(ids) or any(row.get("demonstration_position") for row in targets):
        raise ConfigurationError("Duplicate variable or demonstration in evaluation population")
    required = {"corpus", "population", "prompts", "schema", "scorer", "runtime"}
    if not required <= set(artifact_identities) or any(not artifact_identities[key] for key in required):
        raise ConfigurationError("Plan needs verified corpus/population/prompt/schema/scorer/runtime identities")
    grid = data["parameter_grid"]
    population = sorted(ids)
    scope = {"snapshot": configuration.sha256, "mode": mode, "artifacts": artifact_identities,
             "population": population}
    scope_hash = content_hash(scope)
    runs, tasks, configurations = [], [], []
    for provider_name in sorted(data["providers"]):
        for model in data["providers"][provider_name]["models"]:
            for reasoning, prompt, shots, temperature in product(
                    model["reasoning_profiles"], sorted(grid["prompt_variants"]),
                    sorted(grid["shot_counts"]), sorted(grid["temperatures"])):
                parameters = {"provider": provider_name, "model_id": model["id"],
                    "model_revision": model["revision"], "reasoning_mode": reasoning["mode"],
                    "reasoning_fields": reasoning["request_fields"],
                    "reasoning_prompt_suffix": reasoning.get("prompt_suffix"), "prompt_variant": prompt,
                    "shot_count": shots, "temperature": temperature, "top_p": grid["top_p"],
                    "max_output_tokens": grid["max_output_tokens"], "artifacts": artifact_identities,
                    "generation_policy": data["generation"], "evaluation_policy": data["evaluation"],
                    "mode": mode}
                configuration_id = content_hash(parameters)
                configurations.append({"configuration_id": configuration_id, **parameters})
                repetitions = grid["repetitions"]["temperature_zero" if temperature == 0 else "nonzero_temperature"]
                for repetition in range(1, repetitions + 1):
                    fingerprint = content_hash({"scope": scope_hash, "configuration": configuration_id,
                                                "repetition": repetition})
                    run = {"id": fingerprint, "run_id": fingerprint, "sha256": fingerprint,
                           "configuration_id": configuration_id, "repetition": repetition,
                           **parameters, "configuration": parameters}
                    runs.append(run)
                    for variable_id in population:
                        identity = {"run_id": fingerprint, "variable_id": variable_id}
                        tasks.append({**identity, "id": content_hash(identity), "provider": provider_name,
                                      "model_id": model["id"]})
    if not runs or len({row["id"] for row in runs}) != len(runs):
        raise ConfigurationError("Empty grid or duplicate run identity")
    runs.sort(key=lambda row: row["id"])
    tasks.sort(key=lambda row: row["id"])
    configurations.sort(key=lambda row: row["configuration_id"])
    counts = Counter((task["provider"], task["model_id"]) for task in tasks)
    maximum_attempts = data["generation"]["max_generation_attempts_per_task"]
    plan = {"version": "experiment-plan-v1", "mode": mode, "scope_sha256": scope_hash,
        "configuration_sha256": configuration.sha256, "artifact_identities": artifact_identities,
        "population": population, "population_size": len(population),
        "configurations": configurations, "runs": runs, "tasks": tasks,
        "counts": {"configurations": len(configurations), "runs": len(runs), "tasks": len(tasks),
                   "initial_calls": len(tasks), "maximum_calls": len(tasks) * maximum_attempts,
                   "by_model": [{"provider": provider, "model_id": model, "initial_calls": count,
                                  "maximum_calls": count * maximum_attempts}
                                 for (provider, model), count in sorted(counts.items())]},
        "max_attempts_per_task": maximum_attempts}
    return {**plan, "sha256": content_hash(plan)}


def synthetic_configuration(original: Configuration) -> Configuration:
    """Create a separate tiny, explicit test snapshot without modifying live parameters.

    Args:
        original: Schema-valid user YAML snapshot supplying accepted scientific policies.
    Returns:
        Resolved snapshot with two mock-owned model IDs, three target fixtures and no live rights.
    Raises:
        ConfigurationError: Original policy is incompatible with fixture resolution.
    Side Effects:
        None. The original file and snapshot are unchanged; no real model is simulated by name.
    """
    data = original.data
    # Mirror whichever providers the base configuration actually carries; hardcoding
    # both breaks a deliberately single-provider campaign.
    data["campaign"].update(name="SYNTHETIC offline acceptance",
                            providers=sorted(data["providers"]), live_calls_enabled=False)
    data["parameter_grid"].update(prompt_variants=["strict-minimal"], shot_counts=[0], temperatures=[0.0],
                                 top_p=1.0, max_output_tokens=512,
                                 repetitions={"temperature_zero": 1, "nonzero_temperature": 1})
    for provider in data["providers"].values():
        provider.update(timeout_seconds=10, max_concurrency=2, requests_per_minute=60000, tokens_per_minute=100000000)
        provider["models"] = [{"id": "synthetic-fixture-v1", "display_name": "SYNTHETIC, no model call",
            "enabled": True, "revision": "fixture-v1", "context_window_tokens": 100000,
            "capabilities": {"temperature": True, "top_p": True, "seed": False,
                "structured_output": False, "reasoning_control": False,
                "evidence": {"source": "local fixture, not provider capability evidence", "verified_at": "fixture-v1"}},
            "reasoning_profiles": [{"mode": "not_applicable", "request_fields": {}}]}]
    data["execution"]["worker_count"] = 4
    data["evaluation"]["similarity_model_revision"] = "synthetic-equality-only-v1"
    return resolve_configuration(Configuration(canonical_json_bytes(data), original.original_bytes))
