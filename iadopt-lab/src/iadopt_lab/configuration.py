"""Strict draft loading, selected-provider resolution, and private runtime settings."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .canonical import canonical_json_bytes, sha256_bytes
from .domain import LabError


class ConfigurationError(LabError):
    """Invalid configuration; messages contain paths/reasons, never rejected values."""


class _UniqueLoader(yaml.SafeLoader):
    """Safe YAML without duplicate mappings; aliases are rejected before loading."""


def _mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    """Build a mapping while rejecting duplicate keys and non-string keys.

    Args: loader: Safe YAML parser; node: mapping node; deep: recursive construction flag.
    Returns: One dictionary of parsed values.
    Raises: ConfigurationError for duplicate/non-string keys.
    Side Effects: None beyond parser-local state.
    """
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ConfigurationError("YAML has duplicate or non-string mapping keys")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@dataclass(frozen=True)
class Configuration:
    """Byte-backed snapshot; callers receive independent parsed dictionaries."""

    canonical_bytes: bytes
    original_bytes: bytes
    issues: tuple[str, ...] = ()

    @property
    def data(self) -> dict[str, Any]:
        """Return an independent JSON copy of this snapshot.

        Args: None beyond the snapshot.
        Returns: Configuration dictionary whose mutation cannot change this object.
        Raises: JSONDecodeError only if constructed with corrupt bytes.
        Side Effects: None.
        """
        return json.loads(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Hash the canonical selected snapshot.

        Args: None.
        Returns: Deterministic SHA-256 hex digest.
        Raises: TypeError only for incorrectly constructed non-byte state.
        Side Effects: None.
        """
        return sha256_bytes(self.canonical_bytes)


def load_parameters(path: str | Path, schema_path: str | Path | None = None) -> Configuration:
    """Read and validate an explicit draft YAML file without resolving unknown live facts.

    Args: path: Parameter file; schema_path: explicit schema or the packaged project schema.
    Returns: Byte-backed original configuration with valid finite JSON primitives.
    Raises: OSError for unavailable files; ConfigurationError for unsafe YAML/schema violations.
    Side Effects: Reads only the two selected local files; no credentials/database/network.
    """
    original = Path(path).read_bytes()
    if len(original) > 2_000_000:
        raise ConfigurationError("Parameter YAML exceeds the documented 2 MB limit")
    try:
        if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(original)):
            raise ConfigurationError("YAML aliases are unsupported; use explicit values")
        raw = yaml.load(original, Loader=_UniqueLoader)
    except (yaml.YAMLError, UnicodeError):
        raise ConfigurationError("Invalid parameter YAML syntax") from None
    schema_file = Path(schema_path) if schema_path else Path(__file__).resolve().parents[2] / "schemas/parameters.schema.json"
    schema = json.loads(schema_file.read_bytes())
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda e: str(list(e.path)))
    if errors:
        locations = ["/" + "/".join(map(str, error.path)) + ": " + str(error.validator) for error in errors]
        raise ConfigurationError("Parameter schema violations: " + "; ".join(locations))
    try:
        canonical = canonical_json_bytes(raw)
    except (ValueError, TypeError):
        raise ConfigurationError("Configuration contains unsupported/non-finite values") from None
    return Configuration(canonical, original)


def resolve_configuration(raw: Configuration, environment_capabilities: Mapping | None = None) -> Configuration:
    """Canonicalize only selected provider/model entries and report missing planning facts.

    Args: raw: Schema-valid snapshot; environment_capabilities: optional injected non-secret facts.
    Returns: Selected immutable snapshot and all plan-readiness issue paths.
    Raises: ConfigurationError for ownership/duplicate/model/mode inconsistencies.
    Side Effects: None; no model capability is inferred or fetched.
    """
    data = raw.data
    issues = []
    selected = sorted(data["campaign"]["providers"])
    data["campaign"]["providers"] = selected
    data["providers"] = {name: data["providers"][name] for name in selected}
    for name in ("prompt_variants", "shot_counts", "temperatures"):
        data["parameter_grid"][name] = sorted(data["parameter_grid"][name])
    for name, provider in data["providers"].items():
        if provider["adapter"] != f"{name}_chat_completions" or provider["api_key_env"] != f"{name.upper()}_API_KEY":
            raise ConfigurationError("Provider adapter/credential ownership mismatch")
        expected_base = "default_base_url" if name == "psnc" else "base_url"
        if not provider.get(expected_base):
            raise ConfigurationError("Selected provider is missing its explicit base URL")
        entries = provider["models"]
        ids = [entry["id"] for entry in entries if entry["id"] is not None]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("Duplicate model ID within provider")
        provider["models"] = sorted((entry for entry in entries if entry["enabled"]), key=lambda m: m["id"] or "")
        if not provider["models"]:
            issues.append(f"providers.{name}.models: no enabled models")
        for model in provider["models"]:
            prefix = f"providers.{name}.models.{model['id']}"
            if model["id"] is None:
                issues.append(prefix + ": concrete model ID required")
            caps = model["capabilities"]
            if caps is None:
                issues.append(prefix + ": capability evidence missing")
                continue
            modes = [profile["mode"] for profile in model["reasoning_profiles"]]
            # D-033: capabilities declare what the model can do; reasoning_profiles declares
            # which of those modes this campaign actually tests. A non-empty subset is a
            # deliberate scientific scope, not a disagreement. An empty list, a duplicate,
            # or a mode the declared capability cannot support remains an error, so a
            # reasoning profile can never be asserted for a model that has no such control.
            allowed = {"disabled", "enabled"} if caps["reasoning_control"] else {"not_applicable"}
            if not modes or len(modes) != len(set(modes)) or not set(modes) <= allowed:
                raise ConfigurationError("Reasoning profiles disagree with declared capabilities")
            for profile in model["reasoning_profiles"]:
                if profile["mode"] == "not_applicable" and profile["request_fields"]:
                    raise ConfigurationError("Not-applicable reasoning must send no native fields")
                if profile["mode"] != "not_applicable" and not profile["request_fields"]:
                    raise ConfigurationError("Both controlled reasoning mappings must be explicit")
                from .providers.base import NATIVE_FIELDS
                if set(profile["request_fields"]) - NATIVE_FIELDS:
                    raise ConfigurationError("Reasoning profile contains unsupported or overriding fields")
            if len(model["reasoning_profiles"]) == 2 and model["reasoning_profiles"][0]["request_fields"] == model["reasoning_profiles"][1]["request_fields"]:
                raise ConfigurationError("Enabled and disabled reasoning mappings must differ")
            model["reasoning_profiles"].sort(key=lambda p: p["mode"])
            if not caps["temperature"]:
                issues.append(prefix + ": temperature grid unsupported")
            if caps.get("top_p") is not True:
                issues.append(prefix + ": top_p capability evidence missing or unsupported")
            limit = data["parameter_grid"]["max_output_tokens"]
            if limit is not None and caps.get("max_output_tokens") is not None and limit > caps["max_output_tokens"]:
                issues.append(prefix + ": configured output limit exceeds model capability")
            if "temperature_values" in caps and not set(data["parameter_grid"]["temperatures"]) <= set(caps["temperature_values"]):
                issues.append(prefix + ": unsupported grid temperature")
            if model["context_window_tokens"] is None:
                issues.append(prefix + ": context window missing")
            if not caps.get("evidence"):
                issues.append(prefix + ": capability source/date missing")
    for name in ("top_p", "max_output_tokens"):
        if data["parameter_grid"][name] is None:
            issues.append("parameter_grid." + name + ": value not frozen")
    if data["execution"]["heartbeat_seconds"] >= data["execution"]["task_lease_seconds"]:
        raise ConfigurationError("Heartbeat interval must be shorter than the task lease")
    return Configuration(canonical_json_bytes(data), raw.original_bytes, tuple(issues))


def validate_live_readiness(resolved: Configuration, facts: Mapping | None = None) -> dict:
    """Evaluate all live gates against injected evidence without making a request.

    Args: resolved: Selected configuration; facts: checked artifacts/DB/credentials/estimate/approval.
    Returns: Safe report containing readiness, issue list, snapshot hash and live mode.
    Raises: ConfigurationError for malformed injected facts.
    Side Effects: None; facts must already be read by explicit boundary services.
    """
    facts = facts or {}
    data = resolved.data
    issues = list(resolved.issues)
    if not data["campaign"]["live_calls_enabled"]:
        issues.append("Live execution is disabled")
    for gate in ("artifacts_verified", "database_verified", "credentials_present", "estimate_disclosed", "live_authorized"):
        if facts.get(gate) is not True:
            issues.append(gate + ": required before live dispatch")
    for provider, profile in data["providers"].items():
        for name in ("timeout_seconds", "max_concurrency", "requests_per_minute", "tokens_per_minute"):
            if profile[name] is None:
                issues.append(f"providers.{provider}.{name}: not frozen")
    for name, value in {"worker_count": data["execution"]["worker_count"],
                        "price_card_manifest": data["cost_accounting"]["price_card_manifest"],
                        "similarity_model_revision": data["evaluation"]["similarity_model_revision"]}.items():
        if value is None:
            issues.append(name + ": not frozen")
    return {"ready": not issues, "issues": issues, "configuration_sha256": resolved.sha256,
            "live_calls_enabled": data["campaign"]["live_calls_enabled"]}


class RuntimeSecrets:
    """Private process-local settings; representation and serialization never expose values."""

    def __init__(self, values: Mapping[str, str]) -> None:
        """Copy already allowlisted literal secret values.

        Args: values: Name/value mapping selected by the loader.
        Returns: None.
        Raises: TypeError for a non-mapping.
        Side Effects: Private in-memory allocation only.
        """
        self.__values = dict(values)

    def get(self, name: str) -> str | None:
        """Resolve one allowed setting for an explicit connection boundary.

        Args: name: Previously allowlisted setting name.
        Returns: Literal value or None; callers must never log returned secrets.
        Raises: None.
        Side Effects: None.
        """
        return self.__values.get(name) or None

    def __repr__(self) -> str:
        """Return a secret-free description.

        Args: None.
        Returns: Fixed redacted text.
        Raises: None.
        Side Effects: None.
        """
        return "RuntimeSecrets(<redacted>)"

    def __reduce__(self) -> Any:
        """Prevent accidental pickle serialization of credentials.

        Args: None.
        Returns: Never returns.
        Raises: TypeError always.
        Side Effects: None.
        """
        raise TypeError("Runtime secrets cannot be serialized")


def load_runtime_secrets(env_file: str | Path | None, required_names: list[str],
                         process_environment: Mapping[str, str]) -> RuntimeSecrets:
    """Load allowlisted literal environment settings without executing or interpolating text.

    Args: env_file: Explicit optional file; required_names: allowlist; process_environment: injected environment.
    Returns: Non-serializable private resolver, with process values taking precedence.
    Raises: OSError for unavailable explicit file; ConfigurationError for malformed selected entries.
    Side Effects: Reads only the selected file; never changes the environment or persists values.
    """
    if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) for name in required_names):
        raise ConfigurationError("Invalid runtime setting allowlist")
    allowed = set(required_names)
    values = {}
    if env_file is not None:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry.startswith("export "):
                entry = entry[7:].lstrip()
            key, separator, value = entry.partition("=")
            if not separator or key.strip() not in allowed:
                continue
            try:
                parts = shlex.split(value, comments=True, posix=True)
            except ValueError:
                raise ConfigurationError("Malformed selected runtime setting") from None
            if len(parts) > 1:
                raise ConfigurationError("Selected runtime setting needs literal quoting")
            if key.strip() in values:
                raise ConfigurationError("Duplicate selected runtime setting")
            values[key.strip()] = parts[0] if parts else ""
    values.update({key: value for key, value in process_environment.items() if key in allowed})
    return RuntimeSecrets(values)
