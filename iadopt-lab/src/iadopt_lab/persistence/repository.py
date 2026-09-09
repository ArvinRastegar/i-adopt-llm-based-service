"""PostgreSQL 16 registries, immutable evidence, fenced leases, and accounting.

The public API is synchronous and thread-safe. Each method borrows one connection
for one short transaction; callers must release it before any external model work.
Mappings are validated at each boundary and retained alongside normalized columns.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

_NAMESPACE = uuid.UUID("16b36d4c-bd19-5788-bcd0-6c00e58d2140")
_SCORER = "january-derived-member-credit-v1"
_FIELDS = {
    "hasStatisticalModifier",
    "hasProperty",
    "hasObjectOfInterest",
    "hasMatrix",
    "hasContextObject",
    "hasConstraint",
}
_SECRET_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "password",
    "cookie",
    "set-cookie",
    "database_url",
    "dsn",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class PersistenceError(RuntimeError):
    """A typed invariant or admission failure with no secret-bearing payload."""


class EvidenceConflict(PersistenceError):
    """The same immutable identity was presented with different evidence."""


class StaleLeaseError(PersistenceError):
    """An expired or superseded lease attempted a mutable state change."""


class AttemptLimitError(PersistenceError):
    """An attempt was nonconsecutive or would exceed the total three-call limit."""


class BudgetError(PersistenceError):
    """An optional cost cap or required cost evidence prevents admission."""


class RateLimitError(PersistenceError):
    """A shared provider rate window needs a persisted pre-dispatch cooldown."""

    def __init__(self, retry_after_seconds: float) -> None:
        """Set a bounded nonnegative retry delay; expose no request or credential data.

        Input is seconds until the oldest admission expires. Output is a typed
        exception with retry_after_seconds; no attempt or other state is changed.
        """
        self.retry_after_seconds = max(0.001, float(retry_after_seconds))
        super().__init__("Provider request/token window has no admission capacity")


def _json_default(value: Any) -> Any:
    """Convert supported exact metadata values to JSON; reject other objects.

    Input is one value encountered by JSON encoding. Output is a lossless string
    or rational mapping. Raises TypeError for unsupported values; no side effects.
    """
    if isinstance(value, (Decimal, uuid.UUID)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    """Encode a JSON-compatible input deterministically as UTF-8 bytes.

    Reject non-finite floats and unsupported types. Returns canonical bytes;
    does not mutate the value or perform I/O.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    """Return SHA-256 of exact input bytes or canonical JSON; perform no I/O."""
    return hashlib.sha256(value if isinstance(value, bytes) else _canonical(value)).hexdigest()


def _id(kind: str, fingerprint: str) -> str:
    """Derive a stable UUID string from a record kind and fingerprint, without I/O."""
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{fingerprint}"))


def _clean(value: Any) -> Any:
    """Validate and normalize non-secret JSON evidence without changing semantics.

    Input is a mapping/list/scalar; output is its JSON-normalized copy. Prohibited
    credential keys and connection strings raise PersistenceError. Raw response
    bytes are stored separately and never passed through this metadata filter.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _SECRET_KEYS:
                raise PersistenceError("Credential-bearing evidence key is prohibited")
            _clean(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _clean(child)
    elif isinstance(value, str) and re.search(
        r"(?:postgres(?:ql)?://[^\s]+:[^\s]+@|Bearer\s+\S+)", value, re.I
    ):
        raise PersistenceError("Credential-bearing connection/header evidence is prohibited")
    return json.loads(_canonical(value))


def _require(record: Mapping[str, Any], *keys: str) -> None:
    """Require non-null fields in an input mapping; raise PersistenceError otherwise."""
    if not isinstance(record, Mapping) or any(
        key not in record or record[key] is None for key in keys
    ):
        raise PersistenceError("Required record fields are missing: " + ", ".join(keys))


def _money(value: Any, *, nullable: bool = False) -> Decimal | None:
    """Validate one nonnegative finite monetary value, preserving exact decimals.

    Return Decimal, or None only when nullable=True. Raise BudgetError for absent,
    negative, malformed, or non-finite values. No state is changed.
    """
    if value is None and nullable:
        return None
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise BudgetError("Invalid monetary amount") from exc
    if not result.is_finite() or result < 0:
        raise BudgetError("Monetary amounts must be finite and nonnegative")
    return result


def _cap(config: Mapping[str, Any], name: str) -> Decimal | None:
    """Read an optional named amount/currency object; return its validated cap."""
    value = config.get(name)
    return _money(value.get("amount") if isinstance(value, Mapping) else value, nullable=True)


def _row(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Copy a database row and stringify UUIDs; preserve bytes and exact numbers."""
    if record is None:
        return None
    return {
        key: str(value) if isinstance(value, uuid.UUID) else value for key, value in record.items()
    }


def _fraction(value: Any) -> Fraction:
    """Decode an exact rational metric receipt; reject invalid/non-finite values.

    Supports numerator/denominator mappings, Fraction, integers, and exact decimal
    strings. Floats are interpreted from their serialized decimal source receipt,
    never reconstructed from a rounded display field. Returns a reduced Fraction.
    """
    if isinstance(value, Mapping):
        if "numerator" in value and "denominator" in value:
            numerator, denominator = value["numerator"], value["denominator"]
        elif "n" in value and "d" in value:
            numerator, denominator = value["n"], value["d"]
        else:
            raise PersistenceError("Metric lacks an exact numerator/denominator receipt")
        n, d = Decimal(str(numerator)), Decimal(str(denominator))
        if (
            not n.is_finite()
            or not d.is_finite()
            or n != n.to_integral_value()
            or d != d.to_integral_value()
            or d <= 0
        ):
            raise PersistenceError("Invalid rational metric receipt")
        return Fraction(int(n), int(d))
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise PersistenceError("Invalid exact metric") from exc


def migrate(dsn: str, migrations_dir: str | Path | None = None) -> list[str]:
    """Apply forward, hash-verified SQL migrations to an explicit PostgreSQL 16 DSN.

    Inputs: a connection string kept only in memory and an optional migration
    directory. Output: names newly applied, empty on an idempotent repeat. Raises
    PersistenceError for major-version/hash drift; database errors roll back the
    migration. Side effects: schema/DDL writes under an advisory transaction lock.
    """
    directory = (
        Path(migrations_dir)
        if migrations_dir
        else Path(__file__).resolve().parents[3] / "migrations"
    )
    paths = sorted(directory.glob("[0-9]*.sql"))
    if not paths:
        raise PersistenceError("No forward migration files were found")
    applied: list[str] = []
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        if conn.info.server_version // 10000 != 16:
            raise PersistenceError("I-ADOPT Lab requires PostgreSQL major version 16")
        conn.execute("SELECT pg_advisory_xact_lock(1640216001)")
        conn.execute("CREATE SCHEMA IF NOT EXISTS iadopt_lab")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS iadopt_lab.schema_migration (name text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT clock_timestamp())"
        )
        existing = {
            r["name"]: r["sha256"]
            for r in conn.execute("SELECT name,sha256 FROM iadopt_lab.schema_migration")
        }
        for path in paths:
            content = path.read_bytes()
            digest = _hash(content)
            if path.name in existing:
                if existing[path.name] != digest:
                    raise EvidenceConflict("An applied migration's bytes have changed")
                continue
            conn.execute(content.decode("utf-8"), prepare=False)
            conn.execute(
                "INSERT INTO iadopt_lab.schema_migration(name,sha256) VALUES(%s,%s)",
                (path.name, digest),
            )
            applied.append(path.name)
    return applied


class Repository:
    """Thread-safe synchronous store for one dedicated PostgreSQL experiment DB."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8) -> None:
        """Open a bounded connection pool for an explicit DSN without persisting it.

        Inputs include positive pool bounds. Output is an initialized repository;
        raises connection/version errors. Side effects are database connections and
        a read-only PostgreSQL version/migration check, never schema creation.
        """
        self.pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "options": "-c search_path=iadopt_lab,public"},
            open=True,
        )
        self.pool.wait()
        try:
            with self.pool.connection() as conn:
                if conn.info.server_version // 10000 != 16:
                    raise PersistenceError("I-ADOPT Lab requires PostgreSQL major version 16")
                conn.execute("SELECT name FROM schema_migration LIMIT 1").fetchone()
        except Exception:
            self.pool.close()
            raise

    def __enter__(self) -> Repository:
        """Return this open repository for a context manager; no extra I/O occurs."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close pooled connections on context exit; never suppress an exception."""
        self.close()

    def close(self) -> None:
        """Close the repository pool; return None and preserve all durable evidence."""
        self.pool.close()

    def register_artifact(
        self, kind: str, content: bytes | str, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Persist exact protocol/source bytes and metadata idempotently.

        Inputs are a nonempty kind, bytes or UTF-8 text, and non-secret provenance.
        Output is the immutable artifact row. Same kind/hash with conflicting
        metadata raises EvidenceConflict. Only PostgreSQL evidence is written.
        """
        if not kind:
            raise PersistenceError("Artifact kind is required")
        raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        meta = _clean(metadata or {})
        sha, evidence_hash = (
            _hash(raw),
            _hash({"kind": kind, "content_hash": _hash(raw), "metadata": meta}),
        )
        identity = _id("artifact", f"{kind}:{sha}")
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO artifact(id,kind,sha256,content,metadata,evidence_hash) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (identity, kind, sha, raw, Jsonb(meta), evidence_hash),
            )
            row = conn.execute(
                "SELECT * FROM artifact WHERE kind=%s AND sha256=%s", (kind, sha)
            ).fetchone()
            if row["evidence_hash"] != evidence_hash or bytes(row["content"]) != raw:
                raise EvidenceConflict("Artifact identity has conflicting evidence")
            return _row(row)

    def register_corpus(
        self, snapshot: Mapping[str, Any], variables: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Atomically register corpus metadata and every supplied source/gold record.

        Inputs: source snapshot metadata and nonempty, unique variable mappings.
        Output: immutable snapshot ID and records. Required lexical/category/hash
        fields are validated; optional exact source bytes are independently hashed.
        Conflicts/cardinality/hash errors roll back the entire insertion. No files,
        parsers, external services, or provider calls are used.
        """
        if not variables:
            raise PersistenceError("Corpus variables cannot be empty")
        meta = _clean(snapshot)
        expected = int(
            meta.get(
                "expected_count", meta.get("record_count", meta.get("file_count", len(variables)))
            )
        )
        if len(variables) != expected:
            raise PersistenceError("Corpus count differs from declared expected count")
        fingerprint = _hash(meta)
        corpus_id = _id("corpus", fingerprint)
        seen_ids, seen_paths = set(), set()
        prepared = []
        for original in variables:
            record = dict(original)
            raw = record.pop(
                "source_content", record.pop("source_bytes", record.pop("raw_ttl", None))
            )
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            record = _clean(record)
            _require(
                record,
                "variable_id",
                "source_path",
                "definition",
                "gold",
                "category",
                "subcategory",
                "category_path",
                "source_sha256",
                "gold_sha256",
            )
            if record["variable_id"] in seen_ids or record["source_path"] in seen_paths:
                raise EvidenceConflict("Duplicate variable identity/path in corpus")
            seen_ids.add(record["variable_id"])
            seen_paths.add(record["source_path"])
            if set(record["gold"]) != _FIELDS or not all(
                _SHA.fullmatch(record[k]) for k in ("source_sha256", "gold_sha256")
            ):
                raise PersistenceError("Invalid gold fields or source/gold SHA-256")
            if _hash(record["gold"]) != record["gold_sha256"]:
                raise EvidenceConflict("Gold content hash does not match canonical bytes")
            if raw is not None and _hash(bytes(raw)) != record["source_sha256"]:
                raise EvidenceConflict("Source bytes do not match source hash")
            identifier = _id("variable", f"{corpus_id}:{record['variable_id']}")
            prepared.append((identifier, record, raw))
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO corpus_snapshot(id,fingerprint,repository,release,commit_id,tree_id,expected_count,evidence) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    corpus_id,
                    fingerprint,
                    str(meta.get("repository", "synthetic")),
                    str(meta.get("release", meta.get("tag", meta.get("version", "synthetic")))),
                    str(meta.get("commit", meta.get("commit_id", "synthetic"))),
                    str(meta.get("tree", meta.get("tree_id", "synthetic"))),
                    expected,
                    Jsonb(meta),
                ),
            )
            for identifier, record, raw in prepared:
                conn.execute(
                    """INSERT INTO variable(id,corpus_id,variable_id,source_path,label,definition,category,subcategory,category_path,source_sha256,gold_sha256,gold,source_content,demonstration_order,evidence,evidence_hash)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (
                        identifier,
                        corpus_id,
                        record["variable_id"],
                        record["source_path"],
                        record.get("label", record["variable_id"]),
                        record["definition"],
                        record["category"],
                        record["subcategory"],
                        record["category_path"],
                        record["source_sha256"],
                        record["gold_sha256"],
                        Jsonb(record["gold"]),
                        raw,
                        record.get("demonstration_order", record.get("demonstration_position")),
                        Jsonb(record),
                        _hash(record),
                    ),
                )
                existing = conn.execute(
                    "SELECT evidence_hash,source_content FROM variable WHERE id=%s", (identifier,)
                ).fetchone()
                if existing["evidence_hash"] != _hash(record) or (
                    raw is not None and existing["source_content"] != raw
                ):
                    raise EvidenceConflict("Corpus variable identity has conflicting evidence")
            row = conn.execute("SELECT * FROM corpus_snapshot WHERE id=%s", (corpus_id,)).fetchone()
        return {
            **_row(row),
            "variables": [{**r, "id": i, "corpus_id": corpus_id} for i, r, _ in prepared],
        }

    def register_campaign(
        self,
        configuration: Mapping[str, Any],
        mode: str = "synthetic",
        artifact_refs: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> str:
        """Freeze one campaign, selected providers, and their owned enabled models.

        Inputs: complete non-secret configuration, immutable synthetic/live mode,
        optional existing artifact IDs. Output: deterministic campaign UUID string.
        Validate ownership/billing/caps; conflicting evidence raises an error. This
        writes registry rows only and does not authorize live dispatch or plan work.
        """
        config = _clean(configuration)
        resolved = config.get("resolved", config.get("configuration", config))
        if mode not in {"synthetic", "live"}:
            raise PersistenceError("Campaign mode must be synthetic or live")
        selected = resolved.get("campaign", {}).get(
            "providers", resolved.get("selected_providers", [])
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= {"psnc", "openrouter", "mock"}
        ):
            raise PersistenceError("Campaign requires a unique nonempty provider selection")
        if mode == "live" and "mock" in selected:
            raise PersistenceError("Mock providers cannot enter a live campaign")
        fingerprint = _hash({"mode": mode, "configuration": config})
        identity = _id("campaign", fingerprint)
        execution = resolved.get("execution", {})
        cap = _cap(execution, "maximum_campaign_cost")
        currency = (execution.get("maximum_campaign_cost") or {}).get("currency", "EUR")
        prepared = []
        for provider in sorted(selected):
            profile = resolved.get("providers", {}).get(provider)
            if not isinstance(profile, Mapping):
                raise PersistenceError("Selected provider profile is missing")
            billing = profile.get("billing", {})
            billing_mode = billing.get("mode", "synthetic" if mode == "synthetic" else None)
            basis = billing.get("basis", "offline mock" if mode == "synthetic" else "")
            if (
                billing_mode not in {"non_billed", "metered", "synthetic"}
                or not basis
                or (mode == "live" and billing_mode == "synthetic")
            ):
                raise BudgetError("Valid provider billing mode and basis are required")
            models = [m for m in profile.get("models", []) if m.get("enabled", True)]
            ids = [m.get("id", m.get("model_id")) for m in models]
            if (
                not ids
                or any(not isinstance(m, str) or not m for m in ids)
                or len(set(ids)) != len(ids)
            ):
                raise PersistenceError("Each selected provider requires unique owned model IDs")
            prepared.append(
                (
                    provider,
                    profile,
                    billing_mode,
                    basis,
                    _cap(billing, "maximum_provider_cost"),
                    models,
                    ids,
                )
            )
        refs = (
            list(artifact_refs.values())
            if isinstance(artifact_refs, Mapping)
            else list(artifact_refs or [])
        )
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO campaign(id,fingerprint,mode,configuration,configuration_bytes,maximum_cost,currency) VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (identity, fingerprint, mode, Jsonb(config), _canonical(config), cap, currency),
            )
            existing = conn.execute(
                "SELECT configuration_bytes FROM campaign WHERE id=%s FOR UPDATE", (identity,)
            ).fetchone()
            if bytes(existing["configuration_bytes"]) != _canonical(config):
                raise EvidenceConflict("Campaign fingerprint conflicts with configuration bytes")
            for provider, profile, billing_mode, basis, provider_cap, models, ids in prepared:
                provider_currency = (
                    profile.get("billing", {}).get("maximum_provider_cost") or {}
                ).get("currency", currency)
                if provider_currency != currency:
                    raise BudgetError("Provider caps must use the campaign reporting currency")
                conn.execute(
                    "INSERT INTO campaign_provider(campaign_id,provider,profile,billing_mode,billing_basis,maximum_cost,currency) VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (
                        identity,
                        provider,
                        Jsonb(profile),
                        billing_mode,
                        basis,
                        provider_cap,
                        currency,
                    ),
                )
                for model, model_id in zip(models, ids, strict=True):
                    conn.execute(
                        "INSERT INTO model_configuration(campaign_id,provider,model_id,profile,profile_hash) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (identity, provider, model_id, Jsonb(model), _hash(model)),
                    )
            for ref in refs:
                artifact_id = (
                    ref.get("id", ref.get("artifact_id")) if isinstance(ref, Mapping) else ref
                )
                conn.execute(
                    "INSERT INTO campaign_artifact(campaign_id,artifact_id) VALUES(%s,%s) ON CONFLICT DO NOTHING",
                    (identity, artifact_id),
                )
        return identity

    def plan_tasks(
        self,
        campaign_id: str,
        runs: Sequence[Mapping[str, Any]],
        variables: Sequence[Mapping[str, Any] | str],
    ) -> dict[str, Any]:
        """Atomically freeze every run/variable task in a campaign's complete union.

        Inputs are the campaign ID, complete resolved-run list, and registered
        variable records or database IDs. Output contains plan identity and per-
        provider/combined counts. Reject duplicate, missing, demo, cross-provider,
        mixed-corpus, or conflicting membership; live mode requires 97 variables.
        Planning writes no partial dispatchable plan and makes zero provider calls.
        """
        if not runs or not variables:
            raise PersistenceError("Planning requires runs and targets")
        clean_runs = [_clean(run) for run in runs]
        with self.pool.connection() as conn:
            campaign = self._campaign(conn, campaign_id, lock=True)
            target_rows = []
            for item in variables:
                if isinstance(item, Mapping):
                    candidate = item.get("id")
                    if candidate and self._is_uuid(candidate):
                        row = conn.execute(
                            "SELECT * FROM variable WHERE id=%s", (candidate,)
                        ).fetchone()
                    else:
                        row = conn.execute(
                            "SELECT * FROM variable WHERE variable_id=%s AND source_sha256=%s AND gold_sha256=%s ORDER BY corpus_id",
                            (item["variable_id"], item["source_sha256"], item["gold_sha256"]),
                        ).fetchall()
                        if len(row) != 1:
                            raise PersistenceError(
                                "Variable registration is missing or ambiguous; pass its database id"
                            )
                        row = row[0]
                else:
                    row = conn.execute("SELECT * FROM variable WHERE id=%s", (item,)).fetchone()
                if not row:
                    raise PersistenceError("Unknown target variable")
                if row["demonstration_order"] is not None:
                    raise PersistenceError("Demonstrations cannot be scored")
                target_rows.append(row)
            target_rows.sort(key=lambda r: r["source_path"].encode("utf-8"))
            if (
                len({r["id"] for r in target_rows}) != len(target_rows)
                or len({r["corpus_id"] for r in target_rows}) != 1
            ):
                raise PersistenceError("Targets must be unique members of one corpus")
            if campaign["mode"] == "live" and len(target_rows) != 97:
                raise PersistenceError(
                    "Live population must contain all 97 non-demonstration variables"
                )
            if campaign["mode"] == "live":
                all_targets = {
                    r["id"]
                    for r in conn.execute(
                        "SELECT id FROM variable WHERE corpus_id=%s AND demonstration_order IS NULL",
                        (target_rows[0]["corpus_id"],),
                    )
                }
                if all_targets != {r["id"] for r in target_rows}:
                    raise PersistenceError(
                        "Live population omits or adds a corpus evaluation member"
                    )
            population_hash = _hash(
                [
                    {"id": str(r["id"]), "source": r["source_sha256"], "gold": r["gold_sha256"]}
                    for r in target_rows
                ]
            )
            owned = {
                (r["provider"], r["model_id"])
                for r in conn.execute(
                    "SELECT provider,model_id FROM model_configuration WHERE campaign_id=%s",
                    (campaign_id,),
                )
            }
            if {(r.get("provider"), r.get("model_id")) for r in clean_runs} != owned:
                raise PersistenceError(
                    "Run union must cover every selected provider/model exactly within its ownership"
                )
            run_records = []
            for run in clean_runs:
                _require(
                    run,
                    "provider",
                    "model_id",
                    "configuration_id",
                    "reasoning_mode",
                    "prompt_variant",
                    "shot_count",
                    "temperature",
                    "repetition",
                )
                fingerprint = _hash(
                    {"campaign": campaign_id, "run": run, "population": population_hash}
                )
                run_records.append((_id("run", fingerprint), fingerprint, run))
            if len({r[1] for r in run_records}) != len(run_records):
                raise EvidenceConflict("Duplicate resolved run in plan")
            plan = {
                "campaign_id": campaign_id,
                "population_hash": population_hash,
                "variable_ids": [str(r["id"]) for r in target_rows],
                "run_hashes": sorted(r[1] for r in run_records),
            }
            plan_hash = _hash(plan)
            existing = conn.execute(
                "SELECT * FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
            ).fetchone()
            if existing:
                if existing["fingerprint"] != plan_hash:
                    raise EvidenceConflict("A frozen plan cannot change membership")
                return self._plan_summary(conn, campaign_id)
            conn.execute(
                "INSERT INTO campaign_plan(campaign_id,fingerprint,corpus_id,population_hash,population_size,run_count,task_count,evidence) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    campaign_id,
                    plan_hash,
                    target_rows[0]["corpus_id"],
                    population_hash,
                    len(target_rows),
                    len(run_records),
                    len(target_rows) * len(run_records),
                    Jsonb(plan),
                ),
            )
            for position, target in enumerate(target_rows):
                conn.execute(
                    "INSERT INTO population_member(campaign_id,variable_id,position) VALUES(%s,%s,%s)",
                    (campaign_id, target["id"], position),
                )
            for run_id, fingerprint, run in run_records:
                conn.execute(
                    """INSERT INTO resolved_run(id,campaign_id,fingerprint,configuration_id,provider,model_id,reasoning_mode,reasoning_fields,prompt_variant,shot_count,temperature,top_p,max_output_tokens,repetition,evidence,evidence_hash)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        run_id,
                        campaign_id,
                        fingerprint,
                        run["configuration_id"],
                        run["provider"],
                        run["model_id"],
                        run["reasoning_mode"],
                        Jsonb(run.get("reasoning_fields", {})),
                        run["prompt_variant"],
                        run["shot_count"],
                        run["temperature"],
                        run.get("top_p"),
                        run.get("max_output_tokens"),
                        run["repetition"],
                        Jsonb(run),
                        _hash(run),
                    ),
                )
                for target in target_rows:
                    task_hash = _hash(
                        {
                            "campaign": campaign_id,
                            "run": fingerprint,
                            "variable": str(target["id"]),
                            "source": target["source_sha256"],
                            "gold": target["gold_sha256"],
                        }
                    )
                    task_id = _id("task", task_hash)
                    conn.execute(
                        "INSERT INTO task(id,campaign_id,run_id,provider,variable_id,fingerprint) VALUES(%s,%s,%s,%s,%s,%s)",
                        (task_id, campaign_id, run_id, run["provider"], target["id"], task_hash),
                    )
                    self._event(conn, task_id, None, "queued", "planned", {"plan_hash": plan_hash})
            conn.execute(
                "UPDATE campaign SET state='planned',updated_at=clock_timestamp() WHERE id=%s",
                (campaign_id,),
            )
            return self._plan_summary(conn, campaign_id)

    @staticmethod
    def _is_uuid(value: Any) -> bool:
        """Return whether an input is a valid UUID; never raise or perform I/O."""
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _campaign(conn: psycopg.Connection, identity: str, *, lock: bool = False) -> dict[str, Any]:
        """Read an explicit campaign inside the caller's transaction; optionally lock.

        Returns the row; raises PersistenceError for an unknown ID. Does not commit.
        """
        row = conn.execute(
            "SELECT * FROM campaign WHERE id=%s" + (" FOR UPDATE" if lock else ""), (identity,)
        ).fetchone()
        if not row:
            raise PersistenceError("Unknown campaign")
        if _hash(bytes(row["configuration_bytes"])) != _hash(row["configuration"]):
            raise EvidenceConflict("Campaign canonical configuration hash mismatch")
        return row

    @staticmethod
    def _plan_summary(conn: psycopg.Connection, campaign_id: str) -> dict[str, Any]:
        """Read plan counts using the caller's connection; return exact coverage facts."""
        row = conn.execute(
            "SELECT * FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
        ).fetchone()
        counts = list(
            conn.execute(
                "SELECT provider,count(*) AS task_count,count(DISTINCT run_id) AS run_count FROM task WHERE campaign_id=%s GROUP BY provider ORDER BY provider",
                (campaign_id,),
            )
        )
        return {
            **_row(row),
            "plan_hash": row["fingerprint"],
            "providers": [_row(r) for r in counts],
            "maximum_requests": row["task_count"] * 3,
        }

    @staticmethod
    def _event(
        conn: psycopg.Connection,
        task_id: str,
        prior: str | None,
        next_state: str,
        cause: str,
        evidence: Mapping[str, Any],
    ) -> None:
        """Append one state event in the caller's transaction; return None, no commit."""
        conn.execute(
            "INSERT INTO task_event(id,task_id,prior_state,next_state,cause,evidence) VALUES(%s,%s,%s,%s,%s,%s)",
            (str(uuid.uuid4()), task_id, prior, next_state, cause, Jsonb(_clean(evidence))),
        )

    @staticmethod
    def _lease(conn: psycopg.Connection, lease: Mapping[str, Any]) -> dict[str, Any]:
        """Lock and verify an unexpired lease's owner/token/fence in one transaction.

        Input is a returned lease mapping. Output is current task row; stale or
        missing ownership raises StaleLeaseError. This never extends a lease.
        """
        _require(lease, "lease_token", "worker_id", "fence")
        task_id = lease.get("task_id", lease.get("id"))
        row = conn.execute(
            "SELECT *,lease_expires_at>clock_timestamp() AS lease_active FROM task WHERE id=%s FOR UPDATE",
            (task_id,),
        ).fetchone()
        if (
            not row
            or not row["lease_active"]
            or str(row["lease_token"]) != str(lease["lease_token"])
            or row["worker_id"] != lease["worker_id"]
            or row["fence"] != lease["fence"]
        ):
            raise StaleLeaseError("Task lease is absent, expired, or superseded")
        return row

    def claim_tasks(
        self,
        campaign_id: str,
        worker: str,
        limit: int = 1,
        lease_seconds: int = 300,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Claim eligible tasks with SKIP LOCKED and fresh fencing tokens.

        Inputs are campaign/worker IDs, positive claim/lease limits, optional owned
        provider filter. Output leases include complete run/variable evidence.
        Provider pauses block only new dispatch stages; durable local work remains
        claimable. Contention returns fewer rows, never duplicate active owners.
        This writes leases/events and commits before returning; no external I/O.
        """
        if not worker or limit < 1 or lease_seconds < 1:
            raise PersistenceError("Worker, claim limit, and lease duration must be valid")
        ids = []
        with self.pool.connection() as conn:
            campaign = self._campaign(conn, campaign_id, lock=True)
            if campaign["state"] not in {"planned", "running"}:
                return []
            selected = conn.execute(
                "SELECT * FROM campaign_provider WHERE campaign_id=%s AND (%s::text IS NULL OR provider=%s) ORDER BY provider FOR UPDATE",
                (campaign_id, provider, provider),
            ).fetchall()
            config = campaign["configuration"].get(
                "resolved",
                campaign["configuration"].get("configuration", campaign["configuration"]),
            )
            global_limit = config.get("execution", {}).get("worker_count")
            if global_limit is not None:
                active = conn.execute(
                    "SELECT count(*) AS n FROM task WHERE campaign_id=%s AND lease_expires_at>clock_timestamp()",
                    (campaign_id,),
                ).fetchone()["n"]
                limit = min(limit, max(0, int(global_limit) - active))
            if limit == 0:
                return []
            capacities = {}
            for scope in selected:
                provider_limit = scope["profile"].get("max_concurrency")
                active = conn.execute(
                    "SELECT count(*) AS n FROM task WHERE campaign_id=%s AND provider=%s AND lease_expires_at>clock_timestamp()",
                    (campaign_id, scope["provider"]),
                ).fetchone()["n"]
                capacities[scope["provider"]] = (
                    limit if provider_limit is None else max(0, int(provider_limit) - active)
                )
            conn.execute(
                "INSERT INTO worker_session(id) VALUES(%s) ON CONFLICT(id) DO UPDATE SET heartbeat_at=clock_timestamp()",
                (worker,),
            )
            rows = []
            for scope in selected:
                take = min(capacities[scope["provider"]], limit - len(rows))
                if take <= 0:
                    continue
                rows.extend(
                    conn.execute(
                        """SELECT t.* FROM task t JOIN campaign_provider p ON p.campaign_id=t.campaign_id AND p.provider=t.provider
                    WHERE t.campaign_id=%s AND t.provider=%s
                      AND t.state IN ('queued','retry_pending','request_persisted','response_stored','validated','prediction_ready')
                      AND (t.lease_expires_at IS NULL OR t.lease_expires_at<=clock_timestamp())
                      AND (t.state IN ('response_stored','validated','prediction_ready') OR
                           (p.state='ready' OR (p.state='cooldown' AND p.cooldown_until<=clock_timestamp())))
                    ORDER BY t.fingerprint FOR UPDATE OF t SKIP LOCKED LIMIT %s""",
                        (campaign_id, scope["provider"], take),
                    ).fetchall()
                )
            for row in rows:
                if len(ids) >= limit:
                    break
                if capacities.get(row["provider"], 0) <= 0:
                    continue
                capacities[row["provider"]] -= 1
                token = str(uuid.uuid4())
                conn.execute(
                    "UPDATE task SET worker_id=%s,lease_token=%s,fence=fence+1,lease_expires_at=clock_timestamp()+%s*interval '1 second',heartbeat_at=clock_timestamp(),row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                    (worker, token, lease_seconds, row["id"]),
                )
                self._event(
                    conn,
                    str(row["id"]),
                    row["state"],
                    row["state"],
                    "claimed",
                    {"worker": worker, "lease_token": token, "fence": row["fence"] + 1},
                )
                ids.append(str(row["id"]))
        return [self.get_task(identity) for identity in ids]

    def record_live_authorization(
        self,
        campaign_id: str,
        estimate: Mapping[str, Any],
        disclosure: Mapping[str, Any],
        authorization: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Retain a disclosed plan-bound estimate and separate explicit live approval.

        Inputs are a planned live campaign and three immutable receipts. Estimate
        requires plan_fingerprint, policy_version, usable=True, and price_evidence;
        disclosure requires estimate_hash/disclosed_at; authorization requires
        estimate_hash/plan_fingerprint/authorized_at/actor/explicit=True. Returns
        the stored receipt. Missing/mismatched evidence raises PersistenceError.
        This only records supplied authority; it never infers or requests approval.
        """
        est, dis, auth = _clean(estimate), _clean(disclosure), _clean(authorization)
        _require(est, "plan_fingerprint", "policy_version", "usable", "price_evidence")
        _require(dis, "estimate_hash", "disclosed_at")
        _require(auth, "estimate_hash", "plan_fingerprint", "authorized_at", "actor", "explicit")
        estimate_hash = _hash(est)
        evidence_hash = _hash(
            {"campaign": campaign_id, "estimate": est, "disclosure": dis, "authorization": auth}
        )
        with self.pool.connection() as conn:
            campaign = self._campaign(conn, campaign_id, lock=True)
            plan = conn.execute(
                "SELECT fingerprint FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
            ).fetchone()
            if campaign["mode"] != "live" or not plan:
                raise PersistenceError("Only a planned live campaign accepts live authorization")
            if (
                est["policy_version"] != "pre-run-estimate-v1"
                or est["usable"] is not True
                or auth["explicit"] is not True
            ):
                raise PersistenceError(
                    "Usable pre-run estimate and separate explicit approval are required"
                )
            if (
                est["plan_fingerprint"] != plan["fingerprint"]
                or auth["plan_fingerprint"] != plan["fingerprint"]
                or dis["estimate_hash"] != estimate_hash
                or auth["estimate_hash"] != estimate_hash
            ):
                raise EvidenceConflict(
                    "Estimate, disclosure, or authorization is bound to another plan"
                )
            identity = _id("authorization", evidence_hash)
            conn.execute(
                "INSERT INTO live_authorization(id,campaign_id,plan_fingerprint,estimate,estimate_hash,disclosure,authorization_receipt,evidence_hash) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    identity,
                    campaign_id,
                    plan["fingerprint"],
                    Jsonb(est),
                    estimate_hash,
                    Jsonb(dis),
                    Jsonb(auth),
                    evidence_hash,
                ),
            )
            return _row(
                conn.execute("SELECT * FROM live_authorization WHERE id=%s", (identity,)).fetchone()
            )

    def start_attempt(self, lease: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
        """Commit one numbered request and its cost reservation before dispatch.

        Inputs: current fenced lease; messages, prompt, sanitized body, optional
        scientific_parameters/attempt_number/correction_parent, and cost receipt.
        Live requests need bound authorization and price-supported reservation;
        capped paid requests additionally need bounded=True. Return immutable
        attempt plus transient lease context. Reject drift, stale leases, missing
        lineage, cap overflow, or attempt four. Writes one short transaction only.
        """
        evidence = _clean(request)
        _require(evidence, "messages", "prompt", "body")
        if (
            not isinstance(evidence["messages"], list)
            or not evidence["messages"]
            or not isinstance(evidence["prompt"], str)
            or not isinstance(evidence["body"], Mapping)
        ):
            raise PersistenceError("A complete prompt/messages/body record is required")
        body = evidence["body"]
        scientific = _clean(
            evidence.get(
                "scientific_parameters", {k: v for k, v in body.items() if k != "messages"}
            )
        )
        task_id = lease.get("task_id", lease.get("id"))
        with self.pool.connection() as conn:
            identity_row = conn.execute(
                "SELECT campaign_id,provider FROM task WHERE id=%s", (task_id,)
            ).fetchone()
            if not identity_row:
                raise PersistenceError("Unknown task")
            campaign = self._campaign(conn, str(identity_row["campaign_id"]), lock=True)
            provider = conn.execute(
                "SELECT * FROM campaign_provider WHERE campaign_id=%s AND provider=%s FOR UPDATE",
                (campaign["id"], identity_row["provider"]),
            ).fetchone()
            task = self._lease(conn, lease)
            run = conn.execute(
                "SELECT * FROM resolved_run WHERE id=%s", (task["run_id"],)
            ).fetchone()
            number = evidence.get("attempt_number", task["attempt_count"] + 1)
            if not isinstance(number, int) or not 1 <= number <= 3:
                raise AttemptLimitError("Only attempts 1, 2, and 3 are permitted")
            request_hash = _hash(evidence)
            existing = conn.execute(
                "SELECT * FROM attempt WHERE task_id=%s AND attempt_number=%s", (task_id, number)
            ).fetchone()
            if existing:
                if existing["request_hash"] != request_hash:
                    raise EvidenceConflict("Attempt request evidence conflicts with stored bytes")
                return {**_row(existing), "lease": dict(lease)}
            if number != task["attempt_count"] + 1 or task["state"] not in {
                "queued",
                "retry_pending",
            }:
                raise AttemptLimitError("Next request must follow the committed task checkpoint")
            if campaign["state"] not in {"planned", "running"}:
                raise PersistenceError("Campaign dispatch is paused or terminal")
            eligible = provider["state"] == "ready" or (
                provider["state"] == "cooldown"
                and provider["cooldown_until"] is not None
                and conn.execute(
                    "SELECT %s<=clock_timestamp() AS due", (provider["cooldown_until"],)
                ).fetchone()["due"]
            )
            if not eligible:
                raise PersistenceError("This provider is not eligible for new dispatch")
            self._rate_admission(conn, task, provider, evidence)
            if body.get("model") != run["model_id"] or (
                "provider" in evidence and evidence["provider"] != run["provider"]
            ):
                raise EvidenceConflict("Request provider/model differs from the frozen run")
            for key, column in (
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_tokens", "max_output_tokens"),
                ("max_completion_tokens", "max_output_tokens"),
            ):
                if (
                    key in body
                    and run[column] is not None
                    and Decimal(str(body[key])) != Decimal(str(run[column]))
                ):
                    raise EvidenceConflict(
                        "Request sampling/output value differs from the frozen run"
                    )
            parent = evidence.get("correction_parent")
            if number > 1:
                previous = conn.execute(
                    "SELECT a.*,s.delivery,v.content_invalid FROM attempt a JOIN attempt_state s ON s.attempt_id=a.id LEFT JOIN validation_event v ON v.attempt_id=a.id WHERE a.task_id=%s AND a.attempt_number=%s",
                    (task_id, number - 1),
                ).fetchone()
                if not previous or previous["scientific_hash"] != _hash(scientific):
                    raise EvidenceConflict("Scientific settings changed between attempts")
                if previous["delivery"] in {"dispatch_started", "ambiguous_delivery"}:
                    raise PersistenceError("Ambiguous delivery cannot authorize another request")
                if previous["content_invalid"]:
                    if parent is None:
                        parent = str(previous["id"])
                    if str(parent) != str(previous["id"]):
                        raise EvidenceConflict(
                            "Correction parent is not the immediately previous attempt"
                        )
                elif parent is not None:
                    raise PersistenceError("Transport-only retry cannot invent correction content")
            elif parent is not None:
                raise PersistenceError("Attempt one cannot have a correction parent")
            cost = evidence.get("cost", {})
            non_billed = campaign["mode"] == "synthetic" or provider["billing_mode"] == "non_billed"
            amount = (
                Decimal(0)
                if non_billed
                else _money(cost.get("reservation_amount", cost.get("estimated_cost")))
            )
            bounded = True if non_billed else cost.get("bounded") is True
            authorization_id = None
            if campaign["mode"] == "live":
                plan = conn.execute(
                    "SELECT fingerprint FROM campaign_plan WHERE campaign_id=%s", (campaign["id"],)
                ).fetchone()
                auth = conn.execute(
                    "SELECT * FROM live_authorization WHERE campaign_id=%s AND plan_fingerprint=%s ORDER BY created_at DESC LIMIT 1",
                    (campaign["id"], plan["fingerprint"]),
                ).fetchone()
                if not auth:
                    raise PersistenceError(
                        "Live dispatch requires a disclosed plan estimate and separate explicit authorization"
                    )
                authorization_id = auth["id"]
                if not non_billed and not cost.get("price_evidence"):
                    raise BudgetError(
                        "Metered dispatch requires frozen price-supported cost evidence"
                    )
            if (
                amount > 0
                and (campaign["maximum_cost"] is not None or provider["maximum_cost"] is not None)
                and not bounded
            ):
                raise BudgetError("Capped paid dispatch requires a defensible reservation bound")
            for scope in (campaign, provider):
                if (
                    amount > 0
                    and scope["maximum_cost"] is not None
                    and scope["spent_cost"] + scope["reserved_cost"] + amount
                    > scope["maximum_cost"]
                ):
                    raise BudgetError("An optional monetary cap prevents this reservation")
            attempt_id = _id("attempt", f"{task_id}:{number}")
            key = evidence.get("idempotency_key", _hash({"task": task_id, "attempt": number}))
            conn.execute(
                """INSERT INTO attempt(id,task_id,attempt_number,correction_parent,provider,model_id,messages,prompt,request_body,request_evidence,request_hash,scientific_parameters,scientific_hash,idempotency_key,lease_fence)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    attempt_id,
                    task_id,
                    number,
                    parent,
                    run["provider"],
                    run["model_id"],
                    Jsonb(evidence["messages"]),
                    evidence["prompt"],
                    Jsonb(body),
                    Jsonb(evidence),
                    request_hash,
                    Jsonb(scientific),
                    _hash(scientific),
                    key,
                    task["fence"],
                ),
            )
            conn.execute("INSERT INTO attempt_state(attempt_id) VALUES(%s)", (attempt_id,))
            conn.execute(
                "INSERT INTO cost_reservation(attempt_id,campaign_id,provider,amount,currency,bounded,basis,authorization_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    attempt_id,
                    campaign["id"],
                    run["provider"],
                    amount,
                    campaign["currency"],
                    bounded,
                    Jsonb(
                        {
                            "billing_mode": provider["billing_mode"],
                            "billing_basis": provider["billing_basis"],
                            "cost": cost,
                        }
                    ),
                    authorization_id,
                ),
            )
            conn.execute(
                "UPDATE campaign SET reserved_cost=reserved_cost+%s WHERE id=%s",
                (amount, campaign["id"]),
            )
            conn.execute(
                "UPDATE campaign_provider SET reserved_cost=reserved_cost+%s WHERE campaign_id=%s AND provider=%s",
                (amount, campaign["id"], run["provider"]),
            )
            conn.execute(
                "UPDATE task SET state='request_persisted',attempt_count=%s,row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                (number, task_id),
            )
            self._event(
                conn,
                task_id,
                task["state"],
                "request_persisted",
                "attempt_recorded",
                {"attempt_id": attempt_id, "attempt_number": number},
            )
            return {
                **_row(conn.execute("SELECT * FROM attempt WHERE id=%s", (attempt_id,)).fetchone()),
                "lease": dict(lease),
            }

    @staticmethod
    def _rate_admission(
        conn: psycopg.Connection,
        task: Mapping[str, Any],
        provider: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> None:
        """Check shared rolling 60-second request/token admissions under provider lock.

        Inputs are the caller's locked provider, task, and token_bound request.
        Return None when capacity exists; RateLimitError gives a conservative wait
        until the relevant old admissions expire. No row or attempt is allocated.
        Unknown token bounds block a configured token limit, never assume zero.
        """
        rpm = provider["profile"].get("requests_per_minute")
        tpm = provider["profile"].get("tokens_per_minute")
        if rpm is None and tpm is None:
            return
        bound = request.get("token_bound")
        if tpm is not None and (not isinstance(bound, int) or isinstance(bound, bool) or bound < 1):
            raise PersistenceError(
                "A configured token rate requires a positive token_bound receipt"
            )
        if tpm is not None and bound > int(tpm):
            raise PersistenceError(
                "One request's token bound exceeds the provider's token-minute limit"
            )
        admissions = conn.execute(
            """SELECT a.request_evidence,extract(epoch FROM (a.created_at+interval '60 seconds'-clock_timestamp())) AS wait
            FROM attempt a JOIN task t ON t.id=a.task_id WHERE t.campaign_id=%s AND a.provider=%s
            AND a.created_at>clock_timestamp()-interval '60 seconds' ORDER BY a.created_at""",
            (task["campaign_id"], task["provider"]),
        ).fetchall()
        if tpm is not None and any(
            not isinstance(row["request_evidence"].get("token_bound"), int) for row in admissions
        ):
            raise PersistenceError("Stored admission is missing required token-bound evidence")
        tokens = sum(row["request_evidence"].get("token_bound", 0) for row in admissions)
        count = len(admissions)
        if (rpm is None or count < int(rpm)) and (tpm is None or tokens + bound <= int(tpm)):
            return
        wait = 0.001
        for row in admissions:
            wait = float(row["wait"]) + 0.01
            count -= 1
            tokens -= row["request_evidence"].get("token_bound", 0)
            if (rpm is None or count < int(rpm)) and (tpm is None or tokens + bound <= int(tpm)):
                break
        raise RateLimitError(wait)

    def mark_dispatched(
        self, attempt: Mapping[str, Any], lease: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Record dispatch intent once immediately before a single external request.

        Input is a stored attempt and current lease (embedded or explicit). Output
        includes dispatch_allowed; an already-dispatched attempt returns False and
        must never be sent again. Raises for stale ownership or inconsistent state.
        Writes transport/state evidence only; this method never sends HTTP.
        """
        current_lease = lease or attempt.get("lease")
        if current_lease is None:
            raise StaleLeaseError("Dispatch requires current lease context")
        with self.pool.connection() as conn:
            task = self._lease(conn, current_lease)
            row = conn.execute(
                "SELECT a.*,s.delivery FROM attempt a JOIN attempt_state s ON s.attempt_id=a.id WHERE a.id=%s FOR UPDATE OF s",
                (attempt["id"],),
            ).fetchone()
            if not row or row["task_id"] != task["id"]:
                raise EvidenceConflict("Attempt belongs to another task")
            if row["delivery"] != "not_dispatched":
                return {**_row(row), "dispatch_allowed": False}
            if conn.execute("SELECT 1 FROM response WHERE attempt_id=%s", (row["id"],)).fetchone():
                return {**_row(row), "dispatch_allowed": False}
            conn.execute(
                "UPDATE attempt_state SET delivery='dispatch_started',dispatched_at=clock_timestamp() WHERE attempt_id=%s",
                (row["id"],),
            )
            self._transport(conn, str(row["id"]), "dispatch_started", {"fence": task["fence"]})
            conn.execute(
                "UPDATE task SET state='generating',row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                (task["id"],),
            )
            self._event(
                conn,
                str(task["id"]),
                task["state"],
                "generating",
                "dispatch_started",
                {"attempt_id": str(row["id"])},
            )
            return {**_row(row), "delivery": "dispatch_started", "dispatch_allowed": True}

    @staticmethod
    def _transport(
        conn: psycopg.Connection, attempt_id: str, kind: str, evidence: Mapping[str, Any]
    ) -> None:
        """Append idempotent transport evidence in the caller transaction; no commit."""
        content = _clean(evidence)
        digest = _hash(content)
        conn.execute(
            "INSERT INTO transport_event(id,attempt_id,kind,evidence,evidence_hash) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (
                _id("transport", f"{attempt_id}:{kind}:{digest}"),
                attempt_id,
                kind,
                Jsonb(content),
                digest,
            ),
        )

    def store_response(
        self, attempt: Mapping[str, Any] | str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Persist untouched raw response and metadata before parsing, exactly once.

        Inputs: existing attempt ID/record and result with raw_body bytes/text,
        assistant_text, delivery, optional envelope/usage/error/timing/cost. Output
        is the immutable response row. Identical replay is safe; different bytes
        or metadata raise EvidenceConflict. Late evidence may be retained after a
        lease expires, but only its original fence may advance mutable task state.
        Cost settlement and response commit are one transaction; no provider I/O.
        """
        attempt_id = attempt if isinstance(attempt, str) else str(attempt["id"])
        metadata = dict(result)
        raw = metadata.pop("raw_body", metadata.pop("raw_response", None))
        encoded_raw = metadata.pop("raw_response_base64", None)
        if encoded_raw is not None:
            decoded = base64.b64decode(encoded_raw, validate=True)
            if isinstance(raw, bytes) and raw != decoded:
                raise EvidenceConflict("Raw response bytes disagree with base64 receipt")
            raw = decoded
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if raw is not None:
            raw = bytes(raw)
        metadata = _clean(metadata)
        delivery = metadata.get(
            "delivery", "response_received" if raw is not None else "ambiguous_delivery"
        )
        if delivery not in {
            "response_received",
            "not_dispatched",
            "rejected",
            "ambiguous_delivery",
        }:
            raise PersistenceError("Unknown response delivery classification")
        if delivery == "response_received" and raw is None:
            raise PersistenceError("Delivered response requires exact raw bytes")
        metadata["delivery"] = delivery
        digest = _hash(
            {"raw_sha256": _hash(raw) if raw is not None else None, "metadata": metadata}
        )
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT a.*,t.campaign_id FROM attempt a JOIN task t ON t.id=a.task_id WHERE a.id=%s",
                (attempt_id,),
            ).fetchone()
            if not row:
                raise PersistenceError("Unknown attempt")
            campaign = self._campaign(conn, str(row["campaign_id"]), lock=True)
            provider = conn.execute(
                "SELECT * FROM campaign_provider WHERE campaign_id=%s AND provider=%s FOR UPDATE",
                (row["campaign_id"], row["provider"]),
            ).fetchone()
            task = conn.execute(
                "SELECT * FROM task WHERE id=%s FOR UPDATE", (row["task_id"],)
            ).fetchone()
            existing = conn.execute(
                "SELECT * FROM response WHERE attempt_id=%s", (attempt_id,)
            ).fetchone()
            if existing:
                if existing["evidence_hash"] != digest or existing["raw_body"] != raw:
                    raise EvidenceConflict("Response replay conflicts with immutable raw evidence")
                return _row(existing)
            response_id = _id("response", attempt_id)
            conn.execute(
                """INSERT INTO response(id,attempt_id,raw_body,raw_sha256,assistant_text,reasoning_text,envelope,usage,http_status,returned_model_id,provider_request_id,finish_reason,latency_seconds,delivery,evidence,evidence_hash)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    response_id,
                    attempt_id,
                    raw,
                    _hash(raw) if raw is not None else None,
                    metadata.get("assistant_text"),
                    metadata.get("reasoning_text"),
                    Jsonb(metadata.get("envelope", metadata.get("response_json"))),
                    Jsonb(metadata.get("usage")),
                    metadata.get("http_status", metadata.get("status_code")),
                    metadata.get("returned_model_id", metadata.get("returned_model")),
                    metadata.get("provider_request_id", metadata.get("request_id")),
                    metadata.get("finish_reason"),
                    metadata.get("latency_seconds"),
                    delivery,
                    Jsonb(metadata),
                    digest,
                ),
            )
            conn.execute(
                "UPDATE attempt_state SET delivery=%s,received_at=clock_timestamp() WHERE attempt_id=%s",
                (delivery, attempt_id),
            )
            self._transport(
                conn,
                attempt_id,
                "response_recorded",
                {
                    "response_id": response_id,
                    "delivery": delivery,
                    "raw_sha256": _hash(raw) if raw is not None else None,
                },
            )
            next_state = (
                "response_stored"
                if delivery == "response_received"
                else "ambiguous_delivery"
                if delivery == "ambiguous_delivery"
                else "operational_failed"
            )
            if task["fence"] == row["lease_fence"] and task["state"] not in {
                "complete",
                "prediction_ready",
            }:
                conn.execute(
                    "UPDATE task SET state=%s,row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                    (next_state, task["id"]),
                )
                self._event(
                    conn,
                    str(task["id"]),
                    task["state"],
                    next_state,
                    "response_recorded",
                    {"attempt_id": attempt_id},
                )
            reservation = conn.execute(
                "SELECT * FROM cost_reservation WHERE attempt_id=%s", (attempt_id,)
            ).fetchone()
            non_billed = campaign["mode"] == "synthetic" or provider["billing_mode"] == "non_billed"
            cost = metadata.get("cost", {})
            if isinstance(cost, (int, float, str)):
                cost = {"amount": cost, "state": "actual"}
            amount = Decimal(0) if non_billed else _money(cost.get("amount"), nullable=True)
            state = (
                "confirmed_zero"
                if non_billed
                else "ambiguous"
                if delivery == "ambiguous_delivery"
                else cost.get("state", "unavailable" if amount is None else "estimated")
            )
            if state not in {"confirmed_zero", "actual", "estimated", "unavailable", "ambiguous"}:
                raise BudgetError("Unknown cost settlement state")
            settlement = {
                "cost": cost,
                "delivery": delivery,
                "billing_mode": provider["billing_mode"],
                "billing_basis": provider["billing_basis"],
                "usage_available": metadata.get("usage") is not None,
            }
            conn.execute(
                "INSERT INTO cost_settlement(attempt_id,amount,state,evidence,evidence_hash) VALUES(%s,%s,%s,%s,%s)",
                (attempt_id, amount, state, Jsonb(settlement), _hash(settlement)),
            )
            if amount is not None and state != "ambiguous":
                conn.execute(
                    "UPDATE campaign SET reserved_cost=reserved_cost-%s,spent_cost=spent_cost+%s WHERE id=%s",
                    (reservation["amount"], amount, row["campaign_id"]),
                )
                conn.execute(
                    "UPDATE campaign_provider SET reserved_cost=reserved_cost-%s,spent_cost=spent_cost+%s WHERE campaign_id=%s AND provider=%s",
                    (reservation["amount"], amount, row["campaign_id"], row["provider"]),
                )
            return _row(
                conn.execute("SELECT * FROM response WHERE id=%s", (response_id,)).fetchone()
            )

    def record_validation(
        self, attempt: Mapping[str, Any] | str, evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Persist complete extraction/schema/semantic validity receipt idempotently.

        Input contains valid bool, errors list, optional candidate, versions,
        extraction details, and content_invalid. Output is immutable receipt.
        Requires a delivered stored raw response. Conflicts or malformed validity
        evidence raise; only PostgreSQL evidence/state is changed, never generation.
        """
        attempt_id = attempt if isinstance(attempt, str) else str(attempt["id"])
        content = _clean(evidence)
        _require(content, "valid", "errors")
        if not isinstance(content["valid"], bool) or not isinstance(content["errors"], list):
            raise PersistenceError("Validity must be boolean and errors an ordered list")
        content_invalid = content.get("content_invalid", not content["valid"])
        if not isinstance(content_invalid, bool) or (
            content["valid"] and (content_invalid or content["errors"])
        ):
            raise PersistenceError("Inconsistent validation outcome")
        digest = _hash(content)
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT a.*,r.delivery FROM attempt a JOIN response r ON r.attempt_id=a.id WHERE a.id=%s",
                (attempt_id,),
            ).fetchone()
            if not row or row["delivery"] != "response_received":
                raise PersistenceError(
                    "Validation requires durable delivered raw response evidence"
                )
            identity = _id("validation", attempt_id)
            conn.execute(
                "INSERT INTO validation_event(id,attempt_id,valid,content_invalid,candidate,errors,evidence,evidence_hash) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    identity,
                    attempt_id,
                    content["valid"],
                    content_invalid,
                    Jsonb(content.get("candidate")),
                    Jsonb(content["errors"]),
                    Jsonb(content),
                    digest,
                ),
            )
            stored = conn.execute(
                "SELECT * FROM validation_event WHERE attempt_id=%s", (attempt_id,)
            ).fetchone()
            if stored["evidence_hash"] != digest:
                raise EvidenceConflict("Validation replay conflicts with stored evidence")
            task = conn.execute(
                "SELECT * FROM task WHERE id=%s FOR UPDATE", (row["task_id"],)
            ).fetchone()
            if task["state"] == "response_stored" and task["fence"] == row["lease_fence"]:
                conn.execute(
                    "UPDATE task SET state='validated',row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                    (task["id"],),
                )
                self._event(
                    conn,
                    str(task["id"]),
                    task["state"],
                    "validated",
                    "validation_recorded",
                    {"attempt_id": attempt_id, "valid": content["valid"]},
                )
            return _row(stored)

    @staticmethod
    def _evaluation_metrics(result: Mapping[str, Any]) -> list[tuple[str, str, str, Fraction]]:
        """Validate exact evaluator receipts and flatten queryable metric rows.

        Input is a complete evaluate_item result with six components in each mode.
        Output contains mode/component/metric/reduced-rational tuples. Verify hash,
        nonnegative confusion counts, component sums, and derived PRF arithmetic;
        corruption raises EvidenceConflict without invoking an evaluator or model.
        """
        _require(
            result,
            "scorer_version",
            "gold_hash",
            "prediction_hash",
            "exact",
            "close",
            "result_hash",
        )
        if _hash({k: v for k, v in result.items() if k != "result_hash"}) != result["result_hash"]:
            raise EvidenceConflict("Evaluator result hash does not match its evidence")
        rows = []
        for mode in ("exact", "close"):
            section = result[mode]
            if set(section.get("components", {})) != _FIELDS:
                raise EvidenceConflict("Evaluation does not contain all six component receipts")
            summed = {k: Fraction(0) for k in ("tp", "fp", "fn", "tn")}
            scopes = [
                (name, part["contributions"], part["metrics"])
                for name, part in section["components"].items()
            ]
            scopes.append(("__variable__", section["totals"], section["metrics"]))
            for component, counts, metrics in scopes:
                values = {key: _fraction(counts[key]) for key in ("tp", "fp", "fn", "tn")}
                if any(value < 0 for value in values.values()):
                    raise EvidenceConflict("Confusion contributions cannot be negative")
                tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
                expected = {
                    "precision": tp / (tp + fp) if tp + fp else Fraction(0),
                    "recall": tp / (tp + fn) if tp + fn else Fraction(0),
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0),
                }
                if any(_fraction(metrics[key]) != value for key, value in expected.items()):
                    raise EvidenceConflict("PRF metric does not reconcile with exact contributions")
                if component == "__variable__":
                    if summed != values:
                        raise EvidenceConflict("Variable totals differ from six component sums")
                else:
                    summed = {key: summed[key] + values[key] for key in summed}
                rows.extend(
                    (mode, component, key, value) for key, value in {**values, **expected}.items()
                )
        return rows

    def store_evaluation(
        self, task: Mapping[str, Any] | str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Retain an item evaluation and normalized exact metrics before completion.

        Inputs: task ID/lease and the complete pure evaluator receipt. Output is an
        immutable evaluation row. Validate gold/prediction lineage, result hash,
        all six components, exact confusion sums and PRF arithmetic. A lease input
        is fenced; an ID supports safe idempotent local scoring after a restart.
        Different scorers create new namespaces; conflicting same-scorer results
        fail. This writes evidence/metrics and task completion only, never scores.
        """
        task_id = task if isinstance(task, str) else task.get("task_id", task.get("id"))
        evidence = _clean(result)
        metrics = self._evaluation_metrics(evidence)
        digest = _hash(evidence)
        with self.pool.connection() as conn:
            row = (
                self._lease(conn, task)
                if isinstance(task, Mapping) and task.get("lease_token")
                else conn.execute(
                    "SELECT * FROM task WHERE id=%s FOR UPDATE", (task_id,)
                ).fetchone()
            )
            if not row:
                raise PersistenceError("Unknown task")
            prediction = conn.execute(
                "SELECT * FROM prediction WHERE task_id=%s", (task_id,)
            ).fetchone()
            variable = conn.execute(
                "SELECT * FROM variable WHERE id=%s", (row["variable_id"],)
            ).fetchone()
            if (
                not prediction
                or evidence["prediction_hash"] != prediction["canonical_hash"]
                or evidence["gold_hash"] != variable["gold_sha256"]
            ):
                raise EvidenceConflict(
                    "Evaluation input lineage differs from stored gold/prediction"
                )
            identity = _id("evaluation", f"{task_id}:{evidence['scorer_version']}")
            existing = conn.execute(
                "SELECT * FROM evaluation_item WHERE task_id=%s AND scorer_version=%s",
                (task_id, evidence["scorer_version"]),
            ).fetchone()
            if existing:
                if existing["evidence_hash"] != digest:
                    raise EvidenceConflict("Same scorer/task has conflicting evaluation evidence")
                return _row(existing)
            conn.execute(
                "INSERT INTO evaluation_item(id,task_id,prediction_id,scorer_version,evidence,evidence_hash) VALUES(%s,%s,%s,%s,%s,%s)",
                (
                    identity,
                    task_id,
                    prediction["id"],
                    evidence["scorer_version"],
                    Jsonb(evidence),
                    digest,
                ),
            )
            for mode, component, name, value in metrics:
                with localcontext() as context:
                    context.prec = 80
                    decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
                conn.execute(
                    "INSERT INTO metric_value(evaluation_id,mode,component,metric,numerator,denominator,value) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                    (
                        identity,
                        mode,
                        component,
                        name,
                        value.numerator,
                        value.denominator,
                        decimal_value,
                    ),
                )
            conn.execute(
                "UPDATE task SET state='complete',row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                (task_id,),
            )
            self._event(
                conn,
                task_id,
                row["state"],
                "complete",
                "evaluation_recorded",
                {"evaluation_id": identity},
            )
            return _row(
                conn.execute("SELECT * FROM evaluation_item WHERE id=%s", (identity,)).fetchone()
            )

    def heartbeat(self, lease: Mapping[str, Any], lease_seconds: int = 300) -> dict[str, Any]:
        """Extend one still-current lease using database time and unchanged fencing.

        Input is an active lease and positive duration. Return the renewed task
        row; expired/superseded leases raise StaleLeaseError. Only coordination
        timestamps/row version change; no attempt, prompt, or evidence is replaced.
        """
        if lease_seconds < 1:
            raise PersistenceError("Lease extension must be positive")
        with self.pool.connection() as conn:
            task = self._lease(conn, lease)
            row = conn.execute(
                "UPDATE task SET lease_expires_at=clock_timestamp()+%s*interval '1 second',heartbeat_at=clock_timestamp(),row_version=row_version+1 WHERE id=%s RETURNING *",
                (lease_seconds, task["id"]),
            ).fetchone()
            conn.execute(
                "UPDATE worker_session SET heartbeat_at=clock_timestamp() WHERE id=%s",
                (task["worker_id"],),
            )
            return {**_row(row), "task_id": str(row["id"])}

    def release(
        self, lease: Mapping[str, Any], state: str | None = None, reason: Any = None
    ) -> dict[str, Any]:
        """Release a current lease at a validated durable checkpoint.

        Inputs: current lease, optional next state and sanitized reason. Output is
        the resulting task row. Retry-pending requires safe prior evidence and a
        remaining attempt; complete requires evaluation. Stale/illegal transitions
        raise and roll back. Only lease/state/events change, never attempt counters.
        """
        allowed = {
            "queued",
            "request_persisted",
            "response_stored",
            "validated",
            "retry_pending",
            "prediction_ready",
            "complete",
            "operational_failed",
            "ambiguous_delivery",
            "paused_budget",
            "paused_configuration",
        }
        with self.pool.connection() as conn:
            task = self._lease(conn, lease)
            next_state = state or task["state"]
            if next_state not in allowed:
                raise PersistenceError("Unknown task checkpoint")
            if task["state"] == "complete" and next_state != "complete":
                raise PersistenceError("Completed tasks cannot be reopened")
            if next_state == "queued" and task["attempt_count"]:
                raise PersistenceError("A task with attempts cannot return to initial queued state")
            if next_state == "retry_pending":
                latest = conn.execute(
                    "SELECT a.id,s.delivery,v.valid,v.content_invalid FROM attempt a JOIN attempt_state s ON s.attempt_id=a.id LEFT JOIN validation_event v ON v.attempt_id=a.id WHERE a.task_id=%s ORDER BY a.attempt_number DESC LIMIT 1",
                    (task["id"],),
                ).fetchone()
                if (
                    task["attempt_count"] >= 3
                    or not latest
                    or latest["delivery"] in {"dispatch_started", "ambiguous_delivery"}
                    or latest["valid"] is True
                ):
                    raise AttemptLimitError("No safe numbered retry remains")
                if (
                    latest["delivery"] == "response_received"
                    and latest["content_invalid"] is not True
                ):
                    raise PersistenceError(
                        "A delivered response requires content-invalid evidence before retry"
                    )
            if (
                next_state == "complete"
                and not conn.execute(
                    "SELECT 1 FROM evaluation_item WHERE task_id=%s", (task["id"],)
                ).fetchone()
            ):
                raise PersistenceError("Task completion requires evaluation evidence")
            row = conn.execute(
                "UPDATE task SET state=%s,reason=%s,worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s RETURNING *",
                (next_state, Jsonb(_clean(reason)), task["id"]),
            ).fetchone()
            self._event(
                conn,
                str(task["id"]),
                task["state"],
                next_state,
                "lease_released",
                {"reason": _clean(reason), "fence": task["fence"]},
            )
            return {**_row(row), "task_id": str(row["id"])}

    def set_provider_state(
        self,
        campaign_id: str,
        provider: str,
        state: str,
        reason: Any = None,
        cooldown_until: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Persist one selected provider's dispatch pause or due-time cooldown.

        Inputs identify campaign/provider, ready/cooldown/paused/failed, sanitized
        reason and optional UTC due time. Output is the provider state row. Reject
        unknown scope or cooldown without a due time. This writes state/audit only;
        other providers and local response processing remain unaffected.
        """
        if state not in {"ready", "cooldown", "paused", "failed"} or (
            state == "cooldown" and cooldown_until is None
        ):
            raise PersistenceError("Invalid provider state or missing cooldown due time")
        content = _clean({"reason": reason, "cooldown_until": cooldown_until})
        with self.pool.connection() as conn:
            row = conn.execute(
                "UPDATE campaign_provider SET state=%s,reason=%s,cooldown_until=%s WHERE campaign_id=%s AND provider=%s RETURNING *",
                (state, Jsonb(_clean(reason)), cooldown_until, campaign_id, provider),
            ).fetchone()
            if not row:
                raise PersistenceError("Provider is not selected in this campaign")
            conn.execute(
                "INSERT INTO provider_event(id,campaign_id,provider,state,evidence) VALUES(%s,%s,%s,%s,%s)",
                (str(uuid.uuid4()), campaign_id, provider, state, Jsonb(content)),
            )
            return _row(row)

    def set_campaign_state(self, campaign_id: str, state: str) -> dict[str, Any]:
        """Pause/fail or resume a registered campaign without changing its identity.

        Inputs are explicit campaign ID and planned/running/paused/failed. Output is
        the row. Completion uses complete_campaign instead; terminal campaigns may
        not reopen. State writes do not modify selected models, caps, or evidence.
        """
        if state not in {"planned", "running", "paused", "failed"}:
            raise PersistenceError("Use completion verification for final campaign state")
        with self.pool.connection() as conn:
            row = self._campaign(conn, campaign_id, lock=True)
            if row["state"] == "complete":
                raise PersistenceError("Completed campaign cannot reopen")
            if not conn.execute(
                "SELECT 1 FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
            ).fetchone():
                raise PersistenceError("Campaign has no frozen plan")
            return _row(
                conn.execute(
                    "UPDATE campaign SET state=%s,updated_at=clock_timestamp() WHERE id=%s RETURNING *",
                    (state, campaign_id),
                ).fetchone()
            )

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Read one explicit campaign with plan, providers, runs, costs and states.

        Input is a UUID; output includes immutable configuration plus normalized
        counts, all run evidence, authorization and report IDs. Unknown identity or
        hash corruption raises. Uses read-only transactions and performs no dispatch.
        """
        with self.pool.connection() as conn:
            row = self._campaign(conn, campaign_id)
            plan = conn.execute(
                "SELECT * FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
            ).fetchone()
            providers = [
                _row(r)
                for r in conn.execute(
                    "SELECT * FROM campaign_provider WHERE campaign_id=%s ORDER BY provider",
                    (campaign_id,),
                )
            ]
            runs = [
                _row(r)
                for r in conn.execute(
                    "SELECT * FROM resolved_run WHERE campaign_id=%s ORDER BY fingerprint",
                    (campaign_id,),
                )
            ]
            for run in runs:
                if _hash(run["evidence"]) != run["evidence_hash"]:
                    raise EvidenceConflict("Stored run hash mismatch")
            states = {
                r["state"]: r["n"]
                for r in conn.execute(
                    "SELECT state,count(*) AS n FROM task WHERE campaign_id=%s GROUP BY state",
                    (campaign_id,),
                )
            }
            authorizations = [
                _row(r)
                for r in conn.execute(
                    "SELECT * FROM live_authorization WHERE campaign_id=%s ORDER BY created_at",
                    (campaign_id,),
                )
            ]
            return {
                **_row(row),
                "campaign_id": campaign_id,
                "plan": _row(plan),
                "providers": providers,
                "runs": runs,
                "states": states,
                "authorizations": authorizations,
                "rankings": [
                    str(r["id"])
                    for r in conn.execute(
                        "SELECT id FROM ranking_run WHERE campaign_id=%s ORDER BY created_at",
                        (campaign_id,),
                    )
                ],
                "reports": [
                    str(r["id"])
                    for r in conn.execute(
                        "SELECT id FROM report_manifest WHERE campaign_id=%s ORDER BY created_at",
                        (campaign_id,),
                    )
                ],
            }

    def get_campaign_artifact(
        self, campaign_id: str, kind: str = "experiment-bundle"
    ) -> dict[str, Any]:
        """Read one bound artifact's exact verified bytes for self-contained resume.

        Inputs are an explicit campaign ID and artifact kind. Output is the sole
        immutable artifact row, including content bytes. Missing or ambiguous kind
        and hash mismatch raise; no filesystem lookup or state changes occur.
        """
        with self.pool.connection() as conn:
            self._campaign(conn, campaign_id)
            rows = conn.execute(
                "SELECT a.* FROM artifact a JOIN campaign_artifact c ON c.artifact_id=a.id WHERE c.campaign_id=%s AND a.kind=%s",
                (campaign_id, kind),
            ).fetchall()
            if len(rows) != 1:
                raise PersistenceError("Campaign artifact kind is missing or ambiguous")
            row = rows[0]
            if (
                _hash(bytes(row["content"])) != row["sha256"]
                or _hash(
                    {
                        "kind": row["kind"],
                        "content_hash": row["sha256"],
                        "metadata": row["metadata"],
                    }
                )
                != row["evidence_hash"]
            ):
                raise EvidenceConflict("Bound campaign artifact failed hash verification")
            return _row(row)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Read and hash-verify one task's complete durable resume evidence.

        Input is an explicit task UUID. Output includes lease, run mapping, variable
        mapping/gold/category/source facts, attempts, selected prediction, and all
        evaluations. Unknown IDs or evidence drift raise; reads never mutate state.
        """
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM task WHERE id=%s", (task_id,)).fetchone()
            if not row:
                raise PersistenceError("Unknown task")
            run = conn.execute(
                "SELECT * FROM resolved_run WHERE id=%s", (row["run_id"],)
            ).fetchone()
            variable = conn.execute(
                "SELECT * FROM variable WHERE id=%s", (row["variable_id"],)
            ).fetchone()
            for evidence in (run, variable):
                if _hash(evidence["evidence"]) != evidence["evidence_hash"]:
                    raise EvidenceConflict("Run/variable evidence hash mismatch")
            if _hash(variable["gold"]) != variable["gold_sha256"] or (
                variable["source_content"] is not None
                and _hash(bytes(variable["source_content"])) != variable["source_sha256"]
            ):
                raise EvidenceConflict("Variable raw/gold hash mismatch")
            prediction = conn.execute(
                "SELECT * FROM prediction WHERE task_id=%s", (task_id,)
            ).fetchone()
            if prediction and (
                _hash(prediction["canonical"]) != prediction["canonical_hash"]
                or _hash(prediction["evidence"]) != prediction["evidence_hash"]
            ):
                raise EvidenceConflict("Prediction hash mismatch")
            evaluations = [
                _row(r)
                for r in conn.execute(
                    "SELECT * FROM evaluation_item WHERE task_id=%s ORDER BY scorer_version",
                    (task_id,),
                )
            ]
            for evaluation in evaluations:
                if _hash(evaluation["evidence"]) != evaluation["evidence_hash"]:
                    raise EvidenceConflict("Evaluation hash mismatch")
            attempts = self._attempts(conn, task_id)
            return {
                **_row(row),
                "task_id": task_id,
                "run": run["evidence"],
                "run_record": _row(run),
                "configuration": run["evidence"].get("configuration", run["evidence"]),
                "model_id": run["model_id"],
                "variable": {
                    **variable["evidence"],
                    "id": str(variable["id"]),
                    "corpus_id": str(variable["corpus_id"]),
                    "demonstration_order": variable["demonstration_order"],
                },
                "gold": variable["gold"],
                "attempts": attempts,
                "prediction": _row(prediction),
                "evaluations": evaluations,
                "evaluation": evaluations[-1]["evidence"] if evaluations else None,
            }

    def list_tasks(
        self, campaign_id: str, *, provider: str | None = None, state: str | None = None
    ) -> list[dict[str, Any]]:
        """List full task evidence for an explicit campaign and optional visible filter.

        Return canonical provider/fingerprint order. Filters only change this read;
        completeness is always checked against the full stored plan. Unknown campaign
        or hash failure raises; no implicit latest selection or evidence mutation.
        """
        with self.pool.connection() as conn:
            self._campaign(conn, campaign_id)
            ids = [
                str(r["id"])
                for r in conn.execute(
                    "SELECT id FROM task WHERE campaign_id=%s AND (%s::text IS NULL OR provider=%s) AND (%s::text IS NULL OR state=%s) ORDER BY provider,fingerprint",
                    (campaign_id, provider, provider, state, state),
                )
            ]
        return [self.get_task(identity) for identity in ids]

    @staticmethod
    def _attempts(conn: psycopg.Connection, task_id: str) -> list[dict[str, Any]]:
        """Read hash-verified attempts/responses/validation using the caller connection.

        Input is a task UUID. Return ordered immutable evidence and delivery state.
        Any byte/hash conflict raises; no processing or mutation occurs.
        """
        attempts = []
        for row in conn.execute(
            "SELECT a.*,s.delivery,s.dispatched_at,s.received_at FROM attempt a JOIN attempt_state s ON s.attempt_id=a.id WHERE a.task_id=%s ORDER BY a.attempt_number",
            (task_id,),
        ).fetchall():
            if (
                _hash(row["request_evidence"]) != row["request_hash"]
                or _hash(row["scientific_parameters"]) != row["scientific_hash"]
            ):
                raise EvidenceConflict("Attempt request/scientific hash mismatch")
            response = conn.execute(
                "SELECT * FROM response WHERE attempt_id=%s", (row["id"],)
            ).fetchone()
            validation = conn.execute(
                "SELECT * FROM validation_event WHERE attempt_id=%s", (row["id"],)
            ).fetchone()
            if response:
                raw_hash = (
                    _hash(bytes(response["raw_body"])) if response["raw_body"] is not None else None
                )
                if (
                    raw_hash != response["raw_sha256"]
                    or _hash({"raw_sha256": raw_hash, "metadata": response["evidence"]})
                    != response["evidence_hash"]
                ):
                    raise EvidenceConflict("Raw response or metadata hash mismatch")
            if validation and _hash(validation["evidence"]) != validation["evidence_hash"]:
                raise EvidenceConflict("Validation evidence hash mismatch")
            attempts.append(
                {**_row(row), "response": _row(response), "validation": _row(validation)}
            )
        return attempts

    def list_attempts(self, task_id: str) -> list[dict[str, Any]]:
        """Return ordered hash-verified attempt/response/validation evidence for one task.

        Input is a task UUID. Output is an empty list before generation; evidence
        drift raises EvidenceConflict. This is a read-only recovery operation.
        """
        with self.pool.connection() as conn:
            return self._attempts(conn, task_id)

    def reconcile(self, campaign_id: str) -> dict[str, Any]:
        """Reconcile expired work from durable evidence without resetting requests.

        Input is one campaign. Output is counts plus repaired task IDs and unresolved
        ambiguity. Active unexpired leases remain untouched. A stored raw response
        resumes local work; dispatch without a response becomes ambiguous, never a
        fresh call. Completed evidence is retained. Hash conflicts roll back repairs.
        """
        repaired, ambiguous = [], []
        with self.pool.connection() as conn:
            campaign = self._campaign(conn, campaign_id, lock=True)
            if not conn.execute(
                "SELECT 1 FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
            ).fetchone():
                raise PersistenceError("Cannot reconcile an unplanned campaign")
            tasks = conn.execute(
                "SELECT * FROM task WHERE campaign_id=%s AND state<>'complete' AND (lease_expires_at IS NULL OR lease_expires_at<=clock_timestamp()) FOR UPDATE SKIP LOCKED",
                (campaign_id,),
            ).fetchall()
            for task in tasks:
                attempts = self._attempts(conn, str(task["id"]))
                next_state = task["state"]
                if conn.execute(
                    "SELECT 1 FROM evaluation_item WHERE task_id=%s", (task["id"],)
                ).fetchone():
                    next_state = "complete"
                elif conn.execute(
                    "SELECT 1 FROM prediction WHERE task_id=%s", (task["id"],)
                ).fetchone():
                    next_state = "prediction_ready"
                elif attempts:
                    latest = attempts[-1]
                    if latest["response"] and latest["response"]["delivery"] == "response_received":
                        next_state = "validated" if latest["validation"] else "response_stored"
                    elif latest["delivery"] in {"dispatch_started", "ambiguous_delivery"}:
                        next_state = "ambiguous_delivery"
                        ambiguous.append(str(task["id"]))
                    elif latest["delivery"] == "not_dispatched" and latest["response"] is None:
                        next_state = "request_persisted"
                if task["lease_token"] is not None or next_state != task["state"]:
                    conn.execute(
                        "UPDATE task SET state=%s,worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                        (next_state, task["id"]),
                    )
                    self._event(
                        conn,
                        str(task["id"]),
                        task["state"],
                        next_state,
                        "reconciled",
                        {"attempt_count": task["attempt_count"]},
                    )
                    repaired.append(str(task["id"]))
            if campaign["state"] == "paused":
                conn.execute(
                    "UPDATE campaign SET state='planned',updated_at=clock_timestamp() WHERE id=%s",
                    (campaign_id,),
                )
        return {
            "campaign_id": campaign_id,
            "repaired_tasks": repaired,
            "ambiguous_tasks": ambiguous,
            "campaign": self.get_campaign(campaign_id),
        }

    def save_ranking(self, campaign_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        """Verify and retain a complete configuration ranking from exact DB receipts.

        Inputs: campaign ID and policy_version/configurations record. Each entry
        names configuration_id/provider/model_id and has rank plus exact primary
        fraction, or rank=None/reason for incomplete coverage. Output is immutable
        ranking row. Validate all selected configurations, each repetition's micro
        Close F1 and their arithmetic mean, and shared competition ranks. Missing
        evidence, changed formula, or incorrect rank raises and rolls back. This
        verifies arithmetic only and never invokes generation/evaluation.
        """
        evidence = _clean(record)
        _require(evidence, "policy_version", "configurations")
        if evidence["policy_version"] != "mean-repetition-micro-close-f1-v1" or not isinstance(
            evidence["configurations"], list
        ):
            raise PersistenceError("Unknown ranking policy or configuration list")
        entries = evidence["configurations"]
        if len({r.get("configuration_id") for r in entries}) != len(entries):
            raise EvidenceConflict("Ranking contains duplicate configuration identities")
        with self.pool.connection() as conn:
            self._campaign(conn, campaign_id)
            plan = conn.execute(
                "SELECT * FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
            ).fetchone()
            if not plan:
                raise PersistenceError("Ranking requires a frozen plan")
            runs = conn.execute(
                "SELECT * FROM resolved_run WHERE campaign_id=%s ORDER BY configuration_id,repetition",
                (campaign_id,),
            ).fetchall()
            configs: dict[str, list[dict[str, Any]]] = {}
            for run in runs:
                configs.setdefault(run["configuration_id"], []).append(run)
            if set(configs) != {r.get("configuration_id") for r in entries}:
                raise EvidenceConflict("Ranking must retain every planned configuration")
            expected_values: dict[str, Fraction] = {}
            for configuration_id, repetitions in configs.items():
                repetition_values = []
                for run in repetitions:
                    task_counts = conn.execute(
                        "SELECT count(*) AS total,count(*) FILTER(WHERE state='complete') AS complete FROM task WHERE run_id=%s",
                        (run["id"],),
                    ).fetchone()
                    if (
                        task_counts["total"] != plan["population_size"]
                        or task_counts["complete"] != plan["population_size"]
                    ):
                        break
                    facts = conn.execute(
                        """SELECT t.id AS task_id,m.metric,m.numerator,m.denominator FROM task t
                        JOIN evaluation_item e ON e.task_id=t.id JOIN metric_value m ON m.evaluation_id=e.id
                        WHERE t.run_id=%s AND e.scorer_version=%s AND m.mode='close' AND m.component='__variable__' AND m.metric IN ('tp','fp','fn')""",
                        (run["id"], record.get("scorer_version", _SCORER)),
                    ).fetchall()
                    if (
                        len(facts) != 3 * plan["population_size"]
                        or len({r["task_id"] for r in facts}) != plan["population_size"]
                    ):
                        break
                    sums = {
                        name: sum((_fraction(r) for r in facts if r["metric"] == name), Fraction(0))
                        for name in ("tp", "fp", "fn")
                    }
                    tp, fp, fn = (sums[k] for k in ("tp", "fp", "fn"))
                    repetition_values.append(
                        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
                    )
                if len(repetition_values) == len(repetitions):
                    expected_values[configuration_id] = sum(repetition_values, Fraction(0)) / len(
                        repetition_values
                    )
            for entry in entries:
                _require(entry, "configuration_id", "provider", "model_id")
                configuration_id = entry["configuration_id"]
                owned = configs[configuration_id][0]
                if entry["provider"] != owned["provider"] or entry["model_id"] != owned["model_id"]:
                    raise EvidenceConflict("Rank provider/model does not own its configuration")
                value = expected_values.get(configuration_id)
                if value is None:
                    if entry.get("rank") is not None or not entry.get("reason"):
                        raise EvidenceConflict(
                            "Incomplete configuration needs an unrankable reason"
                        )
                else:
                    expected_rank = 1 + sum(other > value for other in expected_values.values())
                    if (
                        entry.get("rank") != expected_rank
                        or _fraction(entry.get("primary")) != value
                    ):
                        raise EvidenceConflict(
                            "Rank or primary value differs from exact repetition micro Close F1"
                        )
            digest = _hash({"campaign_id": campaign_id, "record": evidence})
            identity = _id("ranking", digest)
            conn.execute(
                "INSERT INTO ranking_run(id,campaign_id,evidence,evidence_hash) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (identity, campaign_id, Jsonb(evidence), digest),
            )
            for entry in entries:
                value = expected_values.get(entry["configuration_id"])
                conn.execute(
                    "INSERT INTO configuration_rank(ranking_id,configuration_id,provider,model_id,rank,reason,primary_numerator,primary_denominator,evidence) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (
                        identity,
                        entry["configuration_id"],
                        entry["provider"],
                        entry["model_id"],
                        entry.get("rank"),
                        entry.get("reason"),
                        value.numerator if value is not None else None,
                        value.denominator if value is not None else None,
                        Jsonb(entry),
                    ),
                )
            return _row(
                conn.execute("SELECT * FROM ranking_run WHERE id=%s", (identity,)).fetchone()
            )

    def save_report(self, campaign_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        """Store a hash-identified derived report manifest without changing evidence.

        Inputs: campaign ID and schema_version=1.0/campaign_id/kind/files/final
        manifest. Each file names path, SHA-256, byte_length. Output is immutable
        manifest row. Final manifests require all tasks complete and a verified
        ranking; incomplete reports must explicitly set final=False. This writes
        only PostgreSQL lineage, never output files or model calls.
        """
        evidence = _clean(record)
        _require(evidence, "schema_version", "campaign_id", "kind", "files", "final")
        if (
            evidence["schema_version"] != "1.0"
            or evidence["campaign_id"] != campaign_id
            or not isinstance(evidence["final"], bool)
        ):
            raise PersistenceError("Report manifest identity/version is invalid")
        if not isinstance(evidence["files"], list) or not evidence["files"]:
            raise PersistenceError("Report must name at least one derived artifact")
        for file in evidence["files"]:
            _require(file, "path", "sha256", "byte_length")
            if (
                not _SHA.fullmatch(file["sha256"])
                or not isinstance(file["byte_length"], int)
                or file["byte_length"] < 0
            ):
                raise PersistenceError("Invalid report file hash/length")
        digest = _hash(evidence)
        identity = _id("report", digest)
        with self.pool.connection() as conn:
            self._campaign(conn, campaign_id)
            if evidence["final"]:
                self._verify_complete(conn, campaign_id, require_report=False)
            conn.execute(
                "INSERT INTO report_manifest(id,campaign_id,evidence,evidence_hash,final) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (identity, campaign_id, Jsonb(evidence), digest, evidence["final"]),
            )
            return _row(
                conn.execute("SELECT * FROM report_manifest WHERE id=%s", (identity,)).fetchone()
            )

    @staticmethod
    def _verify_complete(
        conn: psycopg.Connection, campaign_id: str, *, require_report: bool
    ) -> None:
        """Verify complete union coverage and final ranking/report in a transaction.

        Input names one campaign and whether a final report is already required.
        Return None on exact completeness; missing tasks/evaluations/ranks/reports
        raise PersistenceError. No state changes or scientific recomputation occur.
        """
        plan = conn.execute(
            "SELECT * FROM campaign_plan WHERE campaign_id=%s", (campaign_id,)
        ).fetchone()
        if not plan:
            raise PersistenceError("Campaign has no complete frozen plan")
        counts = conn.execute(
            "SELECT count(*) AS total,count(*) FILTER(WHERE state='complete') AS complete FROM task WHERE campaign_id=%s",
            (campaign_id,),
        ).fetchone()
        if counts["total"] != plan["task_count"] or counts["complete"] != plan["task_count"]:
            raise PersistenceError("Campaign has unfinished selected-provider tasks")
        if conn.execute(
            "SELECT 1 FROM task t WHERE t.campaign_id=%s AND (NOT EXISTS(SELECT 1 FROM prediction p WHERE p.task_id=t.id) OR NOT EXISTS(SELECT 1 FROM evaluation_item e WHERE e.task_id=t.id)) LIMIT 1",
            (campaign_id,),
        ).fetchone():
            raise PersistenceError("Completed task is missing prediction/evaluation evidence")
        expected = conn.execute(
            "SELECT count(DISTINCT configuration_id) AS n FROM resolved_run WHERE campaign_id=%s",
            (campaign_id,),
        ).fetchone()["n"]
        ranking = conn.execute(
            "SELECT r.id FROM ranking_run r JOIN configuration_rank c ON c.ranking_id=r.id WHERE r.campaign_id=%s GROUP BY r.id HAVING count(*)=%s AND count(c.rank)=%s LIMIT 1",
            (campaign_id, expected, expected),
        ).fetchone()
        if not ranking:
            raise PersistenceError("Complete campaign requires a full verified combined ranking")
        if (
            require_report
            and not conn.execute(
                "SELECT 1 FROM report_manifest WHERE campaign_id=%s AND final LIMIT 1",
                (campaign_id,),
            ).fetchone()
        ):
            raise PersistenceError("Complete campaign requires a durable final report manifest")

    def complete_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Mark complete only after all selected work, ranking and final reports exist.

        Input is an explicit campaign ID. Output is its completed immutable-identity
        row. Exact coverage and required evidence are checked in the same transaction;
        gaps raise without changing state. Repeating successful completion is safe.
        """
        with self.pool.connection() as conn:
            self._campaign(conn, campaign_id, lock=True)
            self._verify_complete(conn, campaign_id, require_report=True)
            return _row(
                conn.execute(
                    "UPDATE campaign SET state='complete',updated_at=clock_timestamp() WHERE id=%s RETURNING *",
                    (campaign_id,),
                ).fetchone()
            )

    def select_prediction(
        self,
        lease: Mapping[str, Any],
        prediction: Mapping[str, Any],
        attempt: Mapping[str, Any] | str,
        terminal_invalid: bool = False,
    ) -> dict[str, Any]:
        """Select one valid canonical prediction or a justified terminal empty result.

        Inputs: active lease, six-field canonical mapping, owned source attempt, and
        optional terminal_invalid=True. Output is the immutable prediction. Empty
        terminal selection requires exactly three delivered content-invalid receipts;
        ordinary selection requires validated content. Conflicting second selection
        or stale ownership raises. The task advances to prediction_ready, not complete.
        """
        attempt_id = attempt if isinstance(attempt, str) else str(attempt["id"])
        canonical = _clean(prediction)
        if set(canonical) != _FIELDS:
            raise PersistenceError("Prediction must contain exactly the six decomposition fields")
        with self.pool.connection() as conn:
            task = self._lease(conn, lease)
            row = conn.execute(
                "SELECT a.*,v.valid FROM attempt a LEFT JOIN validation_event v ON v.attempt_id=a.id WHERE a.id=%s",
                (attempt_id,),
            ).fetchone()
            if not row or row["task_id"] != task["id"]:
                raise EvidenceConflict("Selected attempt belongs to another task")
            if terminal_invalid:
                count = conn.execute(
                    "SELECT count(*) AS n FROM attempt a JOIN response r ON r.attempt_id=a.id JOIN validation_event v ON v.attempt_id=a.id WHERE a.task_id=%s AND r.delivery='response_received' AND v.content_invalid",
                    (task["id"],),
                ).fetchone()["n"]
                if (
                    count != 3
                    or row["attempt_number"] != 3
                    or any(value not in ("", []) for value in canonical.values())
                ):
                    raise PersistenceError(
                        "Terminal-empty selection requires exactly three content-invalid delivered attempts"
                    )
            elif row["valid"] is not True:
                raise PersistenceError("Selected prediction requires a passed validation receipt")
            content = {
                "canonical": canonical,
                "attempt_id": attempt_id,
                "terminal_invalid": terminal_invalid,
                "failure_policy": "explicit-empty-after-three-invalid-v1"
                if terminal_invalid
                else None,
            }
            digest = _hash(content)
            identity = _id("prediction", str(task["id"]))
            conn.execute(
                "INSERT INTO prediction(id,task_id,attempt_id,terminal_invalid,canonical,canonical_hash,evidence,evidence_hash) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (
                    identity,
                    task["id"],
                    attempt_id,
                    terminal_invalid,
                    Jsonb(canonical),
                    _hash(canonical),
                    Jsonb(content),
                    digest,
                ),
            )
            stored = conn.execute(
                "SELECT * FROM prediction WHERE task_id=%s", (task["id"],)
            ).fetchone()
            if stored["evidence_hash"] != digest:
                raise EvidenceConflict("A task cannot select a different second prediction")
            if task["state"] != "complete":
                conn.execute(
                    "UPDATE task SET state='prediction_ready',row_version=row_version+1,updated_at=clock_timestamp() WHERE id=%s",
                    (task["id"],),
                )
                if task["state"] != "prediction_ready":
                    self._event(
                        conn,
                        str(task["id"]),
                        task["state"],
                        "prediction_ready",
                        "prediction_selected",
                        {"prediction_id": identity, "terminal_invalid": terminal_invalid},
                    )
            return _row(stored)
