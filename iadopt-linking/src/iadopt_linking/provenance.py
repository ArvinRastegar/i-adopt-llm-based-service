"""What produced a run: its code, corpus, environment, each invocation and the services' releases.

A run may be interrupted and resumed, possibly by different code versions of the report side, on
another day or machine. ``provenance.json`` journals every invocation, so the manifest describes
the evidence that is there rather than only the last invocation that touched it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import RunError
from .sources import OLS_ONTOLOGY_ENDPOINT, QUDT_ENDPOINT, QUDT_RELEASE_QUERY

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The modules that decide what is retrieved and scored and which gold is read. A results folder
# is bound to their hash; the report-side modules may change and the report be rebuilt.
EVIDENCE_FILES = (
    "iadopt_linking/gold.py", "iadopt_linking/http.py", "iadopt_linking/population.py",
    "iadopt_linking/recorder.py", "iadopt_linking/rerank.py", "iadopt_linking/retrieve.py",
    "iadopt_linking/score.py", "iadopt_linking/sources.py", "iadopt_linking_eval/identity.py")
JOURNAL_SCHEMA = 1
LIBRARIES = ("torch", "sentence_transformers", "transformers")
OM_RELEASE_FIELDS = ("version", "loaded", "fileHash", "numberOfTerms", "status")


def now_utc() -> str:
    """Timestamp for the journal and manifest.

    Args: none.
    Returns: UTC time to the second, ISO 8601.
    Raises: None.
    Side effects: reads the clock.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def evidence_code_identity(project_root: Path = PROJECT_ROOT) -> str:
    """Hash the modules that produce a run's evidence.

    Args: project_root: the iadopt-linking folder.
    Returns: a hexadecimal SHA-256 over EVIDENCE_FILES, names and bytes.
    Raises: OSError if a listed module is missing.
    Side effects: filesystem reads.
    """
    digest = hashlib.sha256()
    for name in EVIDENCE_FILES:
        digest.update(name.encode() + b"\0" + (project_root / "src" / name).read_bytes())
    return digest.hexdigest()


def code_identity(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Identify the code that is running.

    Args: project_root: the iadopt-linking folder.
    Returns: a hash over the project's source and pyproject, the git commit the tree is based on
    and whether the tree differs from it (results and outputs excluded). A tree that was never
    committed is dirty, so the hash, not the commit, identifies it.
    Raises: None.
    Side effects: runs git read-only.
    """
    digest = hashlib.sha256()
    files = sorted([*project_root.joinpath("src").rglob("*.py"), project_root / "pyproject.toml"])
    for path in files:
        digest.update(path.relative_to(project_root).as_posix().encode() + b"\0" + path.read_bytes())

    def git(*args: str) -> str | None:
        """Run a read-only git command.

        Args: args: git arguments.
        Returns: stripped output, or None when git fails.
        Raises: None.
        Side effects: runs git.
        """
        try:
            done = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True,
                                  text=True, timeout=30, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    status = git("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)outputs")
    return {"source_sha256": digest.hexdigest(), "files": len(files),
            "git_base_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def corpus_identity(lab_root: Path) -> dict[str, Any]:
    """Identify the corpus release the gold came from.

    Args: lab_root: iadopt-lab root.
    Returns: tag, commit and manifest hashes of the pinned corpus.
    Raises: OSError or KeyError if the manifest is missing or malformed.
    Side effects: filesystem read.
    """
    manifest = json.loads((lab_root / "data/manifests/corpus-v2.0.1.json").read_bytes())
    lock = json.loads((lab_root / "data/manifests/corpus-source-lock-v2.0.1.json").read_bytes())
    return {"tag": lock["tag"], "commit": lock["commit"], "repository": lock["repository"],
            "corpus_manifest_sha256": manifest["manifest_sha256"],
            "source_lock_sha256": lock["manifest_sha256"]}


def fetch_om_release(client: Any) -> dict[str, Any]:
    """Record which OM release the search service currently serves.

    Args: client: HTTP client.
    Returns: probe time and the service's version, load time, file hash, term count and status;
    {"probed_at", "error"} when the service did not give a release.
    Raises: None.
    Side effects: one network request.
    """
    probe: dict[str, Any] = {"probed_at": now_utc()}
    fetched = client.get(OLS_ONTOLOGY_ENDPOINT, None, {"Accept": "application/json"})
    if fetched.text is None:
        return {**probe, "error": fetched.error}
    try:
        data = json.loads(fetched.text)
    except json.JSONDecodeError as exc:
        return {**probe, "error": f"release response is not JSON ({exc})"}
    if not isinstance(data, dict) or not any(data.get(k) for k in ("version", "loaded", "fileHash")):
        return {**probe, "error": "release response has no version, load time or file hash"}
    return {**probe, **{key: data.get(key) for key in OM_RELEASE_FIELDS}}


def fetch_qudt_release(client: Any) -> dict[str, Any]:
    """Record when the QUDT endpoint's data was last modified.

    Args: client: HTTP client.
    Returns: probe time and ``graph_modified``; {"probed_at", "error"} when the endpoint did not
    give a modification time.
    Raises: None.
    Side effects: one network request.
    """
    probe: dict[str, Any] = {"probed_at": now_utc()}
    fetched = client.get(QUDT_ENDPOINT, {"query": QUDT_RELEASE_QUERY},
                         {"Accept": "application/sparql-results+json"})
    if fetched.text is None:
        return {**probe, "error": fetched.error}
    try:
        rows = json.loads(fetched.text)["results"]["bindings"]
        modified = rows[0]["modified"]["value"] if rows else None
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return {**probe, "error": "unexpected release response"}
    if modified is None:
        return {**probe, "error": "the endpoint reports no modification time"}
    return {**probe, "graph_modified": modified}


def probe_releases(client: Any) -> dict[str, Any]:
    """Ask both services what data they currently serve.

    Args: client: HTTP client.
    Returns: {"om": ..., "qudt": ...}, each a probe as fetch_*_release returns it.
    Raises: None.
    Side effects: two network requests.
    """
    return {"om": fetch_om_release(client), "qudt": fetch_qudt_release(client)}


def merge_releases(invocations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine the release probes of every invocation.

    Args: invocations: journal invocations in order.
    Returns: per service the latest usable probe, how many probes were made and failed, and how
    many distinct releases were seen; Wikidata has no release identifier.
    Raises: None.
    Side effects: none.
    """
    merged: dict[str, Any] = {}
    for name in ("om", "qudt"):
        probes = [inv["source_releases"][name] for inv in invocations
                  if inv.get("source_releases") and name in inv["source_releases"]]
        usable = [p for p in probes if "error" not in p]
        releases = {json.dumps({k: v for k, v in p.items() if k != "probed_at"}, sort_keys=True)
                    for p in usable}
        merged[name] = {"latest": usable[-1] if usable else None, "probes": len(probes),
                        "failed_probes": len(probes) - len(usable),
                        "distinct_releases": len(releases)}
    merged["wikidata"] = "live search API; no release identifier"
    return merged


def load_journal(path: Path, *, evidence_exists: bool) -> dict[str, Any]:
    """Read a results folder's journal, or start one.

    Args: path: provenance.json; evidence_exists: whether the folder already holds retrieval or
    scoring evidence.
    Returns: the journal; a new empty one when the folder is new.
    Raises: RunError for evidence without a journal, or a journal that cannot be read.
    Side effects: filesystem read.
    """
    if not path.is_file():
        if evidence_exists:
            raise RunError(f"{path.parent} holds evidence but no {path.name}; without it nothing "
                           "says what produced that evidence. Use a new --out folder.")
        return {"schema": JOURNAL_SCHEMA, "invocations": [], "device_check": None}
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"{path} cannot be read ({exc}); it is evidence and is not recreated") from exc
    if (not isinstance(journal, dict) or journal.get("schema") != JOURNAL_SCHEMA
            or not isinstance(journal.get("invocations"), list)):
        raise RunError(f"{path} is not a version {JOURNAL_SCHEMA} journal")
    journal.setdefault("device_check", None)
    return journal


def begin_invocation(journal: dict[str, Any], *, evidence_sha256: str,
                     http_settings: Mapping[str, Any]) -> dict[str, Any]:
    """Open a journal entry for this invocation.

    Args: journal: the journal; evidence_sha256: hash of the evidence-producing code; http_settings:
    how the client behaves.
    Returns: the new entry, already appended to the journal; its outcome is "running" until the
    invocation ends.
    Raises: None.
    Side effects: runs git read-only; hashes the source files.
    """
    invocation: dict[str, Any] = {
        "started_at": now_utc(), "finished_at": None, "outcome": "running",
        "code": code_identity(), "evidence_code_sha256": evidence_sha256,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "http": dict(http_settings), "seconds": {}, "retrieval": None, "scoring": None,
        "model": None, "source_releases": None}
    journal["invocations"].append(invocation)
    return invocation


def summed_seconds(invocations: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Add up the time each phase took over all invocations.

    Args: invocations: journal invocations.
    Returns: phase -> seconds to a tenth.
    Raises: None.
    Side effects: none.
    """
    total: Counter[str] = Counter()
    for invocation in invocations:
        total.update(invocation["seconds"])
    return {phase: round(seconds, 1) for phase, seconds in sorted(total.items())}


def invocation_summaries(invocations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Condense the journal for the manifest.

    Args: invocations: journal invocations.
    Returns: one entry per invocation: times, outcome, what it did, where it ran and which code.
    Raises: KeyError for a malformed entry.
    Side effects: none.
    """
    return [{"started_at": i["started_at"], "finished_at": i["finished_at"],
             "outcome": i["outcome"], "seconds": {k: round(v, 1) for k, v in i["seconds"].items()},
             "retrieval": i["retrieval"], "scoring": i["scoring"], "python": i["python"],
             "platform": i["platform"], "source_sha256": i["code"]["source_sha256"],
             "git_dirty": i["code"]["git_dirty"]} for i in invocations]


def model_summary(records: Mapping[str, Mapping[str, Any]],
                  invocations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the model and devices that produced the stored scores.

    Args: records: scores records by slot id; invocations: journal invocations.
    Returns: model, revision and batch size from the records, how many slots each device scored
    and the library versions of the latest invocation that loaded the model.
    Raises: None.
    Side effects: none.
    """
    scoring = [record.get("scoring") or {} for record in records.values()]
    first = scoring[0] if scoring else {}
    devices = Counter(s.get("device") for s in scoring)
    loaded = [i["model"] for i in invocations if i.get("model")]
    return {"model": first.get("model"), "revision": first.get("revision"),
            "batch_size": first.get("batch_size"),
            "devices": dict(sorted(devices.items(), key=lambda item: str(item[0]))),
            "libraries": {name: loaded[-1].get(name) for name in LIBRARIES} if loaded else {}}
