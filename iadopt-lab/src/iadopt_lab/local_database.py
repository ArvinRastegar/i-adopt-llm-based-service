"""Isolated local PostgreSQL provisioning without exposing or reusing API secrets."""

from __future__ import annotations

import json
import os
import re
import secrets
import socket
import subprocess
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from .canonical import canonical_json_bytes
from .domain import LabError

ROLES = ("bootstrap", "migrator", "app", "reader")
DATABASES = ("iadopt_lab", "iadopt_lab_test")


def _private_write(path: Path, content: bytes) -> None:
    """Create a new owner-only secret file, refusing overwrite or symlink traversal."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def local_dsn(project_root: str | Path, role: str = "app", database: str = "iadopt_lab") -> str:
    """Resolve one explicit private local connection string for a database boundary."""
    if role not in ROLES or database not in {*DATABASES, "postgres"}:
        raise LabError("Unknown local database role or database")
    path = Path(project_root) / ".runtime/local-postgres.json"
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise LabError("Local database secret file must be owner-only and not a symlink")
    values = json.loads(path.read_bytes())
    return make_conninfo(host="127.0.0.1", port="5433", dbname=database,
                        user="iadopt_lab_" + role, password=values["passwords"][role],
                        connect_timeout=5)


def prepare_local_database(project_root: str | Path, image: str) -> dict:
    """Create private setup artifacts for a previously pulled, digest-pinned PostgreSQL image."""
    if not re.fullmatch(r"(?:docker\.io/library/)?postgres@sha256:[a-f0-9]{64}", image):
        raise LabError("PostgreSQL setup requires the verified official image digest")
    root = Path(project_root).resolve()
    runtime = root / ".runtime"
    if runtime.is_symlink():
        raise LabError("Runtime directory cannot be a symlink")
    runtime.mkdir(mode=0o700, exist_ok=True)
    os.chmod(runtime, 0o700)
    secret_path = runtime / "local-postgres.json"
    if secret_path.exists():
        saved = json.loads(secret_path.read_bytes())
        if saved["image"] != image:
            raise LabError("Existing local database image differs; upgrades require explicit migration")
    else:
        saved = {"version": "local-postgres-v1", "image": image,
                 "passwords": {role: secrets.token_urlsafe(36) for role in ROLES}}
        _private_write(secret_path, canonical_json_bytes(saved))
    password_path = runtime / "postgres-bootstrap-password"
    if not password_path.exists():
        _private_write(password_path, saved["passwords"]["bootstrap"].encode())
    elif password_path.read_text() != saved["passwords"]["bootstrap"]:
        raise LabError("Bootstrap secret artifacts disagree")
    env_path = runtime / "postgres.env"
    env_bytes = ("IADOPT_POSTGRES_IMAGE=" + image + "\n").encode()
    if not env_path.exists():
        _private_write(env_path, env_bytes)
    elif env_path.read_bytes() != env_bytes:
        raise LabError("Existing Compose image binding disagrees")
    return {"image": image, "host": "127.0.0.1", "port": 5433, "database": "iadopt_lab",
            "reader_user": "iadopt_lab_reader", "private_credentials_file": str(secret_path)}


def start_local_database(project_root: str | Path) -> dict:
    """Start only the isolated Compose project, then initialize least-privilege databases."""
    root = Path(project_root).resolve()
    command = ["docker", "compose", "--env-file", str(root / ".runtime/postgres.env"),
               "-f", str(root / "compose.yml"), "-p", "iadopt-lab"]
    owned = subprocess.run(command + ["ps", "-q", "postgres"], capture_output=True, text=True, check=False)
    with socket.socket() as probe:
        occupied = probe.connect_ex(("127.0.0.1", 5433)) == 0
    if occupied and not owned.stdout.strip():
        raise LabError("Port5433 is occupied by a service outside the lab Compose project")
    result = subprocess.run(command + ["up", "-d", "--wait", "--wait-timeout", "60"],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise LabError("Isolated Compose startup failed; inspect docker compose logs locally")
    return initialize_local_roles(root)


def initialize_local_roles(project_root: str | Path) -> dict:
    """Initialize dedicated roles, two owned databases, migrations and read/write grants."""
    from .persistence import migrate

    root = Path(project_root)
    saved = json.loads((root / ".runtime/local-postgres.json").read_bytes())
    with psycopg.connect(local_dsn(root, "bootstrap", "postgres"), autocommit=True) as conn:
        version = conn.execute("SHOW server_version").fetchone()[0]
        for role in ROLES[1:]:
            name = "iadopt_lab_" + role
            if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (name,)).fetchone():
                conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE").format(
                    sql.Identifier(name), sql.Literal(saved["passwords"][role])))
        for database in DATABASES:
            owner = conn.execute("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname=%s", (database,)).fetchone()
            if owner and owner[0] != "iadopt_lab_migrator":
                raise LabError("Refusing a database not owned by the lab migration role")
            if not owner:
                conn.execute(sql.SQL("CREATE DATABASE {} OWNER iadopt_lab_migrator").format(sql.Identifier(database)))
            conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(database)))
            conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO iadopt_lab_app,iadopt_lab_reader").format(sql.Identifier(database)))
    migrations = {}
    for database in DATABASES:
        dsn = local_dsn(root, "migrator", database)
        migrations[database] = migrate(dsn)
        with psycopg.connect(dsn) as conn:
            conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
            conn.execute("GRANT USAGE ON SCHEMA iadopt_lab TO iadopt_lab_app,iadopt_lab_reader")
            conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA iadopt_lab TO iadopt_lab_app,iadopt_lab_reader")
            conn.execute("GRANT INSERT ON ALL TABLES IN SCHEMA iadopt_lab TO iadopt_lab_app")
            conn.execute("REVOKE INSERT ON iadopt_lab.schema_migration FROM iadopt_lab_app")
            for table in ("campaign", "campaign_provider", "task", "attempt_state", "worker_session"):
                conn.execute(sql.SQL("GRANT UPDATE ON iadopt_lab.{} TO iadopt_lab_app").format(sql.Identifier(table)))
            conn.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA iadopt_lab GRANT SELECT ON TABLES TO iadopt_lab_reader")
    return {"server_version": version, "databases": list(DATABASES), "migrations": migrations,
            "host": "127.0.0.1", "port": 5433, "reader_user": "iadopt_lab_reader"}
