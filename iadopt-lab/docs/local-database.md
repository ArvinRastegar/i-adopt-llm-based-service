# Local PostgreSQL, DBeaver, and Credentials

## What “PostgreSQL access” means

The experiment needs a database server plus a host, port, database name, username, and password. PostgreSQL stores the evidence; DBeaver is a separate client for browsing tables and running analysis queries. The server can run on the same computer as both the experiment and DBeaver—AWS is not required. See PostgreSQL's [client/server explanation](https://www.postgresql.org/docs/16/tutorial-arch.html) and DBeaver's [connection guide](https://dbeaver.com/docs/dbeaver/Create-Connection/).

The proposed starting point is local PostgreSQL 16 in Docker, with persistent data and backups. A remote PostgreSQL connection remains possible later without changing scientific logic. The latest user instruction requests documentation first; no container, database, role, password, or implementation script has been created by this update.

## Read-only local checks

The documentation update found Docker CLI `28.4.0` and Docker Compose `v2.39.4-desktop.1`. The Docker server could not be reached at its configured socket. This is a readiness check, not proof that no PostgreSQL exists elsewhere. Neither `psql` nor `postgres` was found on the current shell's PATH. No service was started or installed.

The repository-root `.env` is ignored by Git and not tracked. Its credential contents were not inspected or tested. The owner identifies `OPENROUTER_API_KEY` and `PSNC_API_KEY` as the available keys; existence of a key is not verified provider access and is not live-call authorization.

## Proposed local deployment contract

| Setting | Proposed value or behavior |
|---|---|
| Server | PostgreSQL major 16; exact patch/image digest pinned and tested during implementation |
| Compose project | An isolated I-ADOPT Lab project, separate from existing services |
| Host interface | `127.0.0.1` only; no public or LAN exposure |
| Host port | `5433`, subject to a collision check before provisioning |
| Container port | `5432` |
| Database | `iadopt_lab` |
| Runtime role | `iadopt_lab_app`, limited to the documented experiment operations |
| Migration role | `iadopt_lab_migrator`, used only for schema changes |
| Analysis role | `iadopt_lab_reader`, read-only access for DBeaver |
| Credentials | Unique local secrets generated at setup, never committed or shown in logs |
| Storage | Named Docker volume mounted at `/var/lib/postgresql/data` for PostgreSQL 16 |
| Recovery | Database backups plus verified restore; volume persistence alone is not a backup |

The [official PostgreSQL image documentation](https://hub.docker.com/_/postgres/) describes initialization credentials and the PostgreSQL 16 data-volume location. Pin the image rather than using `latest`. Bootstrap credentials must not become the normal experiment or DBeaver account. Require password authentication and do not enable host `trust` authentication. Scope all initialization and migrations to the new database; never alter an existing service database.

Compose/runtime definitions and setup documentation will live inside `iadopt-lab/`. Database bytes live in the managed volume, not Git; backup artifacts are private and ignored. The existing parent `.env` is an explicitly supplied secret source, not a runtime dependency on legacy service code. This distinction preserves the isolated experiment structure without duplicating credentials.

## Planned setup inputs, actions, outputs, and acceptance

This is a contract for future implementation, not an executable setup command.

- **Inputs:** an available Docker engine, pinned PostgreSQL image, verified free loopback port, isolated project/volume names, generated local database credentials, and reviewed forward migrations.
- **Actions:** create only the new service and durable volume; wait for database health; create least-privilege roles; apply migrations transactionally where supported; verify the schema/version; prepare redacted connection instructions and backup/restore procedure.
- **Outputs:** a reachable isolated database, migration/version evidence, a private runtime `DATABASE_URL`, reader credentials delivered securely, and a successful read-only connection check. Provider API keys are never passed to the database container.
- **Failure behavior:** unavailable engine, occupied port, insufficient disk, authentication failure, or migration failure blocks setup with a specific diagnostic. No automatic database deletion, volume recreation, broad permissions, or modification of another database is allowed.
- **Acceptance:** the experiment can write/read its own fixture evidence, the reader can query but cannot mutate it, a service restart preserves that evidence, backup restore into a separate test database reproduces it, and no secret enters logs or committed files.

During future setup, supply `DATABASE_URL` through the process environment or a private explicitly selected environment file. Do not modify the user's existing `.env` merely as part of documentation cleanup. The host-side URL will use the chosen loopback port; an experiment running inside Compose must use the database service name and container port, not `localhost`. Store only redacted connection metadata in evidence.

## Connect DBeaver after provisioning

In DBeaver, create a PostgreSQL connection with these proposed settings:

| DBeaver field | Value |
|---|---|
| Host | `127.0.0.1` |
| Port | `5433` or the port confirmed during setup |
| Database | `iadopt_lab` |
| Username | `iadopt_lab_reader` |
| Password | The separate reader password supplied privately during setup |

Use **Test Connection**, then finish the connection. DBeaver may need its PostgreSQL driver. Its [connection guide](https://dbeaver.com/docs/dbeaver/Create-Connection/) describes selecting a driver, entering connection settings, and testing access. Prefer the read-only account for analysis so browsing results cannot accidentally change experiment evidence.

## Existing API credentials

The planned CLI accepts an explicit `--env-file` path; from `iadopt-lab/`, `../.env` identifies the owner-supplied repository-root file. This option does not exist until implementation.

The loader contract is:

1. Parse the explicitly selected file as data, never `source` it or evaluate shell commands, command substitutions, or variable interpolation.
2. Use only the declared active-provider environment names and required database connection setting. Ignore unrelated settings from the existing service.
3. Let existing process-environment values take precedence. Never overwrite the file or other services' environment.
4. Validate only presence/required shape during offline preparation. Connectivity checks are separate operations; credentials alone cannot enable live generation.
5. Keep secret values, raw `.env` contents, the file's content hash, and unredacted DSNs out of prompts, snapshots, reports, errors, and PostgreSQL evidence. Record only non-secret variable names/presence and redacted diagnostics.

## Keeping long experiments running

Local execution requires the computer, Docker/database service, and experiment process to remain available. Sleep, shutdown, or a stopped process interrupts progress; durable checkpoints support `resume` afterward but cannot run on a powered-off computer. A restart policy for the database does not by itself restart the experiment runner. Before a long live run, document the chosen process-supervision and power settings; no such settings are changed by this document.

A remote or always-on host is a later option if local availability becomes inconvenient, not a prerequisite for writing or testing the scripts. Regardless of host, back up the database and test restore before relying on it as the only copy of results.
