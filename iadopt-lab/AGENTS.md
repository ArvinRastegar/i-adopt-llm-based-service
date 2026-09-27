# iadopt-lab

Reproducible experiments measuring how well language models decompose scientific-variable
definitions into I-ADOPT lexical components. Results are scientific evidence, so most rules
here exist to stop a change quietly invalidating work that has already been paid for and run.

## Commands

```bash
uv run pytest                    # full suite; DB-backed tests skip silently without a DSN
uv run pytest -q -rs             # ...and list what skipped, which is the number that matters
uv run ruff check .              # lint; this is the gate
iadopt-lab preflight             # draft readiness; --live applies the full dispatch gate
iadopt-lab plan --synthetic      # expand the grid; omit --synthetic for the real population
```

Python 3.12 only, `uv` for everything. Never `pip install` into this project.

## Traps

**A green test run can be mostly untested.** `uv run pytest` reports a large pass count while
silently skipping every PostgreSQL integration and recovery test when
`IADOPT_LAB_TEST_DATABASE_URL` is unset. Anything touching `persistence/`, leases, or campaign
recovery must be verified with the DSN exported — check the `-rs` skip list, never the pass count.

**`ruff format` is not enforced here.** Only `[tool.ruff.lint]` is configured. Roughly half the
files are unformatted and that is the status quo, not a defect. Do not reformat; a formatting
pass would bury a real diff in noise.

**Most of `docs/` is deliberately historical.** `read-only-review-2026-09-09.md`,
`migration-v2.0.1.md` and `repository-audit.md` record what was true when they were written and
say so in their own text. Correcting their "stale" facts is data loss. The same applies to any
dated entry in `campaign-log.md` or `implementation-progress.md`.

**Contracts in `docs/components/` are two-layer.** Planning signatures use provisional names,
and a section at the end of each file ("Implementation interface", "Implementation boundary",
"Initial implementation mapping" — the heading varies) records what was actually built. A
planning name that no longer exists in `src/` is usually not drift; check for that section
before changing anything.

## Non-negotiables

- **Three provider requests per task, ever.** Waiting, cooldowns and retries never create a
  fourth. Only the workflow layer may schedule another numbered attempt.
- **Raw responses commit before interpretation.** A response that was paid for is never lost to
  a parsing or accounting failure downstream.
- **No transaction is held during HTTP or scoring.** `Repository` is synchronous; the async
  runner reaches it through `asyncio.to_thread`.
- **Live calls are gated.** A hashed estimate, a disclosure, and separate explicit
  authorization must exist before any billed dispatch. Never relax a gate to make a run proceed.
- **Changing `parameters.yml` or any hashed implementation path changes the plan hash**, which
  means a new campaign rather than a continuation. Never edit one to make an interrupted
  campaign easier to finish.
- **Exact quantities stay exact.** Money as decimal strings, scores as numerator/denominator
  receipts. Never introduce a float where a fraction is recorded.
- **Synthetic runs are labelled and cannot become results.** A real campaign requires all 97
  targets; a synthetic fixture uses a declared three-variable subset.

## Conventions

- Every public and private function carries a typed docstring: one-line summary, then
  `Args:`, `Returns:`, `Raises:`, `Side Effects:`, and determinism or idempotency where relevant.
  Match the surrounding density; these are terse, single-line fields, not prose blocks.
- Comments explain *why*, and the good ones name the incident — see `workflow.py`'s
  `_DEPLOYMENT_FAILURES`. A comment restating the code is noise.
- `psycopg` 3 with a pool. No ORM.
- Scientific identity is canonical JSON plus SHA-256; operational timestamps live separately
  and never enter a hash.
- `src/iadopt_eval` is a standalone scorer with no dependency on the experiment package. Keep it
  that way — it is the frozen measurement protocol.

## Documentation

`DECISIONS.md` is the decision index (`D-001`…). A settled decision is **superseded by a new
entry, never edited** — the record of a reversal usually matters more than the original.

A behaviour change needs a decision entry. A new executable module needs a contract in
`docs/components/` first. `outputs/` is gitignored, so anything a campaign produced that matters
is recorded in `docs/campaign-log.md` or it is lost.
