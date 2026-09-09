# Generated Outputs

## Purpose

`outputs/` is the destination for reproducible, derived reports and exports created from PostgreSQL evidence. It is not the source of truth for experiment state or results.

No generated output has been created yet, because no campaign has produced results. `iadopt-lab report --out` writes here once observations exist.

## Planned output classes

The planning/reporting phases may create:

- Campaign summary with selected providers, per-provider progress, and combined expected/completed counts
- Per-variable Exact/Close Precision, Recall, and F1 with supporting contributions
- Per-component Exact/Close Precision, Recall, and F1 with supporting contributions
- Exact and Close repetition and overall aggregate metrics
- Prompt-by-shot comparison
- Temperature and repetition summary
- Complete configuration ranking with shared ties and unranked reasons
- Provider/model comparison and explicitly filtered provider views
- Reasoning-enabled versus reasoning-disabled comparison
- Category and subcategory summary
- Validation and retry summary
- Token, latency, and cost summary
- Pre-run cost estimate and disclosure report, before any live dispatch
- Reproducibility manifest
- CSV, JSON, and XLSX exports

Every output must be regenerated from a versioned database query or reporting specification.

## Required provenance

Each output artifact records, in PostgreSQL or an accompanying manifest:

- Campaign and evaluation-run IDs, complete frozen selected-provider set, and exact provider/model identities for each result row or aggregate membership
- Corpus, demonstration, and evaluation-population manifest hashes
- Resolved configuration hash
- Prompt and schema hashes
- Scorer version and hash
- Query/report version and parameters
- Creation timestamp in UTC
- Output SHA-256 and byte size
- Software environment identity

An output must state its evaluated denominator, valid prediction count, terminal-invalid count, and operationally incomplete count.

All result rows retain provider and model identity; aggregate rows declare their provider/model membership explicitly. Combined reports include every selected provider's planned configurations, even when some are incomplete. Provider-specific views retain their filter and never stand in for the combined campaign report. Final campaign completion requires all frozen provider/model tasks to be scored and the combined ranking and required reports to be stored; an interim report must clearly identify the missing work.

The configuration-ranking export is a derived view over the complete stored results. It ranks the union of selected provider/model configurations together, includes every fully resolved configuration, its provider/model identity, rank inputs and tie outcome, and links back to all item-, component-, repetition-, and aggregate-level database evidence. Each configuration uses the same 97-variable population and shared scientific protocol. Ranking never deletes or hides a non-winning configuration.

Accepted policy `mean-repetition-micro-close-f1-v1` ranks by the arithmetic mean of unrounded repetition micro Close F1 values, each calculated from all 97 variables. Exact ties receive shared competition rank. Per-variable mean F1 is a different statistic and does not determine rank.

D-029's current campaign has one repetition at every temperature, so the ranking mean is that single micro Close F1. Keep the repetition column and all score evidence. Reports must identify the singleton design and cannot claim measured run-to-run variability or repetition-based confidence intervals; per-variable/category analyses remain distinct.

Mean, median, variance, mode, standard deviation, range, and IQR over per-variable scores, and further variability/uncertainty analyses, are deferred until the database is populated. Their later exports must carry versioned scope and formula definitions. Their absence does not make an otherwise complete experiment result incomplete.

Cost reports distinguish explicit no-charge, metered actual, estimated, and unavailable values, retaining each provider's billing basis, currency, optional provider/global caps, and reservation evidence even when uncapped. PSNC is initially owner-reported no-charge in this user's access context; reporting zero monetary cost must not imply zero tokens or universally free service. OpenRouter's metered costs use the frozen price-card evidence. A combined cost total states its completeness and never silently treats unknown cost as zero.

Before live execution, present the owner with the required `pre-run-estimate-v1` report: the frozen plan/configuration hash, dated price-card and FX provenance, scenario assumptions, planned and maximum-three-attempt call counts, input/output/reasoning token assumptions and correction-prompt bounds, and per-model/provider/combined costs. Retain its receipt/hash and disclosure record in PostgreSQL alongside separate explicit live-authorization evidence. No generation call is needed to produce the initial estimate. The report must state that it is an estimate, not an exact future bill, a spending cap, or permission to run. Null caps mean uncapped execution after authorization; actual spend exceeding an estimate does not by itself stop the workflow. Changed resolved plans or price/FX bases require updated estimates and disclosure, with prior reports retained.

## Authority and mutability

Outputs may be deleted and regenerated because PostgreSQL contains the authoritative facts. Replacing an output file does not replace or update an evaluation run.

If the query, scorer, grouping, or input campaign changes, create a new artifact identity. Never overwrite a file in a way that hides which evidence produced it.

## Sensitive and large evidence

Full prompts, raw response bodies, validation candidates, errors, and state transitions remain in PostgreSQL. A report may include sanitized excerpts only when its specification explicitly requests them.

Exports must not contain credentials, authorization headers, environment secrets, or unredacted sensitive provider metadata.

## Repository policy

`.gitignore` excludes generated contents here (`outputs/*`) while retaining this README. A scientific result is committed only through a deliberate, documented publication decision with its provenance manifest; ad hoc local outputs are not committed.

Mock dry-run reports must be visibly labeled synthetic/test and must never be combined with live scientific results.
