# January scorer evidence

`randomShotsPhaseOne.py.txt` is an exact unedited copy from the parent
`i-adopt-llm-based-service` repository, retained for attribution and regression
evidence under that repository's existing authorship/license terms; no new
license or permission grant is asserted by this copy.

- Original path: `benchmarking_example/randomShotsPhaseOne.py`
- Source tag: `V1.1-Experiment`
- Immutable commit: `b9683d2242aa5ca5b987440ea4b6f70bc1253c7e`
- SHA-256: `2d4a6271fd86a076127dfb3f85589391cff9744efdfdee8ec570b52c74921df0`
- Retained on: 2026-09-07
- Purpose: test-only behavioral provenance, including explicit approved differences.

Do not execute or import this historical runner: its original initialization
includes provider setup and filesystem effects. The regression fixture checks
the full source hash, extracts only the named pure scoring function AST nodes,
and supplies offline cosine fixtures. The active runtime never reads this file.
Keeping the exact source here makes these tests independent of parent-repository
git history and old benchmark directories.
