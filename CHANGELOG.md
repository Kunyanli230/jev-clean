# Changelog

## 0.2.0 — 2026-10-02

- Add `idac profile` for offline quality counts, issues, evidence and bounded
  candidate previews on the original input, with optional JSON file/stdout output.
- Add `idac baseline` and the reusable `idac.baseline.run_baseline` API. The
  previous example script and `BaselineRunner` import remain available.
- Save baseline snapshots with stable row identities and null values, plus
  source/config/snapshot hashes. Evaluation checks these against the input,
  schema and CSV exports before computing cell metrics.
- Fix baseline evaluation after deleting duplicates from the middle of the
  dataset. Legacy CSV-only exports with incomplete removal ledgers now show
  unavailable cell metrics rather than assigning new row identities.
- Support Python 3.11–3.13, add `idac --version`, and check the offline suite,
  lint and package builds in a Python-version CI matrix.
- Report empty CSVs as CLI input errors and correctly quote removed-row CSV headers.

The Jev model, question definitions, decision thresholds and cleaning operations
remain at their v0.1 contracts. This release makes no new live-model accuracy claims.

## 0.1.0

Initial experimental release: seven deterministic cleaning phases, Jev decision
distributions, policy gates, validation, run artifacts, replay and rollback.
