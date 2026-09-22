# IDAC v0.1 implementation status

Checked on 2026-09-22 on Ubuntu 22.04 (WSL) with Python 3.11.16 and uv 0.12.17.
Every number below was produced by running the commands in this document.

## Fixed versions

| Item | Version |
|---|---|
| Python | 3.11.16 (`>=3.11,<3.12`) |
| uv | 0.12.17 |
| pandas | 3.0.6 |
| pydantic | 2.13.5 |
| PyYAML | 6.0.3 |
| Typer | 0.27.2 |
| typesafe-sdk | 0.7.1 |
| pytest | 9.1.1 |
| ruff | 0.16.8 |
| Jev model | `jev-1.13.0` (fixed in `src/idac/settings.py`) |
| API base URL | `https://api.typesafe.ai` via the SDK `/v1/systemone` |

## Test results

```text
$ uv run pytest -q
112 passed in 8.6s
```

The suite covers parsing and precision,
median/mode rules, IQR-only behavior, exact duplicates, gate boundaries, answer
validation, batching, hard/semantic validation, retry/budget behavior at the
SDK boundary, storage/rollback, distribution math, replay, reports and the
fake-client end-to-end pipeline. `tests/test_acceptance_fixes.py` adds 20
regression tests for the four acceptance-review findings listed below. Tests
make no external network calls.

## Demo status

| Run | Model identity | Result |
|---|---|---|
| `runs/baseline` (`examples/run_baseline.py`) | none (rule baseline) | 200 rows, 10 duplicates removed, 7 commits, 0 model calls, no hard-validation failures |
| `runs/demo` (FakeDecisionClient fixture) | **fake** | outcome `partial`; 7 committed versions; 12 candidates, 12 committed decisions |
| `runs/final` (real TypeSafe API) | **real** | outcome `partial`; 5 committed plans across v000-v005; 10 candidates, 6 committed decisions |
| `runs/final-rolled` (copy of `runs/final`, rolled back) | real | current version `v000`; history and cards retained; stale evaluation hidden |

`runs/demo` was produced with `tests/fake_client.py` and is explicitly labeled
`model_identity: fake` in the manifest, report and cards. It demonstrates the
full pipeline but is **not** a real Jev run. `runs/final` is a real run; its
numbers are below.

Fake demo evaluation (`uv run idac evaluate --run runs/demo --truth
examples/data/ground_truth.csv`):

- quality original -> cleaned: parse failures 30 -> 1 (the intentionally
  ambiguous date), format issues 40 -> 0, range violations 6 -> 0, exact
  duplicate rows 10 -> 0;
- corruption repair coverage 58/58 = 1.0000; truth-match rate 32/58 = 0.5517;
  wrong modifications of unpolluted correct cells: 0;
- numeric imputation: filled 10/10, MAE 25666.119 over 10 filled values
  (unfilled values are never counted as zero error);
- categorical imputation: filled 5/5, accuracy 1/5 = 0.2;
- duplicates: removed 10/10, wrong deletions 0, missed deletions 0;
- API accounting: 14 request payloads (7 decision + 7 postcheck), 0 retries,
  1680 known input tokens, 0 unknown-usage requests, 0 actual SDK attempts
  (fake client), run elapsed 0.548 s;
- integrity metrics: distribution/rubric save 12/12 = 1.0, evidence/policy/
  final-state traceability 12/12 = 1.0, offline gate replay consistency
  12/12 = 1.0.

The rule baseline reaches the same before/after quality counts on this dataset
(repair coverage 1.0, wrong modifications 0, numeric MAE 25666.119) with no
model calls; its probability/distribution metrics are reported as `N/A`, never
as fabricated probabilities.

## Real API status: verified on 2026-09-22

`idac clean` was executed against the live TypeSafe API (`jev-1.13.0`) with the
demo data. The run is `runs/final` (`model_identity: real`):

- outcome `partial` (stop reason `unresolved_issues`); 5 committed plans across
  versions `v000`-`v005`; 10 candidates, 6 committed decisions, 4 abstentions.
- 12 actual SDK attempts (7 decision requests + 5 postcheck requests), 0
  retries, 16,064 known input tokens, 0 unknown-usage requests, 11.4 s wall
  clock, no budget pressure.
- semantic postcheck Noul probabilities: 0.04, 0.13, 0.09, 0.11, 0.11, 0.12
  (threshold 0.20); every committed batch passed.
- offline gate replay: 10/10 valid decisions reproduce the stored gate; request
  evidence and per-candidate hashes verified for all 10 decisions.
- quality original -> cleaned: parse failures 30 -> 13, format issues 40 -> 28,
  range violations 6 -> 0, missing cells 0 -> 20, exact duplicate rows 10 -> 1.
- corruption repair coverage 36/58 = 0.6207; wrong modifications of unpolluted
  correct cells: 0; duplicates removed 9/10 (0 wrong, 1 dependency-blocked);
  numeric imputation 0/10 filled, MAE `N/A`; categorical imputation 0/5 filled,
  accuracy `N/A` because the city null-normalization candidate was rejected and
  the masked cells still hold the empty missing marker.
- integrity: distribution/rubric save 10/10, traceability 10/10, replay
  consistency 10/10, 0 unavailable.
- every decision's `context_ref`/`questions_ref` resolves to the saved
  `requests/req-NNNN.json`, and the stored `context_hash`/`questions_hash`
  reproduce from that payload.

Real model behavior observed:

- Jev abstained on several legitimate operations: whitespace trimming
  (`APPLICABLE_PROBABILITY` 0.73-0.78), city null tokens (0.76), income
  thousands casting (`CHOICE_CONFIDENCE` 0.54), and age imputation
  (`EXPECTED_RISK_LEVEL` 0.60, `HIGH_RISK_MASS` 0.16). The gate rejected them
  with the exact rule and observed values recorded in the cards.
- In two of the real verification runs Jev returned a risk distribution summing
  to 0.99 (for example `{0: 0.55, 1: 0.27, 2: 0.17}`). The fixed validator
  rejected it as `INVALID_DISTRIBUTION` and the candidate abstained; no silent
  renormalization, clipping or zero-filling was applied. The final run had no
  such response.
- Accepted candidates were committed and committed batches passed the Noul
  postcheck. One duplicate group (rows whose income format issue stayed
  unresolved) was dependency-blocked and not removed, which is why 9 of 10
  copies were deleted.
- Run-to-run answers vary, as expected: an earlier real run accepted the name
  trim and rejected the age imputation with a different reason. Replay checks
  the stored policy logic, not whether a repeated request would give the same
  answer.

`runs/final-rolled` is a copy of the real run after `idac rollback --run ...
--to-version v000`: 210 rows and 0 removed rows restored, all 10 decision cards
retained as history, and the `v005` evaluation no longer shown in the report.
`idac evaluate` now refreshes `report.md` with the current version's metrics,
while `inspect` distinguishes saved historical metrics from the evidence
integrity check for the current run.

## Acceptance-review hardening (4 fixes)

The acceptance review found four places where a passing report could have been
produced from the validator's or evaluator's own bookkeeping instead of the real
data. All four are fixed and covered by `tests/test_acceptance_fixes.py` (20
regression tests):

1. **Hard validation compares the actual snapshot and removal set.** The
   validator now re-derives the changed-cell set from `state_before` vs
   `state_after` rows and the removed-row set from the row-id difference, and
   requires `planned == actual == logged` (`PLAN_DIFF_MATCH`, `NO_EXTRA_CHANGES`,
   `REMOVED_ROWS_MATCH`, `ROW_COUNT_CONSERVATION`). A tampered snapshot or a
   removed row that the executor never logged can no longer pass.
2. **Imputation metrics exclude missing markers and invalid values.**
   `_valid_imputed_value` rejects `null`, declared null tokens, unparseable
   values, out-of-range numbers and categorical values outside `allowed_values`
   before counting a masked cell as filled. On the real run this changed the
   categorical metric from an incorrect 5/5 filled to an honest 0/5 filled
   (accuracy `N/A`) because the rejected normalization left empty markers.
3. **Offline replay verifies request evidence and hashes.** `replay_decision`
   now requires `run_dir`, resolves `context_ref`/`questions_ref` inside the run
   directory, locates the candidate context and question subset, and recomputes
   `context_hash`/`questions_hash`. Missing, tampered, out-of-path or
   unparseable evidence is reported as not replayable; `integrity_metrics`
   counts those as unavailable instead of consistent, so a report can no longer
   show a passing replay rate without evidence.
4. **Risk legend must equal the requested ordered criteria.**
   `summarize_risk` compares the returned legend against the exact criteria sent
   in the request (three ordered levels); reversed, shortened or reworded
   legends are `INVALID_DISTRIBUTION`. `agents.py` passes the question's own
   criteria to the validator.

## Known limitations and unresolved items

- The demo's ambiguous date `03/04/2024` is intentionally unresolved
  (`PARSE_AMBIGUOUS`); this is why every demo outcome is `partial`, not
  `completed`.
- The real run's remaining unresolved issues (city whitespace, city null
  tokens, income casting, age imputation) are deliberate model abstentions, not
  code failures; the rule baseline repairs more on this dataset and is faster.
  IDAC's course focus is the complete decision record, not beating the baseline.
- The real run's numeric imputation filled 0/10 masked values because the model
  rejected the age candidate; MAE is `N/A` over 0 filled values rather than a
  fabricated zero error. Categorical imputation also reports 0/5 filled because
  the rejected city null-normalization left the empty missing markers in place;
  a retained missing marker or an out-of-range value is never counted as a
  successful fill (see the hardening section below).
- Thresholds (0.80/0.80/0.50/0.10/0.20/0.03) are course policy choices, not
  calibrated accuracy claims.
- The IQR flag is informational only and never rewrites data.
- Token usage can exceed the 200,000 limit by the size of the last request, as
  documented in the plan; the real demo used 15,189 input tokens.
- Run artifacts under `runs/` are gitignored and not committed; the numbers
  above are recorded here.

## Reproduction

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/check_environment.py

# Demo data and rule baseline (no credentials needed)
uv run python examples/generate_demo.py
uv run python examples/run_baseline.py

# Real clean (requires TYPESAFE_API_KEY); evaluate auto-detects runs/baseline
# as the sibling baseline directory when present
read -rsp 'TypeSafe API key: ' TYPESAFE_API_KEY; echo; export TYPESAFE_API_KEY
uv run idac clean --input examples/data/dirty.csv --config configs/demo.yaml --output runs/demo
uv run idac inspect --run runs/demo
uv run idac evaluate --run runs/demo --truth examples/data/ground_truth.csv
uv run idac rollback --run runs/demo --to-version v000
```
