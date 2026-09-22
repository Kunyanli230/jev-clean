# IDAC

**Interpretable Data Auto-Cleaner** is an experimental, decision-first data
cleaning system powered by [Jev](https://docs.typesafe.ai/introduction). Instead
of asking a model to rewrite data directly, IDAC computes bounded repair
candidates locally and uses Jev to assess whether each candidate is applicable
and semantically safe.

Version **0.1.0** is the first public development release. It focuses on
auditable CSV cleaning, conservative automation, deterministic execution, and
offline-replayable model decisions.

> IDAC is an independent project built with the TypeSafe SDK. It is
> not an official TypeSafe product.

## Why IDAC?

Most data-cleaning pipelines are either fully rule-based or difficult to audit
once a model is involved. IDAC separates computation from judgment:

1. deterministic profilers identify issues;
2. local code produces exact, finite repair candidates;
3. Jev returns explicit probability distributions for action, applicability,
   and semantic risk;
4. a fixed policy gate accepts or abstains;
5. a deterministic executor applies accepted candidates to a copy;
6. hard and semantic validation decide whether to commit or reject the batch.

The model never generates executable code or arbitrary mutations. Every
candidate—accepted, rejected, or abstained—receives a decision record containing
its evidence, complete distributions, expected risk level, policy checks, and
final commit status.

## v0.1 capabilities

- Clean one UTF-8 CSV with a required YAML schema and policy configuration.
- Handle whitespace, declared null tokens, numeric formats, declared date
  formats, exact duplicates, declared-range violations, and bounded
  median/mode imputation.
- Route candidates to standardization, duplicate, outlier, and missing-value
  decision agents.
- Preserve full Jev Choice, Noul, and ordered Score results.
- Gate repairs on confidence, applicability, expected risk, and high-risk
  probability.
- Validate actual snapshot diffs independently from executor logs.
- Commit immutable versioned snapshots or reject the entire candidate batch.
- Replay decisions offline while verifying saved request evidence and hashes.
- Roll back to any committed snapshot without calling the model.
- Compare a Jev-guided run with a deterministic rule baseline.

Current limits are 10,000 rows and 30 columns per input. IDAC v0.1 intentionally
does not perform fuzzy entity matching, free-text completion, unit conversion,
model training, or direct database mutation.

## Architecture

~~~text
CSV + YAML configuration
        |
        v
Deterministic profiler and candidate builders
        |
        v
Issue router -> four specialized decision agents
        |
        v
Jev distributions -> deterministic policy gate
        |
        v
Repair planner -> deterministic executor
        |
        v
Hard validator -> Jev semantic postcheck
        |
        +---- pass ----> commit immutable snapshot
        |
        +---- fail ----> reject batch and retain original values
~~~

See [docs/architecture.md](docs/architecture.md) for component boundaries,
decision flow, storage, budgets, and replay behavior.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- A TypeSafe API key for live Jev decisions

The project pins typesafe-sdk 0.7.1 and uses jev-1.13.0.

## Installation

~~~bash
cd IDAC
uv sync --locked
uv run idac --help
~~~

For the existing WSL development environment:

~~~powershell
wsl -d COMP5584HDC --cd /home/hdc/projects/IDAC
~~~

Verify the local toolchain without making an API request:

~~~bash
uv run python scripts/check_environment.py
uv run pytest -q
uv run ruff check src tests examples scripts
~~~

## Configure Jev

IDAC reads the TypeSafe credential from TYPESAFE_API_KEY. To enter it for the
current shell without placing the value in shell history:

~~~bash
read -rsp 'TypeSafe API key: ' TYPESAFE_API_KEY
echo
export TYPESAFE_API_KEY
~~~

.env.example documents the required variable, but IDAC does not automatically
load environment files. Never commit an API key. The clean command stops when
the key is absent; it never silently substitutes a fake client or the rule
baseline.

## Quick start

The repository includes a reproducible demo dataset and configuration:

~~~bash
uv run idac clean \
  --input examples/data/dirty.csv \
  --config configs/demo.yaml \
  --output runs/my-first-run

uv run idac inspect --run runs/my-first-run

uv run idac evaluate \
  --run runs/my-first-run \
  --truth examples/data/ground_truth.csv
~~~

To inspect a complete decision card:

~~~bash
uv run idac inspect \
  --run runs/my-first-run \
  --candidate-id CANDIDATE_ID
~~~

To restore a committed snapshot without contacting Jev:

~~~bash
uv run idac rollback --run runs/my-first-run --to-version v000
~~~

Output directories must not already exist. The source CSV is never overwritten.

## Configuration

The YAML configuration describes table semantics and explicitly authorizes
operations per column:

~~~yaml
table_description: Customer profile table
record_grain: One row per customer

columns:
  customer_id:
    type: string
    description: Stable customer identifier
    protected: true
    null_tokens: [""]
    allowed_operations: []

  age:
    type: integer
    description: Customer age in years
    min_value: 18
    max_value: 100
    null_tokens: ["", "NA"]
    allowed_operations:
      - trim_whitespace
      - normalize_null_tokens
      - cast_numeric
      - invalidate_out_of_range
      - impute_missing

duplicates:
  enabled: true
  compare_columns: [customer_id]
~~~

Unknown configuration fields are rejected. Protected columns cannot be
modified, and an operation not listed for a column cannot enter a repair plan.
See [configs/demo.yaml](configs/demo.yaml) for a complete example.

## Decision policy

IDAC records complete model outputs rather than reducing them to one label.
For each candidate, the policy evaluates:

- the selected action and its SDK confidence;
- P(applicable) from a Jev Noul answer;
- P(risk level = 0/1/2) from an ordered Score answer;
- the recomputed expected risk level;
- the probability mass assigned to the highest-risk level;
- agreement between the SDK score and the locally recomputed expectation;
- deterministic eligibility checks tied to the current snapshot.

Thresholds are versioned in src/idac/settings.py and copied into every decision
record. They are IDAC policy defaults, not TypeSafe accuracy guarantees. An
abstention is a normal safety outcome, not a pipeline failure.

## Run artifacts

Each run is a self-contained audit bundle:

~~~text
manifest.json          current version, model identity, outcome
cleaned.csv            export of the current committed snapshot
removed_rows.csv       removed duplicate rows
issues.json            detected and unresolved issues
decisions.jsonl        distributions and gate decisions
decision_traces.jsonl  planner, validation, and commit outcomes
changes.jsonl          committed cell-level changes
validation.json        deterministic and semantic checks
audit.jsonl            append-only lifecycle events
requests/              bounded model contexts and questions
snapshots/             immutable dataset versions
report.md              run summary
decision_cards.md      human-readable decision evidence
~~~

Request references are verified against stored hashes during offline replay.
Missing or modified evidence makes a decision non-replayable.

## Reproducible demo and baseline

Regenerate the seeded demo data and run the no-model baseline:

~~~bash
uv run python examples/generate_demo.py
uv run python examples/run_baseline.py
~~~

Ground truth and the corruption manifest are used only by the evaluation
command. They are never included in model context or candidate generation.
The baseline uses the same deterministic profilers, candidate builders,
executor, and hard validator, but makes no model calls.

Recorded environment and live-API verification details are available in
[docs/implementation_status.md](docs/implementation_status.md).

## Development

~~~bash
uv sync --locked
uv run pytest -q
uv run ruff check src tests examples scripts
~~~

Tests use a clearly labeled deterministic fake client at the SDK boundary and
make no external requests. Live Jev runs remain separate and are always marked
with model_identity: real.

Contributions that improve decision traceability, validator independence,
configuration safety, or Jev integration are especially welcome. Please keep
new behavior deterministic outside the model boundary, add regression tests,
and document any change to decision semantics or policy thresholds.

## Project status

IDAC v0.1 is experimental. Its audit and rollback mechanisms are designed for
inspection, but the software has not been certified for production or
high-stakes data processing. Review configuration, policy thresholds, and
generated changes before using results in downstream systems.
