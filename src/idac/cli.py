"""CSV profiling, explicit rule baseline, Jev cleaning and run inspection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from . import __version__
from .baseline import run_baseline
from .config import ConfigError
from .evaluation import evaluate_run, integrity_metrics
from .explanation import decision_card, render_card_markdown, render_decision_table
from .models import DecisionTrace, ModelStatus, decision_from_log
from .orchestrator import run_clean
from .policy import replay_decision
from .profile import profile_csv
from .report import write_run_reports
from .settings import API_KEY_ENV
from .storage import RUN_FILES, RunStore, StorageError, atomic_write_text, load_run_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Jev Clean: inspect CSVs offline, run a rule baseline, or clean with Jev decisions.",
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", is_eager=True, help="Show package version."),
) -> None:
    if version:
        typer.echo(f"Jev Clean {__version__}")
        raise typer.Exit()


def _fail(message: str, code: int = 1) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=code)


@app.command()
def profile(
    input: Path = typer.Option(..., "--input", help="Input UTF-8 CSV path."),
    config: Path = typer.Option(..., "--config", help="YAML configuration path."),
    output: Path | None = typer.Option(None, "--output", help="New JSON profile file."),
    json_output: bool = typer.Option(False, "--json", help="Print the complete JSON profile."),
) -> None:
    """Inspect original cells and authorized candidate previews; makes no model calls."""
    if output is not None and output.exists():
        _fail(f"output file already exists: {output}")
    try:
        result = profile_csv(input, config)
    except (ConfigError, StorageError) as error:
        _fail(str(error))
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output is not None:
        atomic_write_text(output, payload)
    if json_output:
        typer.echo(payload, nl=False)
        return
    typer.echo(f"rows: {result['rows']}; columns: {len(result['columns'])}")
    typer.echo(f"quality: {result['quality']}")
    typer.echo(f"issues: {result['issue_counts']}; candidates: {result['candidate_count']}")
    typer.echo(result["candidate_semantics"])
    typer.echo("model calls: 0; probability distributions: N/A")
    if output is not None:
        typer.echo(f"profile: {output}")


@app.command()
def baseline(
    input: Path = typer.Option(..., "--input", help="Input UTF-8 CSV path."),
    config: Path = typer.Option(..., "--config", help="YAML configuration path."),
    output: Path = typer.Option(..., "--output", help="New rule-baseline output directory."),
) -> None:
    """Apply authorized local rules with hard validation; no Jev gate or semantic postcheck."""
    try:
        summary = run_baseline(input, config, output)
    except (ConfigError, StorageError) as error:
        _fail(str(error))
    typer.echo(
        f"rows: {summary['rows']}; removed: {summary['removed_rows']}; "
        f"commits: {summary['versions']}"
    )
    typer.echo("model calls: 0; probability distributions: N/A")
    typer.echo(f"cleaned: {output / 'cleaned.csv'}")
    if summary["hard_failures"]:
        _fail(f"hard validation failures: {summary['hard_failures']}")


@app.command()
def clean(
    input: Path = typer.Option(..., "--input", help="Input UTF-8 CSV path."),
    config: Path = typer.Option(..., "--config", help="YAML configuration path."),
    output: Path = typer.Option(..., "--output", help="Output run directory (must not exist)."),
) -> None:
    """Clean a CSV with the fixed seven sub-phases and Jev decision records."""
    if output.exists():
        _fail(f"output directory already exists: {output}")
    if not input.is_file():
        _fail(f"input CSV not found: {input}")
    if not config.is_file():
        _fail(f"configuration not found: {config}")
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        _fail(
            f"{API_KEY_ENV} is not set; clean never falls back to a fake client or the rule "
            "baseline. Export the key and retry, or run examples/run_baseline.py offline."
        )
    from .typesafe_client import ConfigurationError

    try:
        result = run_clean(input, config, output, api_key=api_key)
    except (ConfigurationError, ConfigError, StorageError) as error:
        _fail(str(error))
    typer.echo(f"outcome: {result.outcome.value}")
    typer.echo(f"stop_reason: {result.stop_reason}")
    typer.echo(f"current_version: {result.current_version}")
    typer.echo(f"versions: {', '.join(result.versions)}")
    typer.echo(f"model_identity: {result.model_identity}")
    typer.echo(f"report: {output / RUN_FILES['report']}")


def _load_store(run: Path) -> RunStore:
    if not (run / RUN_FILES["manifest"]).is_file():
        _fail(f"not an IDAC run directory: {run}")
    return RunStore(run)


@app.command()
def inspect(
    run: Path = typer.Option(..., "--run", help="Run directory produced by clean."),
    candidate_id: str | None = typer.Option(
        None, "--candidate-id", help="Show the full decision card for one candidate."
    ),
) -> None:
    """Show the run summary, decision table and offline replay checks."""
    store = _load_store(run)
    manifest = store.manifest()
    decisions = [decision_from_log(entry) for entry in store.read_decisions()]
    traces = [DecisionTrace.model_validate(entry) for entry in store.read_traces()]
    if candidate_id:
        matches = [record for record in decisions if record.candidate_id == candidate_id]
        if not matches:
            _fail(f"no decision record for candidate {candidate_id}")
        trace_map = {trace.decision_id: trace for trace in traces}
        for record in matches:
            typer.echo(render_card_markdown(decision_card(record, trace_map.get(record.id))))
        return
    typer.echo(f"run: {manifest['run_id']}")
    typer.echo(
        f"outcome: {manifest.get('outcome')} ({manifest.get('stop_reason')}); "
        f"current version: {manifest['current_version']}; "
        f"chain: {', '.join(store.ancestor_chain())}"
    )
    typer.echo(f"model: {manifest.get('model')} ({manifest.get('model_identity')})")
    typer.echo(
        f"decisions: {len(decisions)}; committed: "
        f"{sum(1 for trace in traces if trace.commit_status == 'committed')}"
    )
    typer.echo("")
    typer.echo(render_decision_table(decisions, traces))
    valid = [record for record in decisions if record.model_status == ModelStatus.OK]
    replayable = consistent = 0
    for record in valid:
        result = replay_decision(record, store.run_dir)
        if result.replayable:
            replayable += 1
            consistent += int(result.consistent)
    if replayable:
        typer.echo(
            f"offline policy replay: {consistent}/{len(valid)} valid decisions reproduce the "
            f"stored gate ({replayable} replayable, {len(valid) - replayable} not replayable)"
        )
    else:
        typer.echo(f"offline policy replay: no replayable decisions ({len(valid)} valid decisions)")
    evaluation = store.read_evaluation(manifest["current_version"])
    typer.echo(f"current evidence integrity: {json.dumps(integrity_metrics(store))}")
    if evaluation:
        typer.echo(
            f"evaluation for {manifest['current_version']}: saved historical metrics; "
            "current evidence integrity is checked above"
        )
    else:
        typer.echo(f"evaluation for {manifest['current_version']}: N/A (not computed)")


@app.command()
def evaluate(
    run: Path = typer.Option(..., "--run", help="Run directory produced by clean."),
    truth: Path = typer.Option(..., "--truth", help="Ground-truth CSV path."),
    baseline: Path | None = typer.Option(
        None, "--baseline", help="Optional rule-baseline run directory (default: sibling 'baseline')."
    ),
) -> None:
    """Compare original, cleaned and rule-baseline data against the ground truth."""
    store = _load_store(run)
    if not truth.is_file():
        _fail(f"ground truth CSV not found: {truth}")
    try:
        config = load_run_config(store)
    except ConfigError as error:
        _fail(str(error))
    try:
        metrics = evaluate_run(store, config, truth, baseline_dir=baseline)
    except StorageError as error:
        _fail(str(error))
    # Refresh the run report so its evaluation summary reflects the version just scored.
    write_run_reports(store, config)
    typer.echo(f"version: {metrics['version']} (model identity: {metrics['model_identity']})")
    typer.echo(f"quality original: {metrics['quality']['original']}")
    typer.echo(f"quality cleaned:  {metrics['quality']['cleaned']}")
    corruption = metrics["corruption"]
    typer.echo(
        f"corruption repair coverage: {_rate(corruption['repair_coverage'])} "
        f"({corruption['repaired_entries']}/{corruption['repairable_entries']}); "
        f"wrong modifications: {corruption['wrong_modifications']}"
    )
    numeric = corruption["numeric_imputation"]
    typer.echo(
        f"numeric imputation: filled {numeric['filled']}/{numeric['masked_total']}; MAE "
        f"{'N/A' if numeric['mae'] is None else round(numeric['mae'], 6)} "
        f"over {numeric['mae_denominator']} filled values"
    )
    categorical = corruption["categorical_imputation"]
    typer.echo(
        f"categorical imputation: filled {categorical['filled']}/{categorical['masked_total']}; "
        f"accuracy {_rate(categorical['accuracy'])}"
    )
    duplicates = metrics["duplicates"]
    typer.echo(
        f"duplicates: removed {duplicates['removed']}/{duplicates['expected_removed']}; "
        f"wrong {duplicates['wrong_deletions']}; missed {duplicates['missed_deletions']}"
    )
    api = metrics["api"]
    typer.echo(
        f"api: {api['requests']} requests, {api['retries']} retries, "
        f"{api['known_input_tokens']} known input tokens, "
        f"{api['unknown_usage_requests']} unknown-usage requests"
    )
    integrity = metrics["integrity"]
    typer.echo(
        f"integrity: distribution {_rate(integrity['distribution_save_rate'])}; traceability "
        f"{_rate(integrity['traceability_rate'])}; replay {_rate(integrity['gate_replay_rate'])}"
    )
    if metrics["baseline"]:
        baseline = metrics["baseline"]
        if baseline["corruption"] is None:
            typer.echo(
                f"baseline quality: {baseline['cleaned']}; cell metrics: N/A "
                f"({baseline['row_mapping']}); {baseline['model_distribution_metrics']}"
            )
            return
        typer.echo(
            f"baseline quality: {baseline['cleaned']}; corruption repair coverage "
            f"{_rate(baseline['corruption']['repair_coverage'])}; wrong modifications "
            f"{baseline['corruption']['wrong_modifications']}; "
            f"numeric MAE {'N/A' if baseline['corruption']['numeric_imputation']['mae'] is None else round(baseline['corruption']['numeric_imputation']['mae'], 6)}; "
            f"{baseline['model_distribution_metrics']}"
        )
    else:
        typer.echo("baseline: N/A (run examples/run_baseline.py to compare)")


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


@app.command()
def rollback(
    run: Path = typer.Option(..., "--run", help="Run directory produced by clean."),
    to_version: str = typer.Option(..., "--to-version", help="Committed version to restore."),
) -> None:
    """Restore a committed snapshot and its exports; never calls the model."""
    store = _load_store(run)
    try:
        state = store.rollback(to_version)
    except StorageError as error:
        _fail(str(error))
    write_run_reports(store, load_run_config(store))
    typer.echo(
        f"rolled back to {to_version}: {len(state.rows)} rows, "
        f"{len(state.removed_rows)} removed rows"
    )
    typer.echo(f"report: {run / RUN_FILES['report']}")
