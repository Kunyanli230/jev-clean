"""v0.2 offline workflows and stable baseline identities after interior deletions."""

from __future__ import annotations

import json

import pytest
from fake_client import DEMO_CONFIG, DEMO_DATA, FakeDecisionClient
from typer.testing import CliRunner

from idac.baseline import run_baseline
from idac.cli import app
from idac.config import load_config
from idac.evaluation import evaluate_run
from idac.models import DatasetState, sha256_bytes
from idac.orchestrator import run_clean
from idac.profile import profile_csv
from idac.storage import RunStore, StorageError

runner = CliRunner()


@pytest.fixture
def offline_only(monkeypatch):
    from typesafe_sdk import TypeSafeClient

    def forbidden(*args, **kwargs):
        raise AssertionError("offline command attempted an SDK request")

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(TypeSafeClient, "system_one", forbidden)


def test_profile_reports_original_without_mutating_or_calling_model(tmp_path, offline_only):
    before = (DEMO_DATA / "dirty.csv").read_bytes()
    profile = profile_csv(DEMO_DATA / "dirty.csv", DEMO_CONFIG)
    assert profile["rows"] == 210
    assert profile["model_calls"] == 0
    assert profile["model_identity"] == "none"
    assert profile["candidate_count"] > 0
    assert profile["declared_null_token_cells"] > 0
    assert "ambiguous_parse" in profile["issue_counts"]
    assert any(
        issue["category"] == "dependency_blocked"
        for phase in profile["phases"] for issue in phase["issues"]
    )
    assert all(
        len(candidate["preview"]) <= 8
        for phase in profile["phases"] for candidate in phase["candidates"]
    )
    assert (DEMO_DATA / "dirty.csv").read_bytes() == before
    assert list(tmp_path.iterdir()) == []


def test_profile_cli_json_and_output_file_agree(tmp_path, offline_only):
    output = tmp_path / "profile.json"
    result = runner.invoke(app, [
        "profile", "--input", str(DEMO_DATA / "dirty.csv"), "--config", str(DEMO_CONFIG),
        "--json", "--output", str(output),
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == json.loads(output.read_text())
    assert payload["kind"] == "input_profile"


def test_profile_refuses_overwriting_existing_file(tmp_path, offline_only):
    output = tmp_path / "profile.json"
    output.write_text("keep me")
    result = runner.invoke(app, [
        "profile", "--input", str(DEMO_DATA / "dirty.csv"), "--config", str(DEMO_CONFIG),
        "--output", str(output),
    ])
    assert result.exit_code == 1
    assert output.read_text() == "keep me"


def test_profile_baseline_keep_protected_and_disallowed_cells(tmp_path, offline_only):
    config = tmp_path / "config.yaml"
    config.write_text('''table_description: test
record_grain: row
columns:
  protected:
    type: string
    description: protected value
    protected: true
    null_tokens: [""]
    allowed_operations: [trim_whitespace]
  blocked:
    type: string
    description: no allowed operations
    null_tokens: [""]
    allowed_operations: []
  name:
    type: string
    description: trimmable value
    null_tokens: [""]
    allowed_operations: [trim_whitespace]
duplicates:
  enabled: false
  compare_columns: []
''')
    source = tmp_path / "input.csv"
    source.write_text("protected,blocked,name\n P , B , C \n")
    profile = profile_csv(source, config)
    assert {
        candidate["column"]
        for phase in profile["phases"] for candidate in phase["candidates"]
    } == {"name"}
    output = tmp_path / "baseline"
    run_baseline(source, config, output)
    snapshot = DatasetState.model_validate_json((output / "snapshot.json").read_bytes())
    assert snapshot.rows == [[" P ", " B ", "C"]]
    assert source.read_text() == "protected,blocked,name\n P , B , C \n"


def test_baseline_cli_and_compatibility_entry_point(tmp_path, offline_only):
    output = tmp_path / "baseline"
    result = runner.invoke(app, [
        "baseline", "--input", str(DEMO_DATA / "dirty.csv"), "--config", str(DEMO_CONFIG),
        "--output", str(output),
    ])
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "baseline.json").read_text())
    assert summary["rows"] == 200
    assert summary["removed_rows"] == 10
    assert summary["hard_failures"] == []
    assert summary["snapshot_hash"] == sha256_bytes((output / "snapshot.json").read_bytes())
    assert not (output / "decisions.jsonl").exists()
    assert not (output / "requests").exists()
    from examples.run_baseline import BaselineRunner as OldRunner
    from idac.baseline import BaselineRunner

    assert OldRunner is BaselineRunner
    second = runner.invoke(app, [
        "baseline", "--input", str(DEMO_DATA / "dirty.csv"), "--config", str(DEMO_CONFIG),
        "--output", str(output),
    ])
    assert second.exit_code == 1


def test_baseline_cli_reports_hard_failures(tmp_path, offline_only, monkeypatch):
    import idac.baseline

    original = idac.baseline.validate_hard

    def rejected(*args):
        result = original(*args)
        return result.model_copy(update={"passed": False})

    monkeypatch.setattr(idac.baseline, "validate_hard", rejected)
    output = tmp_path / "baseline"
    result = runner.invoke(app, [
        "baseline", "--input", str(DEMO_DATA / "dirty.csv"), "--config", str(DEMO_CONFIG),
        "--output", str(output),
    ])
    assert result.exit_code == 1
    snapshot = DatasetState.model_validate_json((output / "snapshot.json").read_bytes())
    assert snapshot.version_id == "v000"
    assert json.loads((output / "baseline.json").read_text())["versions"] == 0
    assert snapshot.rows[0][1] == "  Ulf Dorn "
    assert "hard validation failures" in result.output
    import sys

    from examples.run_baseline import main

    monkeypatch.setattr(sys, "argv", [
        "run_baseline.py", "--input", str(DEMO_DATA / "dirty.csv"),
        "--config", str(DEMO_CONFIG), "--output", str(tmp_path / "compatibility"),
    ])
    with pytest.raises(SystemExit, match="hard validation failures"):
        main()


@pytest.mark.parametrize("command", ["profile", "baseline"])
def test_offline_cli_rejects_empty_csv(tmp_path, offline_only, command):
    source = tmp_path / "empty.csv"
    source.write_text("")
    args = [command, "--input", str(source), "--config", str(DEMO_CONFIG)]
    if command == "baseline":
        args.extend(["--output", str(tmp_path / "baseline")])
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "could not parse input CSV" in result.output


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "0.2.0" in result.output


@pytest.fixture
def interior_duplicate(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('''table_description: interior duplicate regression
record_grain: row
columns:
  name:
    type: string
    description: label
    null_tokens: [""]
    allowed_operations: [trim_whitespace]
  age:
    type: integer
    description: age
    null_tokens: [""]
    allowed_operations: [normalize_null_tokens, cast_numeric]
duplicates:
  enabled: true
  compare_columns: [name, age]
''')
    source = tmp_path / "input.csv"
    source.write_text("name,age\nA,10\nA,10\n C ,30\nD,40\nE,\n")
    truth = tmp_path / "truth.csv"
    truth.write_text("name,age\nA,10\nA,10\nC,30\nD,40\nE,\n")
    (tmp_path / "corruption.json").write_text(json.dumps({
        "duplicate_sources": {"row-000001": "row-000000"},
        "entries": [
            {
                "row_id": "row-000002", "column": "name", "corruption_type": "whitespace",
                "dirty_value": " C ",
            },
            {
                "row_id": "row-000004", "column": "age", "corruption_type": "null_token",
                "dirty_value": "", "recoverable_by_rule": False,
            },
        ],
    }))
    output = tmp_path / "clean"
    run_clean(source, config_path, output, client=FakeDecisionClient())
    baseline = tmp_path / "baseline"
    run_baseline(source, config_path, baseline)
    return RunStore(output), load_config(config_path), truth, baseline


def test_baseline_evaluation_preserves_row_ids_and_null_values(interior_duplicate):
    store, config, truth, baseline = interior_duplicate
    metrics = evaluate_run(store, config, truth, baseline)
    result = metrics["baseline"]
    assert result["row_mapping"] == "verified_snapshot"
    assert result["corruption"]["wrong_modifications"] == 0
    assert result["corruption"]["repair_coverage"] == 1
    assert result["cleaned"]["missing_cells"] == 1
    assert result["duplicates"]["wrong_deletions"] == 0
    snapshot = DatasetState.model_validate_json((baseline / "snapshot.json").read_bytes())
    assert snapshot.row_ids == ["row-000000", "row-000002", "row-000003", "row-000004"]
    assert snapshot.rows[-1][-1] is None


@pytest.mark.parametrize("target", ["snapshot.json", "cleaned.csv", "removed_rows.csv"])
def test_baseline_evaluation_rejects_tampered_artifacts(interior_duplicate, target):
    store, config, truth, baseline = interior_duplicate
    path = baseline / target
    text = path.read_text()
    if target == "snapshot.json":
        path.write_text(text.replace('"row-000002"', '"row-999999"'))
    elif target == "cleaned.csv":
        path.write_text(text.replace("D,40", "X,40"))
    else:
        path.write_text(text.replace("row-000001", "row-999999"))
    with pytest.raises(StorageError, match="baseline"):
        evaluate_run(store, config, truth, baseline)


def test_baseline_evaluation_rejects_invalid_ids_even_with_updated_hash(interior_duplicate):
    store, config, truth, baseline = interior_duplicate
    path = baseline / "snapshot.json"
    payload = json.loads(path.read_text())
    payload["row_ids"][1] = "row-999999"
    path.write_text(json.dumps(payload))
    summary = json.loads((baseline / "baseline.json").read_text())
    summary["snapshot_hash"] = sha256_bytes(path.read_bytes())
    (baseline / "baseline.json").write_text(json.dumps(summary))
    with pytest.raises(StorageError, match="row identities"):
        evaluate_run(store, config, truth, baseline)


def test_legacy_baseline_recovers_ids_from_complete_removal_ledger(interior_duplicate):
    store, config, truth, baseline = interior_duplicate
    (baseline / "snapshot.json").unlink()
    (baseline / "baseline.json").unlink()
    metrics = evaluate_run(store, config, truth, baseline)
    assert metrics["baseline"]["row_mapping"].startswith("legacy_removal_ledger")
    assert metrics["baseline"]["corruption"]["wrong_modifications"] == 0
    assert metrics["baseline"]["corruption"]["repair_coverage"] == 1


def test_legacy_baseline_without_row_mapping_suppresses_cell_metrics(interior_duplicate):
    store, config, truth, baseline = interior_duplicate
    for name in ["snapshot.json", "baseline.json", "removed_rows.csv"]:
        (baseline / name).unlink()
    metrics = evaluate_run(store, config, truth, baseline)
    assert metrics["baseline"]["corruption"] is None
    assert metrics["baseline"]["duplicates"] is None
    assert metrics["baseline"]["cleaned"]["exact_duplicate_rows"] == 0
    result = runner.invoke(app, [
        "evaluate", "--run", str(store.run_dir), "--truth", str(truth),
        "--baseline", str(baseline),
    ])
    assert result.exit_code == 0, result.output
    assert "cell metrics: N/A" in result.output


def test_baseline_evaluation_rejects_different_source(interior_duplicate):
    store, config, truth, baseline = interior_duplicate
    summary_path = baseline / "baseline.json"
    summary = json.loads(summary_path.read_text())
    summary["input_hash"] = "different-source"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(StorageError, match="input does not match"):
        evaluate_run(store, config, truth, baseline)


def test_v02_baseline_does_not_fall_back_when_snapshot_is_missing(interior_duplicate):
    store, config, truth, baseline = interior_duplicate
    (baseline / "snapshot.json").unlink()
    with pytest.raises(StorageError, match="snapshot is missing"):
        evaluate_run(store, config, truth, baseline)


def test_v02_baseline_requires_a_removal_ledger(interior_duplicate):
    store, config, truth, baseline = interior_duplicate
    (baseline / "removed_rows.csv").unlink()
    with pytest.raises(StorageError, match="removal ledger is missing"):
        evaluate_run(store, config, truth, baseline)


@pytest.mark.parametrize("explicit", [True, False])
def test_evaluate_rejects_missing_baseline_export(interior_duplicate, explicit):
    store, config, truth, baseline = interior_duplicate
    (baseline / "cleaned.csv").unlink()
    with pytest.raises(StorageError, match="baseline cleaned CSV not found"):
        evaluate_run(store, config, truth, baseline if explicit else None)


@pytest.mark.parametrize("ledger", ["", "garbage\n"])
def test_empty_removal_ledger_requires_a_valid_header(tmp_path, ledger):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('''table_description: no duplicates
record_grain: row
columns:
  name:
    type: string
    description: label
    null_tokens: [""]
    allowed_operations: []
duplicates:
  enabled: false
  compare_columns: []
''')
    source = tmp_path / "source.csv"
    source.write_text("name\nA\n")
    run_path = tmp_path / "clean"
    run_clean(source, config_path, run_path, client=FakeDecisionClient())
    baseline = tmp_path / "baseline"
    run_baseline(source, config_path, baseline)
    (baseline / "removed_rows.csv").write_text(ledger)
    with pytest.raises(StorageError, match="invalid header"):
        evaluate_run(RunStore(run_path), load_config(config_path), source, baseline)


def test_baseline_ledger_handles_quoted_and_metadata_column_names(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('''table_description: unusual column names
record_grain: row
columns:
  row_id:
    type: string
    description: user identifier
    null_tokens: [""]
    allowed_operations: []
  "label,kind":
    type: string
    description: user label
    null_tokens: [""]
    allowed_operations: []
duplicates:
  enabled: true
  compare_columns: [row_id, "label,kind"]
''')
    source = tmp_path / "source.csv"
    source.write_text('row_id,"label,kind"\nA,"X,Y"\nA,"X,Y"\nB,Z\n')
    (tmp_path / "corruption.json").write_text(json.dumps({
        "duplicate_sources": {"row-000001": "row-000000"}, "entries": [],
    }))
    run_path = tmp_path / "clean"
    run_clean(source, config_path, run_path, client=FakeDecisionClient())
    baseline = tmp_path / "baseline"
    run_baseline(source, config_path, baseline)
    metrics = evaluate_run(RunStore(run_path), load_config(config_path), source, baseline)
    assert metrics["baseline"]["row_mapping"] == "verified_snapshot"
    assert metrics["baseline"]["corruption"]["wrong_modifications"] == 0
    assert metrics["baseline"]["duplicates"]["removed"] == 1
