"""Acceptance 12: snapshots, exports, rollback and version-bound history."""

from __future__ import annotations

import json

import pytest
from fake_client import run_fake_pipeline, write_config, write_csv_file

from idac.config import load_config
from idac.evaluation import evaluate_run
from idac.models import DatasetState, Issue, IssueStatus, ReasonCode
from idac.storage import RunStore, StorageError, load_csv

CONFIG_TEXT = """
table_description: "storage test table"
record_grain: "one row per record"
columns:
  id:
    type: string
    description: "identifier"
    protected: true
    null_tokens: [""]
    allowed_operations: []
  age:
    type: integer
    description: "age"
    min_value: 18
    max_value: 100
    null_tokens: ["", "NA"]
    allowed_operations: [trim_whitespace, normalize_null_tokens, cast_numeric, invalidate_out_of_range, impute_missing]
duplicates:
  enabled: true
  compare_columns: [id, age]
"""


def test_snapshot_distinguishes_empty_string_and_null(tmp_path) -> None:
    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", ""], ["002", "22"]])
    config = load_config(config_path)
    state = load_csv(input_path, config)
    store = RunStore.create(tmp_path / "run", config_path, input_path, config, "fake")
    state.rows[1][1] = None
    store.save_snapshot(state)
    loaded = store.load_snapshot("v000")
    assert loaded.rows[0][1] == ""
    assert loaded.rows[1][1] is None


def test_run_store_refuses_existing_directory(tmp_path) -> None:
    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", "22"]])
    config = load_config(config_path)
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(StorageError):
        RunStore.create(output, config_path, input_path, config, "fake")


def test_empty_removed_rows_export_has_header(tmp_path) -> None:
    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", "22"], ["002", "23"]])
    config = load_config(config_path)
    store = RunStore.create(tmp_path / "run", config_path, input_path, config, "fake")
    state = load_csv(input_path, config)
    store.save_snapshot(state)
    store.record_version(state, plan_id=None, candidate_ids=[])
    store.export_current()
    removed = (tmp_path / "run" / "removed_rows.csv").read_text(encoding="utf-8")
    assert removed.splitlines()[0] == "row_id,reason,candidate_id,id,age"
    assert len(removed.splitlines()) == 1


def test_rollback_restores_snapshot_and_keeps_history(tmp_path) -> None:
    rows = [
        ["001", " 21 "],
        ["001", " 21 "],  # duplicate of row 0: removed in the duplicates phase
        ["002", "22"],
    ]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    manifest = store.manifest()
    versions = list(manifest["versions"])
    assert versions[0] == "v000"
    assert len(versions) >= 3  # trim, duplicate removal, ...
    final_version = manifest["current_version"]
    decisions_before = store.read_decisions()
    traces_before = store.read_traces()
    changes_before = store.read_changes()
    assert changes_before

    state = store.rollback("v000")
    assert state.version_id == "v000"
    assert store.manifest()["current_version"] == "v000"
    assert state.rows[0][1] == " 21 "  # original logical value restored
    assert len(state.rows) == 3
    assert not state.removed_rows
    assert store.read_decisions() == decisions_before
    assert store.read_traces() == traces_before
    assert store.read_changes() == changes_before
    audit = store.read_audit()
    assert any(event["event_type"] == "rollback" for event in audit)

    # The report must not show metrics bound to the rolled-back version.
    store.rollback("v000")
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "No evaluation is bound" in report
    assert final_version != "v000"


def test_rollback_rejects_unknown_version(tmp_path) -> None:
    rows = [["001", "22"], ["002", "23"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    with pytest.raises(StorageError):
        store.rollback("v999")


def test_evaluation_is_bound_to_version_and_hidden_after_rollback(tmp_path) -> None:
    truth = tmp_path / "truth.csv"
    truth.write_text("id,age\n001,21\n002,22\n", encoding="utf-8")
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    config = load_config(tmp_path / "config.yaml")
    metrics = evaluate_run(store, config, truth)
    version = store.manifest()["current_version"]
    assert store.evaluation_path(version).is_file()
    assert metrics["version"] == version
    store.rollback("v000")
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "No evaluation is bound" in report


def test_ancestor_chain_follows_parent_links(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    chain = store.ancestor_chain()
    assert chain[0] == "v000"
    assert chain[-1] == store.manifest()["current_version"]
    manifest = store.manifest()
    for version in chain[1:]:
        assert manifest["versions"][version]["parent"] in chain


def test_issue_statuses_survive_snapshot_roundtrip(tmp_path) -> None:
    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", " 21 "], ["002", "22"]])
    config = load_config(config_path)
    state = load_csv(input_path, config)
    issue = Issue(
        id="iss-1",
        category="whitespace",
        column="age",
        selection_id=None,
        status=IssueStatus.UNRESOLVED,
        reason_codes=[ReasonCode.PARSE_AMBIGUOUS],
    )
    state.issues = [issue]
    store = RunStore.create(tmp_path / "run", config_path, input_path, config, "fake")
    store.save_snapshot(state)
    loaded: DatasetState = store.load_snapshot("v000")
    assert loaded.issues[0].status == IssueStatus.UNRESOLVED
    assert loaded.issues[0].reason_codes == [ReasonCode.PARSE_AMBIGUOUS]


def test_duplicate_input_headers_get_unique_internal_names(tmp_path) -> None:
    config_text = """
table_description: "duplicate headers"
record_grain: "one row per record"
columns:
  a:
    type: string
    description: "first a"
    protected: true
    null_tokens: [""]
    allowed_operations: []
  a.1:
    type: string
    description: "second a"
    protected: true
    null_tokens: [""]
    allowed_operations: []
duplicates:
  enabled: false
  compare_columns: []
"""
    config_path = write_config(tmp_path, config_text)
    input_path = tmp_path / "input.csv"
    input_path.write_text("a,a\nleft,right\n", encoding="utf-8")
    config = load_config(config_path)
    state = load_csv(input_path, config)
    assert state.ordered_columns == ["a", "a.1"]
    assert state.rows == [["left", "right"]]
    assert len(set(state.ordered_columns)) == len(state.ordered_columns)


def test_rollback_restores_issue_statuses(tmp_path) -> None:
    rows = [["001", " 21 "], ["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    chain = store.ancestor_chain()
    target = chain[1] if len(chain) > 1 else chain[0]
    snapshot_issues = store.load_snapshot(target).issues
    store.rollback(target)
    restored = store.current_state()
    assert restored.version_id == target
    assert [(issue.id, issue.status) for issue in restored.issues] == [
        (issue.id, issue.status) for issue in snapshot_issues
    ]
    assert len(restored.removed_rows) == len(store.load_snapshot(target).removed_rows)


def test_manifest_records_request_files(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    requests = sorted((tmp_path / "run" / "requests").glob("req-*.json"))
    assert requests
    payload = json.loads(requests[0].read_text(encoding="utf-8"))
    assert payload["state"]["candidates"]
    assert payload["questions"]
    assert payload["budget"]
