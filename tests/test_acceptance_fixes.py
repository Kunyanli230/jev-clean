"""Regressions for the four acceptance-review findings; no live API calls."""

import json
from types import SimpleNamespace

import pytest
from fake_client import FakeDecisionClient, run_fake_pipeline
from test_validation import CONFIG_TEXT, prepare

from idac.config import TableConfig
from idac.distributions import InvalidDistributionError, summarize_risk
from idac.evaluation import _corruption_metrics, integrity_metrics
from idac.executor import execute
from idac.models import DatasetState, Operation, PlannedOperation, RemovedRow
from idac.planner import AcceptedCandidate, build_plan
from idac.policy import replay_decision
from idac.settings import RISK_LEVEL_CRITERIA
from idac.validator import validate_hard


@pytest.mark.parametrize("row,value", [(1, "99"), (0, " 21 ")])
def test_snapshot_diff_detects_unlogged_or_missing_change(tmp_path, row, value):
    config, state, candidates, evidence = prepare(tmp_path)
    plan = build_plan(state, config, evidence,
                      [AcceptedCandidate(candidate=candidates[0], decision_id="test")], "plan")
    result = execute(state, plan, config, "v001")
    result.state.rows[row][1] = value
    validation = validate_hard(state, result, plan, config)
    assert not validation.passed
    assert any(c.rule_id == "PLAN_DIFF_MATCH" and c.status.value == "failed"
               for c in validation.hard_checks)


def test_snapshot_removed_ids_must_match_plan(tmp_path):
    config, state, candidates, evidence = prepare(tmp_path)
    plan = build_plan(state, config, evidence,
                      [AcceptedCandidate(candidate=candidates[0], decision_id="test")], "plan")
    result = execute(state, plan, config, "v001")
    planned_removed = state.row_ids[1]
    plan.ordered_operations.append(PlannedOperation(
        operation=Operation.REMOVE_EXACT_DUPLICATES, candidate_ids=["test"],
        selection_id=candidates[0].selection_id, removed_row_ids=[planned_removed],
    ))
    result.removed_rows = [RemovedRow(row_id=planned_removed, values=state.rows[1])]
    # Same count, wrong row removed from the actual snapshot.
    result.state.rows = result.state.rows[1:]
    result.state.row_ids = result.state.row_ids[1:]
    validation = validate_hard(state, result, plan, config)
    assert any(c.rule_id == "REMOVED_ROWS_MATCH" and c.status.value == "failed"
               for c in validation.hard_checks)


@pytest.mark.parametrize("kind,value,filled", [
    ("categorical", "", 0), ("categorical", "NA", 0),
    ("categorical", "invalid", 0), ("categorical", "A", 1),
    ("integer", "", 0), ("integer", "NA", 0),
    ("integer", "oops", 0), ("integer", "999", 0), ("integer", "21", 1),
])
def test_imputation_metrics_require_valid_non_missing_value(kind, value, filled):
    column = {"type": kind, "description": "test", "null_tokens": ["", "NA"]}
    column.update({"allowed_values": ["A", "B"]} if kind == "categorical"
                  else {"min_value": 18, "max_value": 100})
    config = TableConfig.model_validate({
        "table_description": "test", "record_grain": "one row", "columns": {"x": column},
        "duplicates": {"enabled": False, "compare_columns": []},
    })
    original = DatasetState(
        version_id="v000", parent_version_id=None, input_hash="test",
        ordered_columns=["x"], row_ids=["row-000000"], rows=[[""]],
        schema=config.logical_schema(),
    )
    cleaned = original.model_copy(update={"rows": [[value]]})
    corruption = {"entries": [{
        "row_id": "row-000000", "column": "x",
        "corruption_type": "categorical_missing" if kind == "categorical" else "numeric_missing",
        "dirty_value": "",
    }]}
    metrics = _corruption_metrics(
        corruption, original, cleaned, [["A" if kind == "categorical" else "21"]],
        {"x": 0}, config,
    )
    imputation = metrics["categorical_imputation" if kind == "categorical" else "numeric_imputation"]
    assert imputation["filled"] == filled
    assert imputation["coverage"] == filled
    assert metrics["repaired_entries"] == filled
    if not filled:
        assert imputation.get("accuracy", imputation.get("mae")) is None


@pytest.mark.parametrize("damage", ["missing", "context", "questions", "invalid_json"])
def test_replay_requires_intact_request_evidence(tmp_path, damage):
    _, store, _ = run_fake_pipeline(
        tmp_path, CONFIG_TEXT, ["id", "age"], [["001", " 21 "], ["002", "22"]]
    )
    record = store.read_decisions()[0]
    assert replay_decision(record, store.run_dir).consistent
    assert not replay_decision(record).replayable
    path = store.run_dir / record["context_ref"]
    payload = json.loads(path.read_text())
    if damage == "missing":
        path.rename(path.with_suffix(".hidden"))
    elif damage == "invalid_json":
        path.write_text("{")
    else:
        if damage == "context":
            payload["state"]["candidates"][0]["target_count"] += 1
        else:
            payload["questions"]["c0_action"]["criteria"]["keep_original"] = "tampered"
        path.write_text(json.dumps(payload))
    result = replay_decision(record, store.run_dir)
    assert not result.replayable and not result.consistent
    metrics = integrity_metrics(store)
    assert metrics["traceability_rate"] == 0
    assert metrics["gate_replay_rate"] == 0
    assert metrics["gate_replay_denominator"] == 1


@pytest.mark.parametrize("legend", [
    {0: "High", 1: "Moderate", 2: "Low"},
    {0: "wrong", 1: RISK_LEVEL_CRITERIA[1], 2: RISK_LEVEL_CRITERIA[2]},
    {0: RISK_LEVEL_CRITERIA[0], 1: RISK_LEVEL_CRITERIA[1]},
])
def test_risk_legend_must_match_request(legend):
    answer = SimpleNamespace(legend=legend, probabilities={0: .95, 1: .04, 2: .01},
                             score=.06, confidence=.95)
    with pytest.raises(InvalidDistributionError):
        summarize_risk(answer)
    answer.legend = dict(enumerate(RISK_LEVEL_CRITERIA))
    assert summarize_risk(answer).expected_risk_level == pytest.approx(.06)


def test_agent_rejects_reversed_risk_legend(tmp_path):
    bad = SimpleNamespace(legend={0: "High", 1: "Moderate", 2: "Low"},
                          probabilities={0: .95, 1: .04, 2: .01},
                          score=.06, confidence=.95)
    client = FakeDecisionClient(answers_override={"c0_risk": bad})
    _, store, _ = run_fake_pipeline(
        tmp_path, CONFIG_TEXT, ["id", "age"], [["001", " 21 "], ["002", "22"]],
        client=client,
    )
    record = store.read_decisions()[0]
    assert record["model_status"] == "invalid_response"
    assert not record["gate_accepted"]
