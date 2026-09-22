"""Acceptance 7, 8: hard validation, conflicts and postcheck rejection."""

from __future__ import annotations

import pytest
from fake_client import (
    FakeDecisionClient,
    failing_outcome,
    run_fake_pipeline,
    write_config,
    write_csv_file,
)

from idac.config import load_config
from idac.executor import ExecutionError, execute
from idac.models import (
    CellChange,
    CheckStatus,
    ModelStatus,
    Operation,
    Phase,
    RepairCandidate,
    Selection,
)
from idac.planner import AcceptedCandidate, build_plan
from idac.profiling import detect_phase
from idac.storage import load_csv
from idac.typesafe_client import RunBudget
from idac.validator import confirmed_repaired, validate_hard

CONFIG_TEXT = """
table_description: "validation test table"
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
    null_tokens: [""]
    allowed_operations: [trim_whitespace, normalize_null_tokens, cast_numeric, invalidate_out_of_range, impute_missing]
duplicates:
  enabled: false
  compare_columns: []
"""


def prepare(tmp_path):
    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", " 21 "], ["002", "22"]])
    config = load_config(config_path)
    state = load_csv(input_path, config)
    detection = detect_phase(state, config, Phase.TRIM)
    state = state.model_copy(update={"selections": dict(detection.selections)})
    candidates = build_candidates(state, config, detection)
    evidence = {item.id: item for item in detection.evidence}
    return config, state, candidates, evidence


def build_candidates(state, config, detection):
    from idac.candidates import build_candidates as builder

    return builder(state, config, detection)


def test_plan_and_execution_produce_matching_diff(tmp_path) -> None:
    config, state, candidates, evidence = prepare(tmp_path)
    accepted = [AcceptedCandidate(candidate=candidates[0], decision_id="dec")]
    plan = build_plan(state, config, evidence, accepted, "plan-0001")
    assert plan.ordered_operations
    execution = execute(state, plan, config, "v001")
    result = validate_hard(state, execution, plan, config)
    assert result.passed
    assert all(check.status == CheckStatus.PASSED for check in result.hard_checks)


def test_extra_cell_modification_is_rejected(tmp_path) -> None:
    config, state, candidates, evidence = prepare(tmp_path)
    accepted = [AcceptedCandidate(candidate=candidates[0], decision_id="dec")]
    plan = build_plan(state, config, evidence, accepted, "plan-0001")
    execution = execute(state, plan, config, "v001")
    execution.state.rows[0][0] = "HACKED"
    execution.changes.append(
        CellChange(row_id="row-000000", column="id", before="001", after="HACKED")
    )
    result = validate_hard(state, execution, plan, config)
    assert not result.passed
    failed = {check.rule_id for check in result.hard_checks if check.status == CheckStatus.FAILED}
    assert {"PLAN_DIFF_MATCH", "PROTECTED_CELLS_UNCHANGED"} <= failed


def test_stale_candidate_and_selection_are_discarded(tmp_path) -> None:
    config, state, candidates, evidence = prepare(tmp_path)
    stale = candidates[0].model_copy(update={"version_id": "v999"})
    accepted = [AcceptedCandidate(candidate=stale, decision_id="dec")]
    plan = build_plan(state, config, evidence, accepted, "plan-0001")
    assert not plan.ordered_operations
    assert plan.discarded and plan.discarded[0]["reason_code"] == "STALE_VERSION"

    missing_selection = candidates[0].model_copy(update={"selection_id": "sel-missing"})
    plan = build_plan(
        state, config, evidence, [AcceptedCandidate(candidate=missing_selection, decision_id="dec")], "plan-0002"
    )
    assert not plan.ordered_operations


def test_conflicting_candidates_are_all_discarded(tmp_path) -> None:
    config, state, candidates, evidence = prepare(tmp_path)
    base = candidates[0]
    first = base.model_copy(
        update={
            "id": "cand-first",
            "parameters": {**base.parameters, "after_values": {"row-000000": "21"}},
            "fingerprint": "fp-1",
        }
    )
    second = base.model_copy(
        update={
            "id": "cand-second",
            "parameters": {**base.parameters, "after_values": {"row-000000": "22"}},
            "fingerprint": "fp-2",
        }
    )
    plan = build_plan(
        state,
        config,
        evidence,
        [
            AcceptedCandidate(candidate=first, decision_id="dec-1"),
            AcceptedCandidate(candidate=second, decision_id="dec-2"),
        ],
        "plan-0003",
    )
    assert not plan.ordered_operations
    assert any(item["reason_code"] == "CANDIDATE_CONFLICT" for item in plan.discarded)


def test_identical_candidates_are_merged(tmp_path) -> None:
    config, state, candidates, evidence = prepare(tmp_path)
    base = candidates[0]
    first = base.model_copy(update={"id": "cand-a", "fingerprint": "fp-a"})
    second = base.model_copy(update={"id": "cand-b", "fingerprint": "fp-b"})
    plan = build_plan(
        state,
        config,
        evidence,
        [
            AcceptedCandidate(candidate=first, decision_id="dec-1"),
            AcceptedCandidate(candidate=second, decision_id="dec-2"),
        ],
        "plan-0004",
    )
    assert len(plan.ordered_operations) == 1
    assert set(plan.ordered_operations[0].candidate_ids) == {"cand-a", "cand-b"}


def test_protected_column_candidate_is_discarded(tmp_path) -> None:
    config, state, candidates, evidence = prepare(tmp_path)
    protected = candidates[0].model_copy(
        update={
            "id": "cand-protected",
            "parameters": {"column": "id", "after_values": {"row-000000": "001"}},
            "fingerprint": "fp-protected",
        }
    )
    plan = build_plan(
        state, config, evidence, [AcceptedCandidate(candidate=protected, decision_id="dec")], "plan-0005"
    )
    assert not plan.ordered_operations
    assert plan.discarded[0]["reason_code"] == "PROTECTED_COLUMN"


def test_executor_refuses_protected_column(tmp_path) -> None:
    config, state, _candidates, _evidence = prepare(tmp_path)
    selection = Selection(
        id="sel-protected",
        version_id=state.version_id,
        row_ids=["row-000000"],
        column="id",
        fingerprint="fp",
    )
    state = state.model_copy(update={"selections": {**state.selections, selection.id: selection}})
    candidate = RepairCandidate(
        id="cand-protected",
        version_id=state.version_id,
        issue_ids=[],
        operation=Operation.TRIM_WHITESPACE,
        selection_id=selection.id,
        parameters={"column": "id", "after_values": {"row-000000": "001"}},
        preview=[],
        evidence_ids=[],
        fingerprint="fp",
    )
    from idac.models import PlannedOperation, RepairPlan

    plan = RepairPlan(
        id="plan-forced",
        version_id=state.version_id,
        candidate_ids=[candidate.id],
        ordered_operations=[
            PlannedOperation(
                operation=Operation.TRIM_WHITESPACE,
                candidate_ids=[candidate.id],
                selection_id=selection.id,
                column="id",
                after_values={"row-000000": "001"},
            )
        ],
    )
    with pytest.raises(ExecutionError):
        execute(state, plan, config, "v001")


def test_hard_validation_failure_sends_no_postcheck(monkeypatch, tmp_path) -> None:
    import idac.orchestrator as orchestrator

    real_execute = orchestrator.execute

    def tampered_execute(state, plan, config, new_version_id):
        result = real_execute(state, plan, config, new_version_id)
        result.state.rows[0][0] = "HACKED"
        result.changes.append(
            CellChange(row_id="row-000000", column="id", before="001", after="HACKED")
        )
        return result

    monkeypatch.setattr(orchestrator, "execute", tampered_execute)
    client = FakeDecisionClient()
    rows = [["001", " 21 "], ["002", "22"]]
    result, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows, client)
    assert result.outcome.value == "partial"
    assert not [request for request in store.read_requests() if request["kind"] == "postcheck"]
    assert not [entry for entry in store.read_changes() if entry["committed"]]
    assert store.current_state().rows[0][0] == "001"
    assert not [record for record in store.read_decisions() if record["committed"]]


def test_postcheck_above_threshold_rejects_batch(tmp_path) -> None:
    client = FakeDecisionClient(postcheck=0.5)
    rows = [["001", " 21 "], ["002", "22"]]
    result, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows, client)
    assert result.outcome.value == "partial"
    assert store.current_state().rows[0][1] == " 21 "
    assert not [entry for entry in store.read_changes() if entry["committed"]]
    traces = store.read_traces()
    assert traces and all(trace["commit_status"] == "not_committed" for trace in traces)
    assert any(trace["postcheck_status"] == "failed" for trace in traces)
    unresolved = [issue for issue in store.current_state().issues if issue.status.value == "unresolved"]
    assert any("SEMANTIC_VALIDATION_FAILED" in [code.value for code in issue.reason_codes] for issue in unresolved)


TWO_COLUMN_CONFIG = """
table_description: "two-column validation table"
record_grain: "one row per record"
columns:
  id:
    type: string
    description: "identifier"
    protected: true
    null_tokens: [""]
    allowed_operations: []
  name:
    type: string
    description: "name"
    null_tokens: [""]
    allowed_operations: [trim_whitespace, normalize_null_tokens]
  city:
    type: string
    description: "city"
    null_tokens: [""]
    allowed_operations: [trim_whitespace, normalize_null_tokens]
duplicates:
  enabled: false
  compare_columns: []
"""


def test_batch_rejection_records_triggering_candidate(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    from idac.orchestrator import run_clean
    from idac.storage import RunStore

    config_path = write_config(tmp_path, TWO_COLUMN_CONFIG)
    input_path = write_csv_file(
        tmp_path, ["id", "name", "city"], [["001", " Alice ", " Berlin "], ["002", "Bob", "Hamburg"]]
    )
    client = FakeDecisionClient(postcheck=0.05)
    # Identify candidates from a first pass, then make one of them fail the postcheck.
    probe = FakeDecisionClient()
    run_clean(input_path, config_path, tmp_path / "probe", client=probe)
    contexts = [
        context
        for call in probe.calls
        if call["kind"] == "decision"
        for context in call["state"]["candidates"]
    ]
    assert len(contexts) == 2
    failing = contexts[1]["candidate_id"]
    client = FakeDecisionClient(overrides={failing: {"postcheck": 0.5}})
    result = run_clean(input_path, config_path, tmp_path / "run", client=client)
    assert result.outcome.value == "partial"
    store = RunStore(tmp_path / "run")
    traces = {trace["candidate_id"]: trace for trace in store.read_traces()}
    assert traces[failing]["postcheck_status"] == "failed"
    assert traces[failing]["blocking_candidate_ids"] == [failing]
    other = contexts[0]["candidate_id"]
    assert "BATCH_REJECTED" in traces[other]["reason_codes"]
    assert traces[other]["blocking_candidate_ids"] == [failing]


def test_partially_committed_batch_finalizes_rejected_candidate(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    from idac.orchestrator import run_clean
    from idac.storage import RunStore

    config_path = write_config(tmp_path, TWO_COLUMN_CONFIG)
    input_path = write_csv_file(
        tmp_path, ["id", "name", "city"], [["001", " Alice ", " Berlin "], ["002", "Bob", "Hamburg"]]
    )
    probe = FakeDecisionClient()
    run_clean(input_path, config_path, tmp_path / "probe", client=probe)
    contexts = [
        context
        for call in probe.calls
        if call["kind"] == "decision"
        for context in call["state"]["candidates"]
    ]
    rejected_id = next(
        context["candidate_id"] for context in contexts if context["column"] == "city"
    )
    client = FakeDecisionClient(overrides={rejected_id: {"applicable": 0.1}})
    result = run_clean(input_path, config_path, tmp_path / "run", client=client)
    assert result.outcome.value == "partial"
    store = RunStore(tmp_path / "run")
    asked = [
        context["candidate_id"]
        for call in client.calls
        if call["kind"] == "decision"
        for context in call["state"]["candidates"]
        if context["column"] == "city"
    ]
    assert len(asked) == 1, "a rejected candidate must not be re-sampled after a partial commit"
    decisions = {record["candidate_id"]: record for record in store.read_decisions()}
    assert decisions[rejected_id]["committed"] is False
    assert "POLICY_NOT_SUPPORTED" in decisions[rejected_id]["reason_codes"]
    traces = {trace["candidate_id"]: trace for trace in store.read_traces()}
    assert traces[rejected_id]["commit_status"] == "not_committed"
    committed = [record for record in decisions.values() if record["committed"]]
    assert committed and all(record["gate_accepted"] for record in committed)
    unresolved = [issue for issue in store.current_state().issues if issue.status.value == "unresolved"]
    assert any("POLICY_NOT_SUPPORTED" in [code.value for code in issue.reason_codes] for issue in unresolved)


def test_oversized_request_is_unresolved_without_model_call(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    from idac.orchestrator import run_clean
    from idac.storage import RunStore

    huge = "x" * 30_000
    config_text = CONFIG_TEXT.replace(
        'table_description: "validation test table"',
        f'table_description: "{huge}"',
    )
    config_path = write_config(tmp_path, config_text)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", " 21 "], ["002", "22"]])
    client = FakeDecisionClient()
    result = run_clean(input_path, config_path, tmp_path / "run", client=client)
    assert result.outcome.value == "partial"
    assert client.calls == []
    store = RunStore(tmp_path / "run")
    unresolved = [issue for issue in store.current_state().issues if issue.status.value == "unresolved"]
    assert unresolved
    assert any(
        "REQUEST_TOO_LARGE" in [code.value for code in issue.reason_codes] for issue in unresolved
    )


def test_postcheck_api_failure_rejects_batch(tmp_path) -> None:
    class PostcheckFails(FakeDecisionClient):
        def evaluate(self, state, questions, *, kind="decision"):
            if kind == "postcheck":
                return failing_outcome(ModelStatus.FAILED, "network down")
            return super().evaluate(state, questions, kind=kind)

    rows = [["001", " 21 "], ["002", "22"]]
    result, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows, PostcheckFails())
    assert result.outcome.value == "partial"
    assert not [entry for entry in store.read_changes() if entry["committed"]]
    assert not [record for record in store.read_decisions() if record["committed"]]


def test_postcheck_missing_answer_rejects_batch(tmp_path) -> None:
    class MissingPostcheck(FakeDecisionClient):
        def evaluate(self, state, questions, *, kind="decision"):
            outcome = super().evaluate(state, questions, kind=kind)
            if kind == "postcheck":
                outcome.answers = {}
            return outcome

    rows = [["001", " 21 "], ["002", "22"]]
    result, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows, MissingPostcheck())
    assert result.outcome.value == "partial"
    assert not [entry for entry in store.read_changes() if entry["committed"]]


def test_budget_exhaustion_before_postcheck_does_not_commit(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    budget = RunBudget(max_attempts=1)
    client = FakeDecisionClient(budget=budget)
    result, store, _ = run_fake_pipeline(
        tmp_path, CONFIG_TEXT, ["id", "age"], rows, client, budget=budget
    )
    assert result.outcome.value == "partial"
    assert not [entry for entry in store.read_changes() if entry["committed"]]
    assert budget.attempts == 1
    requests = store.read_requests()
    assert len(requests) == 2  # decision attempt plus the not-called postcheck entry
    assert requests[-1]["model_status"] == "not_called"


def test_removing_the_earliest_duplicate_is_rejected(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    from idac.models import PlannedOperation, RepairPlan, Selection
    from idac.storage import load_csv
    from idac.validator import validate_hard as run_validate

    duplicate_config = """
table_description: "duplicate kept-row check"
record_grain: "one row per record"
columns:
  id:
    type: string
    description: "identifier"
    protected: true
    null_tokens: [""]
    allowed_operations: []
  age:
    type: string
    description: "age"
    null_tokens: [""]
    allowed_operations: []
duplicates:
  enabled: true
  compare_columns: [id, age]
"""
    config_path = write_config(tmp_path, duplicate_config)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", "20"], ["001", "20"]])
    config = load_config(config_path)
    state = load_csv(input_path, config)
    selection = Selection(
        id="sel-duplicates",
        version_id=state.version_id,
        row_ids=["row-000000"],
        column=None,
        fingerprint="fp",
    )
    state = state.model_copy(update={"selections": {selection.id: selection}})
    plan = RepairPlan(
        id="plan-wrong-kept",
        version_id=state.version_id,
        candidate_ids=["cand-wrong"],
        ordered_operations=[
            PlannedOperation(
                operation=Operation.REMOVE_EXACT_DUPLICATES,
                candidate_ids=["cand-wrong"],
                selection_id=selection.id,
                removed_row_ids=["row-000000"],  # removes the earliest row instead of the copy
            )
        ],
    )
    execution = execute(state, plan, config, "v001")
    result = run_validate(state, execution, plan, config)
    assert not result.passed
    failed = {check.rule_id for check in result.hard_checks if check.status == CheckStatus.FAILED}
    assert "DUPLICATE_KEPT_ROWS" in failed


def test_confirmed_repaired_requires_recheck(tmp_path) -> None:
    config, state, candidates, evidence = prepare(tmp_path)
    detection = detect_phase(state, config, Phase.TRIM)
    issue = detection.issues[0]
    repaired = confirmed_repaired(state, config, Phase.TRIM, [issue], detection.selections)
    assert repaired == set()  # whitespace is still present in the unchanged state
    accepted = [AcceptedCandidate(candidate=candidates[0], decision_id="dec")]
    plan = build_plan(state, config, evidence, accepted, "plan-0001")
    execution = execute(state, plan, config, "v001")
    repaired = confirmed_repaired(execution.state, config, Phase.TRIM, [issue], detection.selections)
    assert repaired == {issue.id}
