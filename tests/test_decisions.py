"""Acceptance 5, 6, 16: gate boundaries, batching and field-level gate reads."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from fake_client import FakeDecisionClient, run_fake_pipeline

from idac.agents import StandardizationAgent
from idac.candidates import build_candidates
from idac.config import load_config
from idac.distributions import InvalidDistributionError, summarize_choice, summarize_risk
from idac.models import (
    CheckStatus,
    ChoiceAssessment,
    Evidence,
    ModelStatus,
    Phase,
    ReasonCode,
    RiskAssessment,
)
from idac.policy import evaluate_gate
from idac.profiling import detect_phase
from idac.questions import build_candidate_context, build_questions
from idac.storage import load_csv

CONFIG_TEXT = """
table_description: "decision test table"
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


def choice(selected: str, candidate: str, probability: float, confidence: float) -> ChoiceAssessment:
    return ChoiceAssessment(
        selected_option=selected,
        option_descriptions={candidate: "apply", "keep_original": "keep"},
        probabilities={candidate: probability, "keep_original": 1 - probability},
        selected_probability=probability,
        top_two_margin=abs(probability - (1 - probability)),
        sdk_confidence=confidence,
    )


def risk(distribution: dict[int, float], sdk_score: float | None = None) -> RiskAssessment:
    mean = sum(level * probability for level, probability in distribution.items())
    return RiskAssessment(
        legend={0: "low", 1: "moderate", 2: "high"},
        probabilities=distribution,
        sdk_score=mean if sdk_score is None else sdk_score,
        expected_risk_level=mean,
        score_expectation_delta=(mean if sdk_score is None else sdk_score) - mean,
        variance=sum(probability * (level - mean) ** 2 for level, probability in distribution.items()),
        high_risk_probability=distribution.get(2, 0.0),
        sdk_confidence=0.9,
    )


def test_gate_threshold_boundaries() -> None:
    candidate = "cand-1"
    low = risk({0: 0.60, 1: 0.40, 2: 0.00})
    accepted, checks, reasons = evaluate_gate(
        candidate,
        choice(candidate, candidate, 0.95, 0.80),
        0.80,
        low,
        [],
        ModelStatus.OK,
    )
    assert accepted and not reasons
    assert all(check.status == CheckStatus.PASSED for check in checks)

    accepted, checks, reasons = evaluate_gate(
        candidate,
        choice(candidate, candidate, 0.95, 0.7999),
        0.80,
        low,
        [],
        ModelStatus.OK,
    )
    assert not accepted and ReasonCode.LOW_CHOICE_CONFIDENCE in reasons

    accepted, _, reasons = evaluate_gate(
        candidate, choice(candidate, candidate, 0.95, 0.95), 0.7999, low, [], ModelStatus.OK
    )
    assert not accepted and ReasonCode.POLICY_NOT_SUPPORTED in reasons

    edge = risk({0: 0.50, 1: 0.50, 2: 0.00})
    accepted, _, _ = evaluate_gate(
        candidate, choice(candidate, candidate, 0.95, 0.95), 0.95, edge, [], ModelStatus.OK
    )
    assert accepted  # expected level exactly 0.50 passes

    above = risk({0: 0.499, 1: 0.501, 2: 0.00})
    accepted, _, reasons = evaluate_gate(
        candidate, choice(candidate, candidate, 0.95, 0.95), 0.95, above, [], ModelStatus.OK
    )
    assert not accepted and ReasonCode.EXPECTED_RISK_TOO_HIGH in reasons

    tail = risk({0: 0.90, 1: 0.00, 2: 0.10})
    accepted, _, _ = evaluate_gate(
        candidate, choice(candidate, candidate, 0.95, 0.95), 0.95, tail, [], ModelStatus.OK
    )
    assert accepted  # P(L=2) exactly 0.10 passes

    tail_above = risk({0: 0.899, 1: 0.00, 2: 0.101})
    accepted, _, reasons = evaluate_gate(
        candidate, choice(candidate, candidate, 0.95, 0.95), 0.95, tail_above, [], ModelStatus.OK
    )
    assert not accepted and ReasonCode.HIGH_RISK_MASS_TOO_HIGH in reasons


def test_gate_records_every_failure_without_short_circuit() -> None:
    candidate = "cand-1"
    accepted, checks, reasons = evaluate_gate(
        candidate,
        choice("keep_original", candidate, 0.60, 0.55),
        0.30,
        risk({0: 0.20, 1: 0.30, 2: 0.50}, sdk_score=1.9),
        [ReasonCode.PROTECTED_COLUMN],
        ModelStatus.OK,
    )
    assert not accepted
    assert {
        ReasonCode.PROTECTED_COLUMN,
        ReasonCode.MODEL_ABSTAIN,
        ReasonCode.LOW_CHOICE_CONFIDENCE,
        ReasonCode.POLICY_NOT_SUPPORTED,
        ReasonCode.EXPECTED_RISK_TOO_HIGH,
        ReasonCode.HIGH_RISK_MASS_TOO_HIGH,
        ReasonCode.INVALID_SCORE_EXPECTATION,
    } <= set(reasons)
    assert all(check.status == CheckStatus.FAILED for check in checks if check.reason_code)


def test_score_expectation_tolerance_and_replay_uses_recomputed_value() -> None:
    candidate = "cand-1"
    within = risk({0: 0.60, 1: 0.40, 2: 0.00}, sdk_score=0.43)
    accepted, _, _ = evaluate_gate(
        candidate, choice(candidate, candidate, 0.95, 0.95), 0.95, within, [], ModelStatus.OK
    )
    assert accepted
    outside = risk({0: 0.60, 1: 0.40, 2: 0.00}, sdk_score=0.40 + 0.031)
    accepted, _, reasons = evaluate_gate(
        candidate, choice(candidate, candidate, 0.95, 0.95), 0.95, outside, [], ModelStatus.OK
    )
    assert not accepted and ReasonCode.INVALID_SCORE_EXPECTATION in reasons


def test_not_called_gate_has_null_observations() -> None:
    accepted, checks, reasons = evaluate_gate(
        "cand-1", None, None, None, [], ModelStatus.NOT_CALLED
    )
    assert not accepted
    assert ReasonCode.MODEL_NOT_CALLED in reasons
    model_check = next(check for check in checks if check.rule_id == "MODEL_STATUS")
    assert model_check.status == CheckStatus.FAILED
    assert all(
        check.observed_value is None
        for check in checks
        if check.rule_id not in {"MODEL_STATUS", "HARD_ELIGIBILITY"}
    )


def test_distributions_reject_nan_inf_missing_and_bad_sums() -> None:
    with pytest.raises(InvalidDistributionError):
        summarize_choice(
            SimpleNamespace(choice="a", confidence=0.9, probabilities={"a": math.nan, "b": 0.1}),
            {"a": "x", "b": "y"},
            "a",
        )
    with pytest.raises(InvalidDistributionError):
        summarize_choice(
            SimpleNamespace(choice="a", confidence=0.9, probabilities={"a": math.inf, "b": 0.1}),
            {"a": "x", "b": "y"},
            "a",
        )
    with pytest.raises(InvalidDistributionError):
        summarize_choice(
            SimpleNamespace(choice="a", confidence=0.9, probabilities={"a": 0.5}),
            {"a": "x", "b": "y"},
            "a",
        )
    with pytest.raises(InvalidDistributionError):
        summarize_choice(
            SimpleNamespace(choice="a", confidence=0.9, probabilities={"a": 0.8, "b": 0.3}),
            {"a": "x", "b": "y"},
            "a",
        )
    with pytest.raises(InvalidDistributionError):
        summarize_choice(
            SimpleNamespace(choice="z", confidence=0.9, probabilities={"a": 0.9, "b": 0.1}),
            {"a": "x", "b": "y"},
            "a",
        )
    with pytest.raises(InvalidDistributionError):
        summarize_risk(
            SimpleNamespace(
                score=0.5,
                confidence=0.9,
                legend={0: "low", 1: "mid"},
                probabilities={0: 0.5, 1: 0.5},
            )
        )
    with pytest.raises(InvalidDistributionError):
        summarize_risk(
            SimpleNamespace(
                score=0.5,
                confidence=0.9,
                legend={0: "low", 1: "mid", 2: "high"},
                probabilities={0: 0.5, 1: 0.5, 2: 0.001},
            )
        )


def test_gate_reads_confidence_not_selected_probability() -> None:
    # Selected probability 0.99 but SDK confidence 0.50 must fail the confidence gate.
    candidate = "cand-1"
    assessment = choice(candidate, candidate, 0.99, 0.50)
    accepted, checks, reasons = evaluate_gate(
        candidate, assessment, 0.95, risk({0: 1.0, 1: 0.0, 2: 0.0}), [], ModelStatus.OK
    )
    assert not accepted and ReasonCode.LOW_CHOICE_CONFIDENCE in reasons
    confidence_check = next(check for check in checks if check.rule_id == "CHOICE_CONFIDENCE")
    assert confidence_check.observed_value == 0.50


def test_agent_batch_has_three_questions_per_candidate_and_references_state(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(
        tmp_path,
        ["id", "age"],
        [[f"{index:03d}", f" {20 + index} "] for index in range(6)],
    )
    config = load_config(config_path)
    state = load_csv(input_path, config)
    detection = detect_phase(state, config, Phase.TRIM)
    candidates = build_candidates(state, config, detection)
    assert len(candidates) == 1  # one candidate per issue, all rows in one group
    evidence = {item.id: item for item in detection.evidence}
    context = build_candidate_context(state, config, candidates[0], "standardization", evidence)
    questions = build_questions("standardization", [context])
    assert set(questions) == {"c0_action", "c0_applicable", "c0_risk"}
    assert "state.candidates[0]" in questions["c0_action"].instructions
    assert "state.candidates[0]" in questions["c0_applicable"].instructions
    assert "state.candidates[0]" in questions["c0_risk"].instructions
    assert len(questions["c0_risk"].criteria) == 3


def test_batch_size_capped_at_four(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    from idac.orchestrator import _fit_batch, _RunContext
    from idac.storage import RunStore

    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(
        tmp_path,
        ["id", "age"],
        [[f"{index:03d}", f" {20 + index} "] for index in range(30)],
    )
    config = load_config(config_path)
    state = load_csv(input_path, config)
    detection = detect_phase(state, config, Phase.TRIM)
    candidates = build_candidates(state, config, detection)
    # One issue yields one candidate; six synthetic candidates with the same shape
    # exercise the batch-size cap.
    from idac.models import Operation, RepairCandidate

    synthetic = [
        RepairCandidate(
            id=f"cand-{index}",
            version_id=state.version_id,
            issue_ids=[],
            operation=Operation.TRIM_WHITESPACE,
            selection_id=candidates[0].selection_id,
            parameters={"column": "age", "after_values": {}},
            preview=[],
            evidence_ids=[],
            fingerprint=f"fp-{index}",
        )
        for index in range(6)
    ]
    store = RunStore.create(tmp_path / "run", config_path, input_path, config, "fake")
    budget = None
    from idac.typesafe_client import RunBudget

    budget = RunBudget()
    client = FakeDecisionClient()
    context = _RunContext(store, config, state, client, budget)
    batch, sample_limit = _fit_batch(context, synthetic)
    assert batch is not None and len(batch) == 4
    assert sample_limit >= 0


def test_semantic_verifier_receives_actual_diff(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, client = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    postcheck_calls = [call for call in client.calls if call["kind"] == "postcheck"]
    assert postcheck_calls
    contexts = postcheck_calls[0]["state"]["candidates"]
    changes = store.read_changes()
    committed = [entry for entry in changes if entry["committed"]]
    assert committed
    committed_diff = {
        (change["row_id"], change["column"], change["before"], change["after"])
        for entry in committed
        for change in entry["changes"]
    }
    sampled = {
        (sample["row_id"], sample["column"], sample["before"], sample["after"])
        for context in contexts
        for sample in context["actual_diff_samples"]
    }
    assert sampled
    assert sampled <= committed_diff


def test_decision_record_records_model_and_request_metadata(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    decisions = store.read_decisions()
    assert decisions
    for record in decisions:
        assert record["model"] == "fake-jev-1.13.0"
        assert record["request_id"].startswith("fake-req")
        assert record["question_version"]
        assert record["policy_version"]
        assert record["policy_snapshot"]
        assert record["context_ref"] and record["questions_ref"]
        assert record["gate_checks"]


def test_response_model_version_is_preserved(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    request = store.read_requests()[0]
    assert request["model"] == "fake-jev-1.13.0"
    assert request["raw_answers"]
    assert request["attempts"] == [{"attempt": 1, "status": "ok", "error": None, "retry_after_seconds": None}]


def test_agent_marks_missing_answers_invalid(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", " 21 "], ["002", "22"]])
    config = load_config(config_path)
    state = load_csv(input_path, config)
    detection = detect_phase(state, config, Phase.TRIM)
    candidates = build_candidates(state, config, detection)
    evidence = {item.id: item for item in detection.evidence}

    class PartialClient(FakeDecisionClient):
        def evaluate(self, state, questions, *, kind="decision"):
            outcome = super().evaluate(state, questions, kind=kind)
            if kind == "decision":
                outcome.answers.pop("c0_applicable", None)
            return outcome

    agent = StandardizationAgent()
    result = agent.evaluate(state, config, candidates, evidence, PartialClient())
    record = result.records[0]
    assert record.model_status == ModelStatus.INVALID_RESPONSE
    assert ReasonCode.MISSING_ANSWER in record.reason_codes
    assert record.choice_assessment is None


def test_evidence_type_is_required_by_context(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", " 21 "], ["002", "22"]])
    config = load_config(config_path)
    state = load_csv(input_path, config)
    detection = detect_phase(state, config, Phase.TRIM)
    state = state.model_copy(update={"selections": dict(detection.selections)})
    candidates = build_candidates(state, config, detection)
    evidence = {item.id: item for item in detection.evidence}
    context = build_candidate_context(state, config, candidates[0], "standardization", evidence)
    assert isinstance(next(iter(evidence.values())), Evidence)
    assert context["target_count"] == 1
