"""Acceptance 14, 15, 17, 18, 19: distributions, math, replay and cards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fake_client import FakeDecisionClient, run_fake_pipeline

from idac.distributions import (
    InvalidDistributionError,
    expected_expansion,
    expected_level,
    high_risk_probability,
    level_variance,
    summarize_choice,
)
from idac.explanation import (
    decision_card,
    render_card_markdown,
    render_decision_table,
    render_expected_examples,
)
from idac.models import DecisionRecord, ModelStatus, decision_from_log
from idac.policy import evaluate_gate, replay_decision
from idac.report import build_cards_markdown

CONFIG_TEXT = """
table_description: "explanation test table"
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


def test_choice_distribution_keeps_zero_probability_options() -> None:
    assessment = summarize_choice(
        SimpleNamespace(
            choice="cand-1",
            confidence=0.9,
            probabilities={"cand-1": 1.0, "keep_original": 0.0},
        ),
        {"cand-1": "apply", "keep_original": "keep"},
        "cand-1",
    )
    assert assessment.probabilities == {"cand-1": 1.0, "keep_original": 0.0}
    assert assessment.selected_probability == 1.0
    assert assessment.top_two_margin == 1.0


def test_invalid_distributions_are_not_repaired() -> None:
    with pytest.raises(InvalidDistributionError):
        summarize_choice(
            SimpleNamespace(choice="a", confidence=0.9, probabilities={"a": 0.5, "b": 0.4}),
            {"a": "x", "b": "y"},
            "a",
        )
    with pytest.raises(InvalidDistributionError):
        summarize_choice(
            SimpleNamespace(choice="a", confidence=1.5, probabilities={"a": 0.9, "b": 0.1}),
            {"a": "x", "b": "y"},
            "a",
        )


def test_expected_level_variance_and_high_risk_math() -> None:
    a = {0: 0.60, 1: 0.40, 2: 0.00}
    b = {0: 0.80, 1: 0.00, 2: 0.20}
    assert expected_level(a) == pytest.approx(0.40)
    assert expected_level(b) == pytest.approx(0.40)
    assert level_variance(a, 0.40) == pytest.approx(0.24)
    assert level_variance(b, 0.40) == pytest.approx(0.64)
    assert high_risk_probability(a) == 0.0
    assert high_risk_probability(b) == 0.20
    assert expected_expansion(a) == "0*0.6 + 1*0.4 + 2*0 = 0.4"


def test_synthetic_ab_example_same_expected_different_gate() -> None:
    candidate = "cand-1"

    def risk(distribution: dict[int, float]):
        mean = expected_level(distribution)
        return SimpleNamespace(
            probabilities=distribution,
            expected_risk_level=mean,
            high_risk_probability=high_risk_probability(distribution),
            score_expectation_delta=0.0,
        )

    choice = SimpleNamespace(
        selected_option=candidate,
        probabilities={candidate: 0.95, "keep_original": 0.05},
        selected_probability=0.95,
        top_two_margin=0.9,
        sdk_confidence=0.95,
    )
    accepted_a, _, _ = evaluate_gate(
        candidate, choice, 0.95, risk({0: 0.60, 1: 0.40, 2: 0.00}), [], ModelStatus.OK
    )
    accepted_b, _, reasons = evaluate_gate(
        candidate, choice, 0.95, risk({0: 0.80, 1: 0.00, 2: 0.20}), [], ModelStatus.OK
    )
    assert accepted_a and not accepted_b
    assert any(reason.value == "HIGH_RISK_MASS_TOO_HIGH" for reason in reasons)
    rendered = render_expected_examples()
    assert "synthetic" in rendered.lower()
    assert "| A | 0.60 | 0.40 | 0.00 | 0.40 | 0.24 | pass | pass |" in rendered
    assert "| B | 0.80 | 0.00 | 0.20 | 0.40 | 0.64 | pass | fail |" in rendered


def test_decision_card_contains_distribution_and_expansion(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    records = [decision_from_log(entry) for entry in store.read_decisions()]
    card = decision_card(records[0])
    assert card["choice"]["distribution"]
    assert set(card["risk"]["distribution"]) == {0, 1, 2}
    assert card["expected_expansion"].startswith("0*")
    assert card["rules"]
    markdown = render_card_markdown(card)
    for heading in (
        "Object and evidence",
        "Available options",
        "Choice distribution",
        "Applicable (Noul)",
        "Risk distribution (Score)",
        "Expected level expansion",
        "Rule table",
        "Decision and execution status",
    ):
        assert heading in markdown
    assert "rounded for display" in markdown


def test_not_called_card_shows_na_and_no_fabricated_distribution() -> None:
    record = DecisionRecord.model_validate(
        {
            "id": "dec-not-called",
            "candidate_id": "cand-1",
            "role": "missing_value",
            "version_id": "v001",
            "question_version": "q",
            "context_ref": "requests/req-0001.json",
            "questions_ref": "requests/req-0001.json",
            "policy_version": "p",
            "policy_snapshot": {},
            "model_status": "not_called",
            "gate_checks": [
                {
                    "rule_id": "MODEL_STATUS",
                    "metric": "model_status",
                    "observed_value": "not_called",
                    "operator": "==",
                    "threshold_or_required_value": "ok",
                    "status": "failed",
                    "reason_code": "MODEL_NOT_CALLED",
                }
            ],
            "gate_accepted": False,
            "reason_codes": ["MODEL_NOT_CALLED"],
        }
    )
    card = decision_card(record)
    assert card["options"] is None
    assert card["choice"] is None
    assert card["risk"] is None
    assert card["expected_expansion"] is None
    markdown = render_card_markdown(card)
    assert "not obtained" in markdown
    assert "0.000000" not in markdown  # never a fabricated zero-risk distribution


def test_replay_decision_is_consistent_and_offline(tmp_path, monkeypatch) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    records = [decision_from_log(entry) for entry in store.read_decisions()]
    assert records

    def explode(*args, **kwargs):  # pragma: no cover - replay must not call the model
        raise AssertionError("replay must not call the model")

    monkeypatch.setattr(FakeDecisionClient, "evaluate", explode)
    for record in records:
        result = replay_decision(record, store.run_dir)
        assert result.replayable
        assert result.consistent, result.detail


def test_replay_detects_tampered_record_and_missing_policy(tmp_path) -> None:
    record = DecisionRecord.model_validate(
        {
            "id": "dec-1",
            "candidate_id": "cand-1",
            "role": "standardization",
            "version_id": "v001",
            "question_version": "q",
            "context_ref": "requests/req-0001.json",
            "questions_ref": "requests/req-0001.json",
            "policy_version": "p",
            "policy_snapshot": {
                "min_choice_confidence": 0.80,
                "min_applicable_probability": 0.80,
                "max_semantic_risk_score": 0.50,
                "max_high_risk_probability": 0.10,
                "score_expectation_tolerance": 0.03,
            },
            "model_status": "ok",
            "choice_assessment": {
                "selected_option": "cand-1",
                "option_descriptions": {"cand-1": "apply", "keep_original": "keep"},
                "probabilities": {"cand-1": 0.95, "keep_original": 0.05},
                "selected_probability": 0.95,
                "top_two_margin": 0.90,
                "sdk_confidence": 0.95,
            },
            "applicable_probability": 0.95,
            "risk_assessment": {
                "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.9, "1": 0.09, "2": 0.01},
                "sdk_score": 0.11,
                "expected_risk_level": 0.11,
                "score_expectation_delta": 0.0,
                "variance": 0.1099,
                "high_risk_probability": 0.01,
                "sdk_confidence": 0.9,
            },
            "gate_checks": [
                {
                    "rule_id": "HARD_ELIGIBILITY",
                    "metric": "eligibility_reasons",
                    "observed_value": "PROTECTED_COLUMN",
                    "operator": "==",
                    "threshold_or_required_value": "eligible",
                    "status": "failed",
                    "reason_code": "PROTECTED_COLUMN",
                }
            ],
            "gate_accepted": True,  # tampered: the stored gate disagrees with the checks
            "reason_codes": ["PROTECTED_COLUMN"],
        }
    )
    import json

    from idac.models import fingerprint

    context = {"candidate_id": record.candidate_id}
    questions = {"c0_action": {"criteria": {"keep_original": "keep"}}}
    payload = {"state": {"candidates": [context]}, "questions": questions}
    for reference in (record.context_ref, record.questions_ref):
        path = tmp_path / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    record = record.model_copy(update={
        "context_hash": fingerprint(context), "questions_hash": fingerprint(questions)
    })
    result = replay_decision(record, tmp_path)
    assert result.replayable
    assert not result.consistent  # stored gate_accepted disagrees with recomputation

    without_policy = record.model_copy(update={"policy_snapshot": {}})
    result = replay_decision(without_policy, tmp_path)
    assert not result.replayable


def test_decision_table_marks_rounded_values_and_commit_state(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    records = [decision_from_log(entry) for entry in store.read_decisions()]
    from idac.models import DecisionTrace

    traces = [DecisionTrace.model_validate(entry) for entry in store.read_traces()]
    table = render_decision_table(records, traces)
    assert "P(apply)" in table
    assert "P(L=2)" in table
    assert "accepted" in table
    assert "committed" in table
    assert "rounded for display" in table


def test_rollback_cards_remain_readable(tmp_path) -> None:
    rows = [["001", " 21 "], ["002", "22"]]
    _, store, _ = run_fake_pipeline(tmp_path, CONFIG_TEXT, ["id", "age"], rows)
    store.rollback("v000")
    records = [decision_from_log(entry) for entry in store.read_decisions()]
    from idac.models import DecisionTrace

    traces = [DecisionTrace.model_validate(entry) for entry in store.read_traces()]
    cards = build_cards_markdown(store, records, traces)
    assert "Historical" in cards
    assert "Risk distribution" in cards
