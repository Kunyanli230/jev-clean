"""Decision gate and offline policy replay.

The gate evaluates every rule (no short-circuit), records all failures and only
accepts a candidate for planning. Replay recomputes the gate from a stored
record using the historical policy snapshot; it never calls the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .distributions import expected_level, high_risk_probability
from .models import (
    CheckStatus,
    ChoiceAssessment,
    DecisionRecord,
    GateCheck,
    ModelStatus,
    ReasonCode,
    RiskAssessment,
    fingerprint,
)
from .settings import (
    MAX_HIGH_RISK_PROBABILITY,
    MAX_SEMANTIC_RISK_SCORE,
    MIN_APPLICABLE_PROBABILITY,
    MIN_CHOICE_CONFIDENCE,
    SCORE_EXPECTATION_TOLERANCE,
)

KEEP_ORIGINAL = "keep_original"


def _check(
    rule_id: str,
    metric: str,
    observed: float | str | None,
    operator: str,
    threshold: float | str | None,
    passed: bool | None,
    reason_code: ReasonCode | None,
) -> GateCheck:
    status = CheckStatus.NOT_EVALUATED if passed is None else (CheckStatus.PASSED if passed else CheckStatus.FAILED)
    return GateCheck(
        rule_id=rule_id,
        metric=metric,
        observed_value=observed,
        operator=operator,
        threshold_or_required_value=threshold,
        status=status,
        reason_code=None if passed else reason_code,
    )


def evaluate_gate(
    candidate_id: str,
    choice: ChoiceAssessment | None,
    applicable: float | None,
    risk: RiskAssessment | None,
    eligibility_reasons: list[ReasonCode],
    model_status: ModelStatus,
    thresholds: dict[str, Any] | None = None,
) -> tuple[bool, list[GateCheck], list[ReasonCode]]:
    """Evaluate all gate rules and collect every failure reason."""
    values = thresholds or {
        "min_choice_confidence": MIN_CHOICE_CONFIDENCE,
        "min_applicable_probability": MIN_APPLICABLE_PROBABILITY,
        "max_semantic_risk_score": MAX_SEMANTIC_RISK_SCORE,
        "max_high_risk_probability": MAX_HIGH_RISK_PROBABILITY,
        "score_expectation_tolerance": SCORE_EXPECTATION_TOLERANCE,
    }
    checks: list[GateCheck] = []
    reasons: list[ReasonCode] = []

    def add(check: GateCheck) -> None:
        checks.append(check)
        if check.status == CheckStatus.FAILED and check.reason_code is not None:
            reasons.append(check.reason_code)

    add(
        _check(
            "HARD_ELIGIBILITY",
            "eligibility_reasons",
            ";".join(reason.value for reason in eligibility_reasons) or "eligible",
            "==",
            "eligible",
            not eligibility_reasons,
            eligibility_reasons[0] if eligibility_reasons else None,
        )
    )
    add(
        _check(
            "MODEL_STATUS",
            "model_status",
            model_status.value,
            "==",
            ModelStatus.OK.value,
            model_status == ModelStatus.OK,
            _model_reason(model_status),
        )
    )
    add(
        _check(
            "CHOICE_SELECTION",
            "selected_option",
            choice.selected_option if choice else None,
            "==",
            candidate_id,
            None if choice is None else choice.selected_option == candidate_id,
            None if choice is None else (
                ReasonCode.MODEL_ABSTAIN
                if choice.selected_option == KEEP_ORIGINAL
                else ReasonCode.INVALID_DISTRIBUTION
            ),
        )
    )
    add(
        _check(
            "CHOICE_CONFIDENCE",
            "sdk_confidence",
            choice.sdk_confidence if choice else None,
            ">=",
            values["min_choice_confidence"],
            None if choice is None else choice.sdk_confidence >= values["min_choice_confidence"],
            ReasonCode.LOW_CHOICE_CONFIDENCE,
        )
    )
    add(
        _check(
            "APPLICABLE_PROBABILITY",
            "applicable_probability",
            applicable,
            ">=",
            values["min_applicable_probability"],
            None if applicable is None else applicable >= values["min_applicable_probability"],
            ReasonCode.POLICY_NOT_SUPPORTED,
        )
    )
    delta = risk.score_expectation_delta if risk else None
    add(
        _check(
            "SCORE_EXPECTATION_CONSISTENT",
            "abs(score_expectation_delta)",
            None if delta is None else abs(delta),
            "<=",
            values["score_expectation_tolerance"],
            None if delta is None else abs(delta) <= values["score_expectation_tolerance"],
            ReasonCode.INVALID_SCORE_EXPECTATION,
        )
    )
    mean = risk.expected_risk_level if risk else None
    add(
        _check(
            "EXPECTED_RISK_LEVEL",
            "expected_risk_level",
            mean,
            "<=",
            values["max_semantic_risk_score"],
            None if mean is None else mean <= values["max_semantic_risk_score"],
            ReasonCode.EXPECTED_RISK_TOO_HIGH,
        )
    )
    high = risk.high_risk_probability if risk else None
    add(
        _check(
            "HIGH_RISK_PROBABILITY",
            "P(risk_level=2)",
            high,
            "<=",
            values["max_high_risk_probability"],
            None if high is None else high <= values["max_high_risk_probability"],
            ReasonCode.HIGH_RISK_MASS_TOO_HIGH,
        )
    )
    accepted = all(check.status == CheckStatus.PASSED for check in checks)
    return accepted, checks, reasons


def _model_reason(status: ModelStatus) -> ReasonCode:
    if status == ModelStatus.NOT_CALLED:
        return ReasonCode.MODEL_NOT_CALLED
    if status == ModelStatus.FAILED:
        return ReasonCode.API_FAILURE
    return ReasonCode.INVALID_RESPONSE


@dataclass
class ReplayResult:
    decision_id: str
    replayable: bool
    consistent: bool
    checks: list[GateCheck] = field(default_factory=list)
    reason_codes: list[ReasonCode] = field(default_factory=list)
    detail: str = ""


def verify_request_evidence(record: DecisionRecord | dict[str, Any], run_dir: str | Path | None) -> bool:
    """Resolve bounded request references and verify their per-candidate hashes."""
    if isinstance(record, dict):
        from .models import decision_from_log

        record = decision_from_log(record)
    if run_dir is None or not record.context_ref or not record.questions_ref:
        return False
    root = Path(run_dir).resolve()
    try:
        payloads = []
        for reference in (record.context_ref, record.questions_ref):
            path = (root / reference).resolve()
            if not path.is_relative_to(root):
                return False
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        context_payload, question_payload = payloads
        matches = [
            context for context in context_payload["state"]["candidates"]
            if context["candidate_id"] == record.candidate_id
        ]
        indices = [
            index for index, context in enumerate(question_payload["state"]["candidates"])
            if context["candidate_id"] == record.candidate_id
        ]
        if len(matches) != 1 or len(indices) != 1:
            return False
        prefix = f"c{indices[0]}_"
        questions = {
            key: value for key, value in question_payload["questions"].items()
            if key.startswith(prefix)
        }
        return (
            bool(questions)
            and fingerprint(matches[0]) == record.context_hash
            and fingerprint(questions) == record.questions_hash
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def replay_decision(
    record: DecisionRecord | dict[str, Any], run_dir: str | Path | None = None
) -> ReplayResult:
    """Recompute the gate from a stored record using its historical policy."""
    if isinstance(record, dict):
        from .models import decision_from_log

        record = decision_from_log(record)
    if record.model_status != ModelStatus.OK:
        return ReplayResult(
            decision_id=record.id,
            replayable=False,
            consistent=False,
            detail=f"model_status is {record.model_status.value}; no distribution to replay",
        )
    if record.choice_assessment is None or record.risk_assessment is None:
        return ReplayResult(
            decision_id=record.id,
            replayable=False,
            consistent=False,
            detail="stored record has no complete choice/risk assessment",
        )
    if record.applicable_probability is None:
        return ReplayResult(
            decision_id=record.id,
            replayable=False,
            consistent=False,
            detail="stored record has no applicable probability",
        )
    if not verify_request_evidence(record, run_dir):
        return ReplayResult(
            decision_id=record.id,
            replayable=False,
            consistent=False,
            detail="request evidence is missing, invalid, or does not match its stored hashes",
        )
    thresholds = {
        key: record.policy_snapshot[key]
        for key in (
            "min_choice_confidence",
            "min_applicable_probability",
            "max_semantic_risk_score",
            "max_high_risk_probability",
            "score_expectation_tolerance",
        )
        if key in record.policy_snapshot
    }
    if len(thresholds) != 5:
        return ReplayResult(
            decision_id=record.id,
            replayable=False,
            consistent=False,
            detail="policy snapshot is incomplete",
        )
    eligibility_reasons = [
        check.reason_code
        for check in record.gate_checks
        if check.rule_id == "HARD_ELIGIBILITY" and check.status == CheckStatus.FAILED
    ]
    accepted, checks, reasons = evaluate_gate(
        record.candidate_id,
        record.choice_assessment,
        record.applicable_probability,
        record.risk_assessment,
        [reason for reason in eligibility_reasons if reason is not None],
        record.model_status,
        thresholds,
    )
    expected = expected_level(record.risk_assessment.probabilities)
    high = high_risk_probability(record.risk_assessment.probabilities)
    consistent = (
        accepted == record.gate_accepted
        and reasons == list(record.reason_codes)
        and abs(expected - record.risk_assessment.expected_risk_level) <= 1e-12
        and abs(high - record.risk_assessment.high_risk_probability) <= 1e-12
    )
    detail = "recomputed gate matches the stored record"
    if not consistent:
        detail = (
            "recomputed gate differs from the stored record: "
            f"accepted {accepted} vs {record.gate_accepted}, "
            f"reasons {[reason.value for reason in reasons]} vs "
            f"{[reason.value for reason in record.reason_codes]}"
        )
    return ReplayResult(
        decision_id=record.id,
        replayable=True,
        consistent=consistent,
        checks=checks,
        reason_codes=reasons,
        detail=detail,
    )
