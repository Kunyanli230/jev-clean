"""The four role-specialized decision agents.

Each agent holds its own question template, bounded evidence and decision
records, but shares the same SDK wrapper and decision gate. An agent is a plain
Python component; there is no free-form model dialogue or autonomous planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import TableConfig
from .distributions import (
    InvalidDistributionError,
    summarize_choice,
    summarize_risk,
    validate_noul,
)
from .models import (
    CheckStatus,
    ClientOutcome,
    DatasetState,
    DecisionRecord,
    Evidence,
    GateCheck,
    ModelStatus,
    ReasonCode,
    RepairCandidate,
    fingerprint,
)
from .planner import eligibility_reasons
from .policy import evaluate_gate
from .questions import (
    CATEGORY_ROLE,
    ROLE_DUPLICATE,
    ROLE_MISSING,
    ROLE_OUTLIER,
    ROLE_STANDARDIZATION,
    build_candidate_context,
    build_questions,
    serialize_questions,
)
from .settings import MAX_SAMPLES_PER_CANDIDATE, POLICY_VERSION, QUESTION_VERSION, policy_snapshot
from .typesafe_client import DecisionClient


@dataclass
class AgentBatchResult:
    role: str
    contexts: list[dict[str, Any]]
    questions: dict[str, Any]
    outcome: ClientOutcome
    records: list[DecisionRecord] = field(default_factory=list)


class DecisionAgent:
    """Shared batch evaluation protocol; subclasses only fix the role."""

    role: str = ""

    def evaluate(
        self,
        state: DatasetState,
        config: TableConfig,
        candidates: list[RepairCandidate],
        evidence_map: dict[str, Evidence],
        client: DecisionClient,
        *,
        sample_limit: int = MAX_SAMPLES_PER_CANDIDATE,
    ) -> AgentBatchResult:
        contexts = [
            build_candidate_context(
                state, config, candidate, self.role, evidence_map, sample_limit=sample_limit
            )
            for candidate in candidates
        ]
        questions = build_questions(self.role, contexts)
        outcome = client.evaluate({"candidates": contexts}, questions, kind="decision")
        records = [
            self._record(state, config, candidate, index, contexts, questions, outcome, evidence_map)
            for index, candidate in enumerate(candidates)
        ]
        return AgentBatchResult(
            role=self.role,
            contexts=contexts,
            questions=questions,
            outcome=outcome,
            records=records,
        )

    def _record(
        self,
        state: DatasetState,
        config: TableConfig,
        candidate: RepairCandidate,
        index: int,
        contexts: list[dict[str, Any]],
        questions: dict[str, Any],
        outcome: ClientOutcome,
        evidence_map: dict[str, Evidence],
    ) -> DecisionRecord:
        eligibility = eligibility_reasons(state, config, evidence_map, candidate)
        decision_id = (
            "dec-"
            + fingerprint(
                {
                    "candidate_id": candidate.id,
                    "role": self.role,
                    "version_id": state.version_id,
                    "question_version": QUESTION_VERSION,
                }
            )[:16]
        )
        selection = state.selection(candidate.selection_id)
        question_subset = {
            name: question
            for name, question in questions.items()
            if name.startswith(f"c{index}_")
        }
        base: dict[str, Any] = {
            "id": decision_id,
            "candidate_id": candidate.id,
            "role": self.role,
            "version_id": state.version_id,
            "question_version": QUESTION_VERSION,
            "operation": candidate.operation.value,
            "column": candidate.parameters.get("column")
            or (selection.column if selection else None),
            "target_count": len(selection.row_ids) if selection else None,
            "context_ref": f"requests/candidate/{candidate.id}",
            "questions_ref": f"requests/questions/{candidate.id}",
            "context_hash": fingerprint(contexts[index]),
            "questions_hash": fingerprint(serialize_questions(question_subset)),
            "evidence_ids": list(candidate.evidence_ids),
            "policy_version": POLICY_VERSION,
            "policy_snapshot": policy_snapshot(),
            "model_status": outcome.model_status,
            "request_id": outcome.request_id,
            "model": outcome.model,
        }
        if outcome.model_status != ModelStatus.OK:
            accepted, checks, reasons = evaluate_gate(
                candidate.id, None, None, None, eligibility, outcome.model_status
            )
            return DecisionRecord(
                **base,
                gate_checks=checks,
                gate_accepted=False,
                reason_codes=reasons,
            )
        answers = outcome.answers or {}
        action = answers.get(f"c{index}_action")
        applicable_answer = answers.get(f"c{index}_applicable")
        risk_answer = answers.get(f"c{index}_risk")
        if action is None or applicable_answer is None or risk_answer is None:
            checks = [
                GateCheck(
                    rule_id="MODEL_STATUS",
                    metric="answer_keys",
                    observed_value="missing",
                    operator="required",
                    threshold_or_required_value=(
                        f"c{index}_action, c{index}_applicable, c{index}_risk"
                    ),
                    status=CheckStatus.FAILED,
                    reason_code=ReasonCode.MISSING_ANSWER,
                )
            ]
            return DecisionRecord(
                **{**base, "model_status": ModelStatus.INVALID_RESPONSE},
                gate_checks=checks,
                gate_accepted=False,
                reason_codes=[ReasonCode.MISSING_ANSWER],
            )
        option_descriptions = _option_descriptions(questions, index)
        try:
            choice = summarize_choice(action, option_descriptions, candidate.id)
            applicable = validate_noul(applicable_answer)
            risk = summarize_risk(
                risk_answer, dict(enumerate(questions[f"c{index}_risk"].criteria))
            )
        except InvalidDistributionError as error:
            checks = [
                GateCheck(
                    rule_id="DISTRIBUTION_VALID",
                    metric="distribution",
                    observed_value="invalid",
                    operator="valid",
                    threshold_or_required_value="complete finite distribution",
                    status=CheckStatus.FAILED,
                    reason_code=error.reason,
                )
            ]
            return DecisionRecord(
                **{**base, "model_status": ModelStatus.INVALID_RESPONSE},
                gate_checks=checks,
                gate_accepted=False,
                reason_codes=[error.reason],
            )
        accepted, checks, reasons = evaluate_gate(
            candidate.id, choice, applicable, risk, eligibility, outcome.model_status
        )
        return DecisionRecord(
            **base,
            choice_assessment=choice,
            applicable_probability=applicable,
            risk_assessment=risk,
            gate_checks=checks,
            gate_accepted=accepted,
            reason_codes=reasons,
        )


def _option_descriptions(questions: dict[str, Any], index: int) -> dict[str, str]:
    question = questions.get(f"c{index}_action")
    criteria = getattr(question, "criteria", None) or {}
    return {str(key): str(value) for key, value in criteria.items()}


class StandardizationAgent(DecisionAgent):
    role = ROLE_STANDARDIZATION


class DuplicateAgent(DecisionAgent):
    role = ROLE_DUPLICATE


class OutlierAgent(DecisionAgent):
    role = ROLE_OUTLIER


class MissingValueAgent(DecisionAgent):
    role = ROLE_MISSING


_AGENTS: dict[str, DecisionAgent] = {
    ROLE_STANDARDIZATION: StandardizationAgent(),
    ROLE_DUPLICATE: DuplicateAgent(),
    ROLE_OUTLIER: OutlierAgent(),
    ROLE_MISSING: MissingValueAgent(),
}


def agent_for_category(category) -> DecisionAgent:
    """Deterministic issue router: category -> role -> agent."""
    return _AGENTS[CATEGORY_ROLE[category]]


def agent_for_role(role: str) -> DecisionAgent:
    return _AGENTS[role]
