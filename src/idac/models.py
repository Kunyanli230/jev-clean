"""Pydantic data model shared by every IDAC component.

The interfaces in this module are frozen first: profiling, candidates, agents,
planner, executor, validator and storage all exchange these types. Cells are
stored as strings or ``None`` only; logical types come from the table schema.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

CellValue = str | None


class Operation(StrEnum):
    """The seven deterministic cleaning operations."""

    TRIM_WHITESPACE = "trim_whitespace"
    NORMALIZE_NULL_TOKENS = "normalize_null_tokens"
    CAST_NUMERIC = "cast_numeric"
    PARSE_DATETIME = "parse_datetime"
    REMOVE_EXACT_DUPLICATES = "remove_exact_duplicates"
    INVALIDATE_OUT_OF_RANGE = "invalidate_out_of_range"
    IMPUTE_MISSING = "impute_missing"


class Phase(StrEnum):
    """The fixed seven sub-phases, executed in this order."""

    TRIM = "trim"
    NULL_NORMALIZATION = "null_normalization"
    NUMERIC_CAST = "numeric_cast"
    DATE_PARSING = "date_parsing"
    DUPLICATES = "duplicates"
    OUT_OF_RANGE = "out_of_range"
    IMPUTATION = "imputation"


PHASE_ORDER: list[Phase] = [
    Phase.TRIM,
    Phase.NULL_NORMALIZATION,
    Phase.NUMERIC_CAST,
    Phase.DATE_PARSING,
    Phase.DUPLICATES,
    Phase.OUT_OF_RANGE,
    Phase.IMPUTATION,
]

PHASE_OPERATION: dict[Phase, Operation] = {
    Phase.TRIM: Operation.TRIM_WHITESPACE,
    Phase.NULL_NORMALIZATION: Operation.NORMALIZE_NULL_TOKENS,
    Phase.NUMERIC_CAST: Operation.CAST_NUMERIC,
    Phase.DATE_PARSING: Operation.PARSE_DATETIME,
    Phase.DUPLICATES: Operation.REMOVE_EXACT_DUPLICATES,
    Phase.OUT_OF_RANGE: Operation.INVALIDATE_OUT_OF_RANGE,
    Phase.IMPUTATION: Operation.IMPUTE_MISSING,
}


class IssueCategory(StrEnum):
    WHITESPACE = "whitespace"
    NULL_TOKEN = "null_token"
    NUMERIC_FORMAT = "numeric_format"
    DATE_FORMAT = "date_format"
    DUPLICATE_RECORD = "duplicate_record"
    OUT_OF_RANGE = "out_of_range"
    MISSING_VALUE = "missing_value"
    IQR_OUTLIER = "iqr_outlier"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    PARSE_FAILURE = "parse_failure"
    AMBIGUOUS_PARSE = "ambiguous_parse"
    STALE_VERSION = "stale_version"
    MODEL_FAILURE = "model_failure"
    BUDGET_STOP = "budget_stop"


class IssueStatus(StrEnum):
    OPEN = "open"
    REPAIRED = "repaired"
    UNRESOLVED = "unresolved"
    INFORMATIONAL = "informational"


class ModelStatus(StrEnum):
    OK = "ok"
    NOT_CALLED = "not_called"
    FAILED = "failed"
    INVALID_RESPONSE = "invalid_response"


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class RunOutcome(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class RunState(StrEnum):
    LOADED = "LOADED"
    PROFILING = "PROFILING"
    BUILDING_CANDIDATES = "BUILDING_CANDIDATES"
    DECIDING = "DECIDING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class ReasonCode(StrEnum):
    """Reason codes attached to issues, decisions and traces."""

    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"
    PROTECTED_COLUMN = "PROTECTED_COLUMN"
    CANDIDATE_UNAVAILABLE = "CANDIDATE_UNAVAILABLE"
    PARSE_AMBIGUOUS = "PARSE_AMBIGUOUS"
    PARSE_FAILURE = "PARSE_FAILURE"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    NO_VALID_DONORS = "NO_VALID_DONORS"
    ALL_VALUES_NULL = "ALL_VALUES_NULL"
    MODEL_ABSTAIN = "MODEL_ABSTAIN"
    LOW_CHOICE_CONFIDENCE = "LOW_CHOICE_CONFIDENCE"
    POLICY_NOT_SUPPORTED = "POLICY_NOT_SUPPORTED"
    EXPECTED_RISK_TOO_HIGH = "EXPECTED_RISK_TOO_HIGH"
    HIGH_RISK_MASS_TOO_HIGH = "HIGH_RISK_MASS_TOO_HIGH"
    INVALID_SCORE_EXPECTATION = "INVALID_SCORE_EXPECTATION"
    INVALID_DISTRIBUTION = "INVALID_DISTRIBUTION"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    MISSING_ANSWER = "MISSING_ANSWER"
    API_FAILURE = "API_FAILURE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MODEL_NOT_CALLED = "MODEL_NOT_CALLED"
    STALE_VERSION = "STALE_VERSION"
    HARD_VALIDATION_FAILED = "HARD_VALIDATION_FAILED"
    SEMANTIC_VALIDATION_FAILED = "SEMANTIC_VALIDATION_FAILED"
    POSTCHECK_ABOVE_THRESHOLD = "POSTCHECK_ABOVE_THRESHOLD"
    POSTCHECK_UNAVAILABLE = "POSTCHECK_UNAVAILABLE"
    BATCH_REJECTED = "BATCH_REJECTED"
    CANDIDATE_CONFLICT = "CANDIDATE_CONFLICT"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    NO_CHANGE = "NO_CHANGE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    CONFIGURATION_FAILURE = "CONFIGURATION_FAILURE"
    RUN_TIME_LIMIT = "RUN_TIME_LIMIT"
    UNSUPPORTED_COLUMN = "UNSUPPORTED_COLUMN"
    INFORMATIONAL_IQR = "INFORMATIONAL_IQR"
    REPLAY_UNAVAILABLE = "REPLAY_UNAVAILABLE"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(_Model):
    id: str
    version_id: str
    check_name: str
    counts: dict[str, int | float] = Field(default_factory=dict)
    summary: str = ""
    details: list[dict[str, Any]] = Field(default_factory=list)


class Selection(_Model):
    id: str
    version_id: str
    row_ids: list[str] = Field(default_factory=list)
    column: str | None = None
    fingerprint: str = ""


class Issue(_Model):
    id: str
    category: IssueCategory
    column: str | None = None
    selection_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    status: IssueStatus = IssueStatus.OPEN
    reason_codes: list[ReasonCode] = Field(default_factory=list)


class CellPreview(_Model):
    row_id: str
    column: str
    before: CellValue = None
    after: CellValue = None


class RepairCandidate(_Model):
    id: str
    version_id: str
    issue_ids: list[str] = Field(default_factory=list)
    operation: Operation
    selection_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    preview: list[CellPreview] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    fingerprint: str = ""


class RemovedRow(_Model):
    row_id: str
    values: list[CellValue] = Field(default_factory=list)
    reason: str = ""
    candidate_id: str | None = None


class ChoiceAssessment(_Model):
    selected_option: str
    option_descriptions: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    selected_probability: float
    top_two_margin: float
    sdk_confidence: float


class RiskAssessment(_Model):
    legend: dict[int, str] = Field(default_factory=dict)
    probabilities: dict[int, float] = Field(default_factory=dict)
    sdk_score: float
    expected_risk_level: float
    score_expectation_delta: float
    variance: float
    high_risk_probability: float
    sdk_confidence: float


class GateCheck(_Model):
    rule_id: str
    metric: str
    observed_value: float | int | str | None = None
    operator: str = ""
    threshold_or_required_value: float | int | str | None = None
    status: CheckStatus = CheckStatus.NOT_EVALUATED
    reason_code: ReasonCode | None = None


class DecisionRecord(_Model):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _check_status_completeness(self) -> DecisionRecord:
        if self.model_status == ModelStatus.OK:
            if (
                self.choice_assessment is None
                or self.risk_assessment is None
                or self.applicable_probability is None
                or not self.gate_checks
            ):
                raise ValueError(
                    "a valid model decision requires complete choice/risk assessments, "
                    "the applicable probability and gate checks"
                )
        elif not self.reason_codes:
            raise ValueError("a not-called/failed/invalid decision requires reason codes")
        return self

    id: str
    candidate_id: str
    role: str
    version_id: str
    question_version: str
    operation: str | None = None
    column: str | None = None
    target_count: int | None = None
    context_ref: str
    questions_ref: str
    context_hash: str = ""
    questions_hash: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    policy_version: str
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_status: ModelStatus
    choice_assessment: ChoiceAssessment | None = None
    applicable_probability: float | None = None
    risk_assessment: RiskAssessment | None = None
    gate_checks: list[GateCheck] = Field(default_factory=list)
    gate_accepted: bool = False
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    request_id: str | None = None
    model: str | None = None


class DecisionTrace(_Model):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    candidate_id: str
    plan_id: str | None = None
    planner_status: str = "not_planned"
    hard_validation_status: str = "not_run"
    postcheck_status: str = "not_run"
    postcheck_probability: float | None = None
    commit_status: str = "not_committed"
    committed_version: str | None = None
    blocking_candidate_ids: list[str] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)


class PlannedOperation(_Model):
    operation: Operation
    candidate_ids: list[str] = Field(default_factory=list)
    selection_id: str
    column: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    after_values: dict[str, CellValue] = Field(default_factory=dict)
    removed_row_ids: list[str] = Field(default_factory=list)


class RepairPlan(_Model):
    id: str
    version_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    ordered_operations: list[PlannedOperation] = Field(default_factory=list)
    discarded: list[dict[str, Any]] = Field(default_factory=list)


class CellChange(_Model):
    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: str
    column: str
    before: CellValue = None
    after: CellValue = None
    candidate_id: str | None = None


class SemanticCheck(_Model):
    candidate_id: str
    probability: float
    threshold: float
    request_id: str | None = None
    model: str | None = None
    passed: bool
    reason_code: ReasonCode | None = None


class ValidationResult(_Model):
    plan_id: str
    version_id: str
    hard_checks: list[GateCheck] = Field(default_factory=list)
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)
    passed: bool = False
    reason_codes: list[ReasonCode] = Field(default_factory=list)


class AuditEvent(_Model):
    timestamp: str
    event_type: str
    version_id: str | None = None
    plan_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Field name "schema"')

    class DatasetState(_Model):
        """Logical dataset snapshot: strings/null cells plus schema and audit state."""

        version_id: str
        parent_version_id: str | None = None
        input_hash: str
        ordered_columns: list[str] = Field(default_factory=list)
        row_ids: list[str] = Field(default_factory=list)
        rows: list[list[CellValue]] = Field(default_factory=list)
        schema: dict[str, str] = Field(default_factory=dict)
        removed_rows: list[RemovedRow] = Field(default_factory=list)
        issues: list[Issue] = Field(default_factory=list)
        selections: dict[str, Selection] = Field(default_factory=dict)

        def row_index(self) -> dict[str, int]:
            return {row_id: index for index, row_id in enumerate(self.row_ids)}

        def column_index(self) -> dict[str, int]:
            return {column: index for index, column in enumerate(self.ordered_columns)}

        def selection(self, selection_id: str) -> Selection | None:
            return self.selections.get(selection_id)


class RequestAttempt(_Model):
    attempt: int
    status: str
    error: str | None = None
    retry_after_seconds: float | None = None


class UsageInfo(_Model):
    input_tokens: int | None = None
    output_tokens: int | None = None
    known: bool = False


class ClientOutcome(_Model):
    """Normalized result of one decision request, successful or not."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_status: ModelStatus
    model: str | None = None
    request_id: str | None = None
    answers: dict[str, Any] | None = None
    usage: UsageInfo | None = None
    error: str | None = None
    invalid_details: dict[str, Any] | None = None
    attempts: list[RequestAttempt] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    raw_answers: dict[str, Any] | None = None


def decision_from_log(entry: dict[str, Any]) -> DecisionRecord:
    """Validate a decisions.jsonl entry, dropping the log-only ``committed`` flag."""
    payload = {key: value for key, value in entry.items() if key != "committed"}
    return DecisionRecord.model_validate(payload)


def fingerprint(payload: Any) -> str:
    """Stable sha256 fingerprint of a JSON-serializable payload."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
