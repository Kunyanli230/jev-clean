"""Question templates and bounded state for the four decision roles.

Questions are written in English, explicitly address ``state.candidates[index]``
and never ask the model to compute something code already knows. Only relevant
configuration, summaries and bounded samples leave the machine.
"""

from __future__ import annotations

from typing import Any

from typesafe_sdk import Choice, Noul, Score

from .config import TableConfig
from .models import (
    CellChange,
    DatasetState,
    Evidence,
    IssueCategory,
    Operation,
    RepairCandidate,
    RepairPlan,
)
from .settings import MAX_CELL_CHARS, MAX_SAMPLES_PER_CANDIDATE, RISK_LEVEL_CRITERIA

ROLE_STANDARDIZATION = "standardization"
ROLE_DUPLICATE = "duplicate"
ROLE_OUTLIER = "outlier"
ROLE_MISSING = "missing_value"

CATEGORY_ROLE: dict[IssueCategory, str] = {
    IssueCategory.WHITESPACE: ROLE_STANDARDIZATION,
    IssueCategory.NULL_TOKEN: ROLE_STANDARDIZATION,
    IssueCategory.NUMERIC_FORMAT: ROLE_STANDARDIZATION,
    IssueCategory.DATE_FORMAT: ROLE_STANDARDIZATION,
    IssueCategory.DUPLICATE_RECORD: ROLE_DUPLICATE,
    IssueCategory.OUT_OF_RANGE: ROLE_OUTLIER,
    IssueCategory.MISSING_VALUE: ROLE_MISSING,
}

OPERATION_ROLE: dict[Operation, str] = {
    Operation.TRIM_WHITESPACE: ROLE_STANDARDIZATION,
    Operation.NORMALIZE_NULL_TOKENS: ROLE_STANDARDIZATION,
    Operation.CAST_NUMERIC: ROLE_STANDARDIZATION,
    Operation.PARSE_DATETIME: ROLE_STANDARDIZATION,
    Operation.REMOVE_EXACT_DUPLICATES: ROLE_DUPLICATE,
    Operation.INVALIDATE_OUT_OF_RANGE: ROLE_OUTLIER,
    Operation.IMPUTE_MISSING: ROLE_MISSING,
}

KEEP_ORIGINAL = "keep_original"

_ACTION_INSTRUCTIONS = {
    ROLE_STANDARDIZATION: (
        "For `state.candidates[{index}]`, should the proposed representation change be "
        "applied to this field, or should the original values be retained?"
    ),
    ROLE_DUPLICATE: (
        "For `state.candidates[{index}]`, should the redundant duplicate rows be removed, "
        "or should all original rows be retained?"
    ),
    ROLE_OUTLIER: (
        "For `state.candidates[{index}]`, should the out-of-range values be set to null, "
        "or should the original values be retained?"
    ),
    ROLE_MISSING: (
        "For `state.candidates[{index}]`, should the missing values be filled with the "
        "computed representative value, or should they stay missing?"
    ),
}

_APPLICABLE_INSTRUCTIONS = {
    ROLE_STANDARDIZATION: (
        "Do the field description, record grain and cleaning policy in "
        "`state.candidates[{index}]` support the business assumptions required by its "
        "proposed representation change?"
    ),
    ROLE_DUPLICATE: (
        "Do the table description, record grain and duplicate policy in "
        "`state.candidates[{index}]` support treating the listed rows as redundant copies "
        "of the kept rows?"
    ),
    ROLE_OUTLIER: (
        "Do the field description, declared numeric range and cleaning policy in "
        "`state.candidates[{index}]` support treating these measurements as invalid?"
    ),
    ROLE_MISSING: (
        "Do the field description and cleaning policy in `state.candidates[{index}]` "
        "support replacing the missing values with this representative value?"
    ),
}

_RISK_INSTRUCTIONS = {
    ROLE_STANDARDIZATION: (
        "How much risk of changing the business meaning does the proposed representation "
        "change in `state.candidates[{index}]` introduce?"
    ),
    ROLE_DUPLICATE: (
        "How much risk of losing meaningful business records does removing the rows in "
        "`state.candidates[{index}]` introduce?"
    ),
    ROLE_OUTLIER: (
        "How much risk of discarding meaningful information does setting the out-of-range "
        "values in `state.candidates[{index}]` to null introduce?"
    ),
    ROLE_MISSING: (
        "How much risk of changing the field's business meaning does filling the missing "
        "values in `state.candidates[{index}]` with this representative value introduce?"
    ),
}


def truncate_cell(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= MAX_CELL_CHARS:
        return value
    return value[: MAX_CELL_CHARS - 1] + "\u2026"


def build_candidate_context(
    state: DatasetState,
    config: TableConfig,
    candidate: RepairCandidate,
    role: str,
    evidence_map: dict[str, Evidence],
    *,
    sample_limit: int = MAX_SAMPLES_PER_CANDIDATE,
) -> dict[str, Any]:
    """Bounded, factual context for one candidate; no full data leaves the machine."""
    selection = state.selection(candidate.selection_id)
    row_ids = selection.row_ids if selection else []
    column_name = candidate.parameters.get("column") or (selection.column if selection else None)
    column_config = config.columns.get(column_name) if column_name else None
    samples = [
        {
            "row_id": preview.row_id,
            "column": preview.column,
            "before": truncate_cell(preview.before),
            "after": truncate_cell(preview.after),
        }
        for preview in candidate.preview[:sample_limit]
    ]
    context: dict[str, Any] = {
        "candidate_id": candidate.id,
        "role": role,
        "operation": candidate.operation.value,
        "column": column_name,
        "column_type": column_config.type if column_config else None,
        "column_description": column_config.description if column_config else None,
        "column_protected": column_config.protected if column_config else None,
        "allowed_operations": (
            [operation.value for operation in column_config.allowed_operations]
            if column_config
            else []
        ),
        "table_description": config.table_description,
        "record_grain": config.record_grain,
        "selection_id": candidate.selection_id,
        "target_count": len(row_ids),
        "samples": samples,
        "sample_note": (
            "Samples illustrate the policy question for this group; local validation "
            "checks every target row."
        ),
        "evidence_summary": [
            evidence_map[evidence_id].summary
            for evidence_id in candidate.evidence_ids
            if evidence_id in evidence_map
        ],
    }
    if candidate.operation == Operation.IMPUTE_MISSING:
        context["proposed_value"] = {
            "method": candidate.parameters.get("method"),
            "fill_value": candidate.parameters.get("fill_value"),
            "donor_count": candidate.parameters.get("donor_count"),
        }
    if candidate.operation == Operation.INVALIDATE_OUT_OF_RANGE and column_config:
        context["declared_range"] = {
            "min": str(column_config.min_value),
            "max": str(column_config.max_value),
        }
    if candidate.operation == Operation.REMOVE_EXACT_DUPLICATES:
        context["duplicate_policy"] = {
            "compare_columns": candidate.parameters.get("compare_columns", []),
            "groups": len(candidate.parameters.get("removed_to_kept", {})),
            "kept_rows": len(candidate.parameters.get("kept_row_ids", [])),
        }
    return context


def _candidate_option_text(context: dict[str, Any]) -> str:
    operation = context["operation"]
    column = context.get("column")
    target_count = context.get("target_count")
    samples = context.get("samples", [])
    sample_text = "; ".join(
        f"{sample['row_id']}.{sample['column']}: {sample['before']!r} -> {sample['after']!r}"
        for sample in samples[:3]
    )
    if operation == Operation.REMOVE_EXACT_DUPLICATES:
        return (
            f"Remove {target_count} redundant rows that exactly match a kept row on the "
            f"compare columns {context.get('duplicate_policy', {}).get('compare_columns', [])}. "
            f"Examples: {sample_text}."
        )
    return (
        f"Apply {operation} to {target_count} rows in column {column!r} "
        f"(samples: {sample_text})."
    )


def build_questions(role: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the fixed three questions per candidate for one batch."""
    questions: dict[str, Any] = {}
    for index, context in enumerate(contexts):
        candidate_id = context["candidate_id"]
        questions[f"c{index}_action"] = Choice(
            instructions=_ACTION_INSTRUCTIONS[role].format(index=index),
            criteria={
                candidate_id: _candidate_option_text(context),
                KEEP_ORIGINAL: "Retain the input values when the proposal is unsuitable or unclear.",
            },
        )
        questions[f"c{index}_applicable"] = Noul(
            instructions=_APPLICABLE_INSTRUCTIONS[role].format(index=index)
        )
        questions[f"c{index}_risk"] = Score(
            instructions=_RISK_INSTRUCTIONS[role].format(index=index),
            criteria=list(RISK_LEVEL_CRITERIA),
        )
    return questions


def build_postcheck_contexts(
    state_before: DatasetState,
    config: TableConfig,
    plan: RepairPlan,
    changes: list[CellChange],
    candidates: dict[str, RepairCandidate],
    *,
    sample_limit: int = MAX_SAMPLES_PER_CANDIDATE,
) -> list[dict[str, Any]]:
    """Bounded contexts describing the actual executed diff per candidate."""
    contexts: list[dict[str, Any]] = []
    for candidate_id in plan.candidate_ids:
        candidate = candidates.get(candidate_id)
        candidate_changes = [change for change in changes if change.candidate_id == candidate_id]
        column_name = None
        operation = None
        if candidate is not None:
            operation = candidate.operation.value
            selection = state_before.selection(candidate.selection_id)
            column_name = candidate.parameters.get("column") or (selection.column if selection else None)
        column_config = config.columns.get(column_name) if column_name else None
        samples = [
            {
                "row_id": change.row_id,
                "column": change.column,
                "before": truncate_cell(change.before),
                "after": truncate_cell(change.after),
            }
            for change in candidate_changes[:sample_limit]
        ]
        contexts.append(
            {
                "candidate_id": candidate_id,
                "role": "semantic_postcheck",
                "operation": operation,
                "column": column_name,
                "column_description": column_config.description if column_config else None,
                "table_description": config.table_description,
                "record_grain": config.record_grain,
                "applied_change_count": len(candidate_changes),
                "actual_diff_samples": samples,
            }
        )
    return contexts


def build_postcheck_questions(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for index, _ in enumerate(contexts):
        questions[f"c{index}_postcheck"] = Noul(
            instructions=(
                f"Does the operation actually applied for `state.candidates[{index}]` "
                "violate the business meaning described for its column or the stated "
                "cleaning policy? A true answer means a semantic error exists."
            ),
            criteria={
                "true": "The applied operation violates the described business meaning or policy.",
                "false": "The applied operation is consistent with the described business meaning and policy.",
            },
        )
    return questions


def serialize_questions(questions: dict[str, Any]) -> dict[str, Any]:
    """Convert SDK question objects to plain JSON for the request log."""
    payload: dict[str, Any] = {}
    for name, question in questions.items():
        if hasattr(question, "model_dump"):
            payload[name] = question.model_dump(mode="json", exclude_none=True)
        else:
            payload[name] = question
    return payload
