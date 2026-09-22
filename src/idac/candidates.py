"""Deterministic candidate builders.

Candidates carry values computed locally by code; the model only chooses whether
to apply them. A candidate represents one group of target rows in the same
snapshot, sub-phase, column, operation and parameters. The exact ``after``
values are stored with the candidate so the planner and executor never have to
trust model output for values.
"""

from __future__ import annotations

from .config import TableConfig
from .models import (
    CellPreview,
    CellValue,
    DatasetState,
    Operation,
    RepairCandidate,
    fingerprint,
)
from .profiling import PhaseDetection
from .settings import MAX_SAMPLES_PER_CANDIDATE


def build_candidates(
    state: DatasetState, config: TableConfig, detection: PhaseDetection
) -> list[RepairCandidate]:
    """Turn phase detection data into bounded, fully-computed candidates."""
    candidates: list[RepairCandidate] = []
    for data in detection.candidates_data:
        operation: Operation = data["operation"]
        selection = data["selection"]
        column_name: str | None = data["column"]
        parameters: dict = dict(data["parameters"])
        targets: list[tuple[str, str | None]] = list(data["targets"])
        if operation == Operation.REMOVE_EXACT_DUPLICATES:
            if not config.duplicates.enabled:
                continue
            after_values: dict[str, CellValue] = {}
        else:
            if column_name is None or not config.columns[column_name].allows(operation):
                continue
            after_values = _compute_after_values(operation, column_name, parameters, targets, data)
            if not after_values:
                continue
        parameters["after_values"] = after_values
        preview = _build_preview(
            state, operation, column_name, parameters, targets, after_values, data
        )
        if not preview:
            continue
        payload = {
            "version_id": state.version_id,
            "operation": operation.value,
            "selection_id": selection.id,
            "parameters": parameters,
            "issue_ids": data["issue_ids"],
        }
        candidates.append(
            RepairCandidate(
                id="cand-" + fingerprint(payload)[:16],
                version_id=state.version_id,
                issue_ids=data["issue_ids"],
                operation=operation,
                selection_id=selection.id,
                parameters=parameters,
                preview=preview,
                evidence_ids=data["evidence_ids"],
                fingerprint=fingerprint(payload),
            )
        )
    return candidates


def _compute_after_values(
    operation: Operation,
    column_name: str,
    parameters: dict,
    targets: list[tuple[str, str | None]],
    data: dict,
) -> dict[str, CellValue]:
    if operation == Operation.CAST_NUMERIC or operation == Operation.PARSE_DATETIME:
        canonical: dict[str, str] = data.get("canonical", {})
        return {row_id: canonical[row_id] for row_id, _ in targets if row_id in canonical}
    if operation == Operation.TRIM_WHITESPACE:
        return {
            row_id: raw.strip()
            for row_id, raw in targets
            if raw is not None and raw.strip() != raw
        }
    if operation in (Operation.NORMALIZE_NULL_TOKENS, Operation.INVALIDATE_OUT_OF_RANGE):
        return {row_id: None for row_id, raw in targets if raw is not None}
    if operation == Operation.IMPUTE_MISSING:
        fill_value = parameters.get("fill_value")
        if fill_value is None:
            return {}
        return {row_id: fill_value for row_id, _ in targets}
    return {}


def _build_preview(
    state: DatasetState,
    operation: Operation,
    column_name: str | None,
    parameters: dict,
    targets: list[tuple[str, str | None]],
    after_values: dict[str, CellValue],
    data: dict,
) -> list[CellPreview]:
    sample = targets[:MAX_SAMPLES_PER_CANDIDATE]
    if operation == Operation.REMOVE_EXACT_DUPLICATES:
        compare_columns = list(parameters.get("compare_columns", []))
        compare_column = compare_columns[0] if compare_columns else None
        removed_to_kept: dict[str, str] = dict(parameters.get("removed_to_kept", {}))
        index = state.column_index()
        row_lookup = {row_id: row for row, row_id in enumerate(state.row_ids)}
        preview: list[CellPreview] = []
        for row_id, _ in sample:
            kept_row = removed_to_kept.get(row_id)
            if compare_column is not None and kept_row is not None:
                column = index[compare_column]
                preview.append(
                    CellPreview(
                        row_id=row_id,
                        column=compare_column,
                        before=state.rows[row_lookup[row_id]][column],
                        after=state.rows[row_lookup[kept_row]][column],
                    )
                )
            else:
                preview.append(CellPreview(row_id=row_id, column="*", before=row_id, after=None))
        return preview
    if column_name is None:
        return []
    return [
        CellPreview(row_id=row_id, column=column_name, before=raw, after=after_values[row_id])
        for row_id, raw in sample
        if row_id in after_values and after_values[row_id] != raw
    ]
