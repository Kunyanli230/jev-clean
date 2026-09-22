"""Deterministic repair planner.

Only candidates that passed the decision gate can enter a plan. The planner
re-checks eligibility against the current snapshot, merges identical
suggestions, discards conflicting ones and never executes anything the
configuration did not authorize.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TableConfig
from .models import (
    CellValue,
    DatasetState,
    Evidence,
    Operation,
    PlannedOperation,
    ReasonCode,
    RepairCandidate,
    RepairPlan,
)


@dataclass
class AcceptedCandidate:
    candidate: RepairCandidate
    decision_id: str


def eligibility_reasons(
    state: DatasetState,
    config: TableConfig,
    evidence_map: dict[str, Evidence],
    candidate: RepairCandidate,
) -> list[ReasonCode]:
    reasons: list[ReasonCode] = []
    if candidate.version_id != state.version_id:
        reasons.append(ReasonCode.STALE_VERSION)
    selection = state.selection(candidate.selection_id)
    if selection is None or selection.version_id != state.version_id:
        reasons.append(ReasonCode.STALE_VERSION)
    known_evidence = all(
        evidence_id in evidence_map and evidence_map[evidence_id].version_id == state.version_id
        for evidence_id in candidate.evidence_ids
    )
    if not known_evidence:
        reasons.append(ReasonCode.STALE_VERSION)
    if candidate.operation == Operation.REMOVE_EXACT_DUPLICATES:
        if not config.duplicates.enabled:
            reasons.append(ReasonCode.OPERATION_NOT_ALLOWED)
        if selection is not None:
            allowed = set(selection.row_ids)
            planned = set(candidate.parameters.get("removed_to_kept", {}))
            if not planned or not planned.issubset(allowed):
                reasons.append(ReasonCode.CANDIDATE_UNAVAILABLE)
    else:
        column = candidate.parameters.get("column") or (selection.column if selection else None)
        if column is None or column not in config.columns:
            reasons.append(ReasonCode.UNSUPPORTED_COLUMN)
        elif not config.columns[column].allows(candidate.operation):
            reasons.append(
                ReasonCode.PROTECTED_COLUMN
                if config.columns[column].protected
                else ReasonCode.OPERATION_NOT_ALLOWED
            )
        if selection is not None and candidate.parameters.get("after_values") is None:
            reasons.append(ReasonCode.CANDIDATE_UNAVAILABLE)
    return reasons


def _proposal(
    state: DatasetState,
    config: TableConfig,
    accepted: AcceptedCandidate,
) -> PlannedOperation | None:
    candidate = accepted.candidate
    selection = state.selection(candidate.selection_id)
    if selection is None:
        return None
    if candidate.operation == Operation.REMOVE_EXACT_DUPLICATES:
        removed = [row_id for row_id in selection.row_ids if row_id in candidate.parameters.get("removed_to_kept", {})]
        if not removed:
            return None
        return PlannedOperation(
            operation=candidate.operation,
            candidate_ids=[candidate.id],
            selection_id=selection.id,
            column=None,
            parameters={"compare_columns": candidate.parameters.get("compare_columns", [])},
            removed_row_ids=removed,
        )
    column = candidate.parameters.get("column") or selection.column
    if column is None:
        return None
    after_values: dict[str, CellValue] = dict(candidate.parameters.get("after_values", {}))
    index = state.column_index()[column]
    row_index = state.row_index()
    effective: dict[str, CellValue] = {}
    for row_id in selection.row_ids:
        if row_id not in after_values:
            continue
        before = state.rows[row_index[row_id]][index]
        after = after_values[row_id]
        if before != after:
            effective[row_id] = after
    if not effective:
        return None
    return PlannedOperation(
        operation=candidate.operation,
        candidate_ids=[candidate.id],
        selection_id=selection.id,
        column=column,
        parameters={
            "method": candidate.parameters.get("method"),
            "fill_value": candidate.parameters.get("fill_value"),
            "donor_count": candidate.parameters.get("donor_count"),
        },
        after_values=effective,
    )


def _merge(proposals: list[PlannedOperation]) -> tuple[list[PlannedOperation], list[dict]]:
    """Merge identical proposals and discard conflicting cell assignments."""
    merged: list[PlannedOperation] = []
    discarded: list[dict] = []
    for proposal in proposals:
        if proposal.operation == Operation.REMOVE_EXACT_DUPLICATES:
            key = ("duplicates", tuple(sorted(proposal.removed_row_ids)))
            for existing in merged:
                if existing.operation == proposal.operation and (
                    "duplicates",
                    tuple(sorted(existing.removed_row_ids)),
                ) == key:
                    existing.candidate_ids.extend(
                        cid for cid in proposal.candidate_ids if cid not in existing.candidate_ids
                    )
                    break
            else:
                merged.append(proposal)
            continue
        key = (
            proposal.operation.value,
            proposal.column,
            tuple(sorted((row_id, after) for row_id, after in proposal.after_values.items())),
        )
        for existing in merged:
            existing_key = (
                existing.operation.value,
                existing.column,
                tuple(sorted((row_id, after) for row_id, after in existing.after_values.items())),
            )
            if existing_key == key:
                existing.candidate_ids.extend(
                    cid for cid in proposal.candidate_ids if cid not in existing.candidate_ids
                )
                break
        else:
            merged.append(proposal)
    cell_owners: dict[tuple[str, str], list[tuple[int, CellValue]]] = {}
    for index, proposal in enumerate(merged):
        if proposal.operation == Operation.REMOVE_EXACT_DUPLICATES or proposal.column is None:
            continue
        for row_id, after in proposal.after_values.items():
            cell_owners.setdefault((row_id, proposal.column), []).append((index, after))
    conflicting: set[int] = set()
    for (row_id, column), owners in cell_owners.items():
        if len({after for _, after in owners}) > 1:
            for index, _ in owners:
                conflicting.add(index)
            discarded.append(
                {
                    "candidate_ids": [cid for index in {i for i, _ in owners} for cid in merged[index].candidate_ids],
                    "reason_code": ReasonCode.CANDIDATE_CONFLICT.value,
                    "detail": {"row_id": row_id, "column": column},
                }
            )
    final = [proposal for index, proposal in enumerate(merged) if index not in conflicting]
    return final, discarded


def build_plan(
    state: DatasetState,
    config: TableConfig,
    evidence_map: dict[str, Evidence],
    accepted: list[AcceptedCandidate],
    plan_id: str,
) -> RepairPlan:
    """Build a version-bound plan from gate-accepted candidates."""
    discarded: list[dict] = []
    proposals: list[PlannedOperation] = []
    for entry in accepted:
        reasons = eligibility_reasons(state, config, evidence_map, entry.candidate)
        if reasons:
            discarded.append(
                {
                    "candidate_ids": [entry.candidate.id],
                    "reason_code": reasons[0].value,
                    "detail": {"reasons": [reason.value for reason in reasons]},
                }
            )
            continue
        proposal = _proposal(state, config, entry)
        if proposal is None:
            discarded.append(
                {
                    "candidate_ids": [entry.candidate.id],
                    "reason_code": ReasonCode.NO_CHANGE.value,
                    "detail": {},
                }
            )
            continue
        proposals.append(proposal)
    ordered, merge_discards = _merge(proposals)
    discarded.extend(merge_discards)
    candidate_ids = [cid for proposal in ordered for cid in proposal.candidate_ids]
    return RepairPlan(
        id=plan_id,
        version_id=state.version_id,
        candidate_ids=candidate_ids,
        ordered_operations=ordered,
        discarded=discarded,
    )
