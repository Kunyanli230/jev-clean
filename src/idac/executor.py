"""Deterministic executor: apply a plan to a candidate copy and record the real diff.

The executor never runs model-returned code, field paths or arbitrary
parameters. Every change is derived from the plan's stored ``after`` values and
the current snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import TableConfig
from .models import (
    CellChange,
    DatasetState,
    Operation,
    RemovedRow,
    RepairPlan,
)


class ExecutionError(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    state: DatasetState
    changes: list[CellChange] = field(default_factory=list)
    removed_rows: list[RemovedRow] = field(default_factory=list)

    @property
    def planned_changes(self) -> list[CellChange]:
        return self.changes


def execute(
    state: DatasetState,
    plan: RepairPlan,
    config: TableConfig,
    new_version_id: str,
) -> ExecutionResult:
    if plan.version_id != state.version_id:
        raise ExecutionError(
            f"plan {plan.id} targets {plan.version_id}, current snapshot is {state.version_id}"
        )
    rows = [list(row) for row in state.rows]
    row_ids = list(state.row_ids)
    row_index = state.row_index()
    column_index = state.column_index()
    changes: list[CellChange] = []
    new_removed: list[RemovedRow] = []
    removed_ids: set[str] = set()

    for operation in plan.ordered_operations:
        selection = state.selection(operation.selection_id)
        if selection is None:
            raise ExecutionError(f"plan references unknown selection {operation.selection_id}")
        if set(selection.row_ids) - set(row_index):
            raise ExecutionError(f"selection {selection.id} contains rows outside the snapshot")
        if operation.operation == Operation.REMOVE_EXACT_DUPLICATES:
            planned = list(operation.removed_row_ids)
            if set(planned) != set(selection.row_ids):
                raise ExecutionError(
                    f"plan removal set does not match selection {selection.id}"
                )
            for row_id in planned:
                position = row_index[row_id]
                if row_id in removed_ids:
                    continue
                removed_ids.add(row_id)
                new_removed.append(
                    RemovedRow(
                        row_id=row_id,
                        values=list(state.rows[position]),
                        reason="exact_duplicate",
                        candidate_id=operation.candidate_ids[0] if operation.candidate_ids else None,
                    )
                )
            continue
        if operation.column is None:
            raise ExecutionError(f"cell operation {operation.operation.value} has no column")
        column = operation.column
        if config.columns[column].protected:
            raise ExecutionError(f"plan attempts to modify protected column {column!r}")
        for row_id, after in operation.after_values.items():
            if row_id not in row_index:
                raise ExecutionError(f"plan targets unknown row {row_id}")
            position = row_index[row_id]
            before = rows[position][column_index[column]]
            if before == after:
                raise ExecutionError(
                    f"plan expects a change for {row_id}.{column} but the value already equals {after!r}"
                )
            rows[position][column_index[column]] = after
            changes.append(
                CellChange(
                    row_id=row_id,
                    column=column,
                    before=before,
                    after=after,
                    candidate_id=operation.candidate_ids[0] if operation.candidate_ids else None,
                )
            )

    if removed_ids:
        kept_rows = [
            row for row_id, row in zip(row_ids, rows, strict=True) if row_id not in removed_ids
        ]
        kept_ids = [row_id for row_id in row_ids if row_id not in removed_ids]
    else:
        kept_rows = rows
        kept_ids = row_ids

    new_state = DatasetState(
        version_id=new_version_id,
        parent_version_id=state.version_id,
        input_hash=state.input_hash,
        ordered_columns=list(state.ordered_columns),
        row_ids=kept_ids,
        rows=kept_rows,
        schema=dict(state.schema),
        removed_rows=[*state.removed_rows, *new_removed],
        issues=[issue.model_copy() for issue in state.issues],
        selections=dict(state.selections),
    )
    return ExecutionResult(state=new_state, changes=changes, removed_rows=new_removed)
