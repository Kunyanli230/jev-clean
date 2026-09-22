"""Two-layer validation: deterministic hard checks, then semantic Noul checks.

Hard validation inspects every row of the candidate copy and never sends a
semantic request when it fails. The semantic verifier reads only the actual
diff produced by the executor.
"""

from __future__ import annotations

from .config import NUMERIC_TYPES, TableConfig
from .models import (
    CellChange,
    CheckStatus,
    ClientOutcome,
    DatasetState,
    GateCheck,
    Issue,
    IssueCategory,
    ModelStatus,
    Operation,
    Phase,
    ReasonCode,
    RepairPlan,
    Selection,
    SemanticCheck,
    ValidationResult,
)
from .profiling import (
    ValueParseError,
    detect_phase,
    format_date,
    format_numeric,
    numeric_in_range,
    parse_date,
    parse_numeric,
)
from .settings import MAX_POSTCHECK_ERROR_PROBABILITY


class HardValidationError(RuntimeError):
    pass


def _check(
    rule_id: str,
    metric: str,
    observed,
    operator: str,
    threshold,
    passed: bool,
    reason_code: ReasonCode | None = None,
) -> GateCheck:
    return GateCheck(
        rule_id=rule_id,
        metric=metric,
        observed_value=observed if isinstance(observed, (int, float, str)) or observed is None else str(observed),
        operator=operator,
        threshold_or_required_value=threshold,
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        reason_code=None if passed else reason_code,
    )


def _planned_changes(state_before: DatasetState, plan: RepairPlan) -> list[CellChange]:
    changes: list[CellChange] = []
    index = state_before.column_index()
    rows = state_before.row_index()
    for operation in plan.ordered_operations:
        if operation.operation == Operation.REMOVE_EXACT_DUPLICATES or operation.column is None:
            continue
        for row_id, after in operation.after_values.items():
            before = state_before.rows[rows[row_id]][index[operation.column]]
            changes.append(
                CellChange(
                    row_id=row_id,
                    column=operation.column,
                    before=before,
                    after=after,
                    candidate_id=operation.candidate_ids[0] if operation.candidate_ids else None,
                )
            )
    return changes


def validate_hard(
    state_before: DatasetState,
    execution,
    plan: RepairPlan,
    config: TableConfig,
) -> ValidationResult:
    state_after: DatasetState = execution.state
    checks: list[GateCheck] = []
    reasons: list[ReasonCode] = []

    def record(check: GateCheck) -> None:
        checks.append(check)
        if check.status == CheckStatus.FAILED and check.reason_code is not None:
            reasons.append(check.reason_code)

    version_ok = plan.version_id == state_before.version_id and state_after.parent_version_id == state_before.version_id
    record(
        _check(
            "PLAN_VERSION_BINDING",
            "plan.version_id",
            plan.version_id,
            "==",
            state_before.version_id,
            version_ok,
            ReasonCode.STALE_VERSION,
        )
    )

    planned = {(c.row_id, c.column, c.before, c.after) for c in _planned_changes(state_before, plan)}
    logged = {(c.row_id, c.column, c.before, c.after) for c in execution.changes}
    before_index = state_before.row_index()
    actual = set()
    shape_ok = (
        state_after.ordered_columns == state_before.ordered_columns
        and len(state_after.rows) == len(state_after.row_ids)
        and all(len(row) == len(state_before.ordered_columns) for row in state_after.rows)
    )
    if not shape_ok:
        return ValidationResult(
            plan_id=plan.id, version_id=state_before.version_id,
            hard_checks=[_check("SNAPSHOT_SHAPE", "shape", "invalid", "==", "valid",
                                False, ReasonCode.HARD_VALIDATION_FAILED)],
            semantic_checks=[], passed=False,
            reason_codes=[ReasonCode.HARD_VALIDATION_FAILED],
        )
    for row_id, row in zip(state_after.row_ids, state_after.rows, strict=True):
        if row_id not in before_index:
            continue
        before = state_before.rows[before_index[row_id]]
        for index, column in enumerate(state_before.ordered_columns):
            if before[index] != row[index]:
                actual.add((row_id, column, before[index], row[index]))
    diff_ok = planned == actual == logged
    record(
        _check(
            "PLAN_DIFF_MATCH",
            "changed_cells",
            len(actual),
            "== planned",
            len(planned),
            diff_ok,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )
    record(
        _check(
            "NO_EXTRA_CHANGES",
            "unplanned_changes",
            len(actual - planned),
            "==",
            0,
            not (actual - planned),
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )

    protected_ok = True
    before_rows = state_before.row_index()
    for row_id, row in zip(state_after.row_ids, state_after.rows, strict=True):
        position = before_rows.get(row_id)
        if position is None:
            continue
        for column_name, column in config.columns.items():
            if not column.protected:
                continue
            if row[state_after.column_index()[column_name]] != state_before.rows[position][
                state_before.column_index()[column_name]
            ]:
                protected_ok = False
    record(
        _check(
            "PROTECTED_CELLS_UNCHANGED",
            "protected_cell_changes",
            0 if protected_ok else 1,
            "==",
            0,
            protected_ok,
            ReasonCode.PROTECTED_COLUMN,
        )
    )

    columns_ok = state_after.ordered_columns == state_before.ordered_columns
    record(
        _check(
            "COLUMNS_STABLE",
            "ordered_columns",
            len(state_after.ordered_columns),
            "== preserved",
            len(state_before.ordered_columns),
            columns_ok,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )

    unique_ids = len(set(state_after.row_ids)) == len(state_after.row_ids)
    record(
        _check(
            "ROW_IDS_UNIQUE",
            "duplicate_row_ids",
            len(state_after.row_ids) - len(set(state_after.row_ids)),
            "==",
            0,
            unique_ids,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )

    order_ok = _is_subsequence(state_after.row_ids, state_before.row_ids)
    record(
        _check(
            "RETAINED_ROW_ORDER",
            "row_order",
            "preserved" if order_ok else "changed",
            "==",
            "preserved",
            order_ok,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )

    planned_removed = {
        row_id
        for operation in plan.ordered_operations
        if operation.operation == Operation.REMOVE_EXACT_DUPLICATES
        for row_id in operation.removed_row_ids
    }
    recorded_removed = {removed.row_id for removed in execution.removed_rows}
    actual_removed = set(before_rows) - set(state_after.row_ids)
    removed_ok = (
        planned_removed == recorded_removed == actual_removed
        and planned_removed.issubset(before_rows)
    )
    record(
        _check(
            "REMOVED_ROWS_MATCH",
            "removed_rows",
            len(recorded_removed),
            "== planned",
            len(planned_removed),
            removed_ok,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )
    conservation_ok = len(state_after.row_ids) == len(state_before.row_ids) - len(planned_removed)
    record(
        _check(
            "ROW_COUNT_CONSERVATION",
            "row_count",
            len(state_after.row_ids),
            "==",
            len(state_before.row_ids) - len(planned_removed),
            conservation_ok,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )

    kept_ok, kept_reason = _check_duplicate_kept(state_before, plan, config)
    record(
        _check(
            "DUPLICATE_KEPT_ROWS",
            "duplicate_kept_errors",
            0 if kept_ok else 1,
            "==",
            0,
            kept_ok,
            kept_reason,
        )
    )

    allowed_null_ops = {
        Operation.NORMALIZE_NULL_TOKENS,
        Operation.INVALIDATE_OUT_OF_RANGE,
    }
    new_nulls = {
        (change.row_id, change.column)
        for change in execution.changes
        if change.before is not None and change.after is None
    }
    allowed_nulls = {
        (row_id, operation.column)
        for operation in plan.ordered_operations
        if operation.operation in allowed_null_ops and operation.column is not None
        for row_id in operation.after_values
    }
    nulls_ok = new_nulls.issubset(allowed_nulls)
    record(
        _check(
            "NO_NEW_NULLS",
            "unexpected_new_nulls",
            len(new_nulls - allowed_nulls),
            "==",
            0,
            nulls_ok,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )

    conversions_ok, conversions_reason = _check_conversions(state_before, execution, plan, config)
    record(
        _check(
            "CONVERSION_CONFORMANCE",
            "conversion_errors",
            0 if conversions_ok else 1,
            "==",
            0,
            conversions_ok,
            conversions_reason,
        )
    )

    impute_ok, impute_reason = _check_imputation(state_before, execution, plan, config)
    record(
        _check(
            "IMPUTE_CORRECTNESS",
            "imputation_errors",
            0 if impute_ok else 1,
            "==",
            0,
            impute_ok,
            impute_reason,
        )
    )

    iqr_selection_ids = {
        issue.selection_id
        for issue in state_before.issues
        if issue.category == IssueCategory.IQR_OUTLIER
    }
    iqr_ok = all(
        operation.selection_id not in iqr_selection_ids
        for operation in plan.ordered_operations
    )
    record(
        _check(
            "IQR_ONLY_NO_REWRITE",
            "iqr_issue_operations",
            0,
            "==",
            0,
            iqr_ok,
            ReasonCode.HARD_VALIDATION_FAILED,
        )
    )

    passed = all(check.status == CheckStatus.PASSED for check in checks)
    return ValidationResult(
        plan_id=plan.id,
        version_id=state_before.version_id,
        hard_checks=checks,
        semantic_checks=[],
        passed=passed,
        reason_codes=reasons,
    )


def _check_duplicate_kept(
    state_before: DatasetState, plan: RepairPlan, config: TableConfig
) -> tuple[bool, ReasonCode]:
    """Every removed duplicate row must leave the earliest row of its group retained."""
    from .profiling import canonical_key

    if not config.duplicates.enabled:
        return True, ReasonCode.HARD_VALIDATION_FAILED
    compare_columns = config.duplicates.compare_columns
    index = state_before.column_index()
    order = state_before.row_index()
    groups: dict[tuple, list[str]] = {}
    for row_id, row in zip(state_before.row_ids, state_before.rows, strict=True):
        key = tuple(
            canonical_key(row[index[column]], config.columns[column]) for column in compare_columns
        )
        groups.setdefault(key, []).append(row_id)
    for operation in plan.ordered_operations:
        if operation.operation is not Operation.REMOVE_EXACT_DUPLICATES:
            continue
        removed = set(operation.removed_row_ids)
        for row_id in removed:
            position = order[row_id]
            row = state_before.rows[position]
            key = tuple(
                canonical_key(row[index[column]], config.columns[column])
                for column in compare_columns
            )
            earliest = min(groups[key], key=lambda item: order[item])
            if earliest in removed:
                return False, ReasonCode.HARD_VALIDATION_FAILED
    return True, ReasonCode.HARD_VALIDATION_FAILED


def _is_subsequence(after: list[str], before: list[str]) -> bool:
    iterator = iter(before)
    return all(row_id in iterator for row_id in after)


def _check_conversions(
    state_before: DatasetState, execution, plan: RepairPlan, config: TableConfig
) -> tuple[bool, ReasonCode]:
    for operation in plan.ordered_operations:
        if operation.column is None or operation.operation == Operation.REMOVE_EXACT_DUPLICATES:
            continue
        column = config.columns[operation.column]
        for after in operation.after_values.values():
            if after is None:
                continue
            if operation.operation == Operation.CAST_NUMERIC:
                try:
                    if format_numeric(parse_numeric(after, column), column) != after:
                        return False, ReasonCode.HARD_VALIDATION_FAILED
                    if not numeric_in_range(parse_numeric(after, column), column):
                        return False, ReasonCode.HARD_VALIDATION_FAILED
                except ValueParseError:
                    return False, ReasonCode.HARD_VALIDATION_FAILED
            elif operation.operation == Operation.PARSE_DATETIME:
                try:
                    if format_date(parse_date(after, column)) != after:
                        return False, ReasonCode.HARD_VALIDATION_FAILED
                except ValueParseError:
                    return False, ReasonCode.HARD_VALIDATION_FAILED
            elif operation.operation == Operation.IMPUTE_MISSING:
                if column.type == "categorical":
                    if after not in column.allowed_values:
                        return False, ReasonCode.HARD_VALIDATION_FAILED
                elif column.type in NUMERIC_TYPES:
                    try:
                        if not numeric_in_range(parse_numeric(after, column), column):
                            return False, ReasonCode.HARD_VALIDATION_FAILED
                    except ValueParseError:
                        return False, ReasonCode.HARD_VALIDATION_FAILED
    for change in execution.changes:
        if change.after is None:
            continue
        column = config.columns[change.column]
        if column.type == "categorical" and change.after not in column.allowed_values:
            return False, ReasonCode.HARD_VALIDATION_FAILED
    return True, ReasonCode.HARD_VALIDATION_FAILED


def _check_imputation(
    state_before: DatasetState, execution, plan: RepairPlan, config: TableConfig
) -> tuple[bool, ReasonCode]:

    for operation in plan.ordered_operations:
        if operation.operation != Operation.IMPUTE_MISSING or operation.column is None:
            continue
        column = config.columns[operation.column]
        selection = state_before.selection(operation.selection_id)
        if selection is None:
            return False, ReasonCode.HARD_VALIDATION_FAILED
        index = state_before.column_index()[operation.column]
        targets = list(selection.row_ids)
        if set(targets) != set(operation.after_values):
            return False, ReasonCode.HARD_VALIDATION_FAILED
        for row_id in targets:
            if state_before.rows[state_before.row_index()[row_id]][index] is not None:
                return False, ReasonCode.HARD_VALIDATION_FAILED
        expected = _donor_value(state_before, operation.column, config)
        if expected is None:
            return False, ReasonCode.HARD_VALIDATION_FAILED
        if any(value != expected for value in operation.after_values.values()):
            return False, ReasonCode.HARD_VALIDATION_FAILED
        for after in operation.after_values.values():
            if after is None:
                return False, ReasonCode.HARD_VALIDATION_FAILED
            if column.type == "categorical":
                if after not in column.allowed_values:
                    return False, ReasonCode.HARD_VALIDATION_FAILED
            else:
                try:
                    if not numeric_in_range(parse_numeric(after, column), column):
                        return False, ReasonCode.HARD_VALIDATION_FAILED
                except ValueParseError:
                    return False, ReasonCode.HARD_VALIDATION_FAILED
    return True, ReasonCode.HARD_VALIDATION_FAILED


def _donor_value(state: DatasetState, column_name: str, config: TableConfig) -> str | None:
    from .profiling import median_decimal, mode_value

    column = config.columns[column_name]
    index = state.column_index()[column_name]
    if column.type == "categorical":
        donors = [
            value
            for value in (state.rows[row][index] for row in range(len(state.row_ids)))
            if value is not None and value in column.allowed_values
        ]
        return mode_value(donors) if donors else None
    values = []
    for row in range(len(state.row_ids)):
        value = state.rows[row][index]
        if value is None:
            continue
        try:
            parsed = parse_numeric(value, column)
        except ValueParseError:
            continue
        if numeric_in_range(parsed, column):
            values.append(parsed)
    if not values:
        return None
    return format_numeric(median_decimal(values, column), column)


def confirmed_repaired(
    state_after: DatasetState,
    config: TableConfig,
    phase: Phase,
    issues: list[Issue],
    selections: dict[str, Selection],
) -> set[str]:
    """Issues are repaired only when a re-check confirms they disappeared."""
    detection = detect_phase(state_after, config, phase)
    detected_by_category: dict[IssueCategory, set[str]] = {}
    for issue in detection.issues:
        selection = detection.selections.get(issue.selection_id or "")
        if selection is None:
            selection = state_after.selection(issue.selection_id or "")
        rows = set(selection.row_ids) if selection else set()
        detected_by_category.setdefault(issue.category, set()).update(rows)
    repaired: set[str] = set()
    for issue in issues:
        selection = selections.get(issue.selection_id or "") or state_after.selection(
            issue.selection_id or ""
        )
        rows = set(selection.row_ids) if selection else set()
        if not (detected_by_category.get(issue.category, set()) & rows):
            repaired.add(issue.id)
    return repaired


def validate_semantic(
    plan: RepairPlan,
    outcome: ClientOutcome,
    threshold: float = MAX_POSTCHECK_ERROR_PROBABILITY,
) -> list[SemanticCheck]:
    """Evaluate the per-candidate postcheck Noul answers against the threshold."""
    checks: list[SemanticCheck] = []
    answers = outcome.answers or {}
    for index, candidate_id in enumerate(plan.candidate_ids):
        question_name = f"c{index}_postcheck"
        answer = answers.get(question_name)
        if outcome.model_status != ModelStatus.OK or answer is None:
            checks.append(
                SemanticCheck(
                    candidate_id=candidate_id,
                    probability=1.0,
                    threshold=threshold,
                    request_id=outcome.request_id,
                    model=outcome.model,
                    passed=False,
                    reason_code=ReasonCode.MISSING_ANSWER,
                )
            )
            continue
        probability = getattr(answer, "noul", None)
        if (
            probability is None
            or not isinstance(probability, (int, float))
            or not (0.0 <= float(probability) <= 1.0)
        ):
            checks.append(
                SemanticCheck(
                    candidate_id=candidate_id,
                    probability=1.0,
                    threshold=threshold,
                    request_id=outcome.request_id,
                    model=outcome.model,
                    passed=False,
                    reason_code=ReasonCode.INVALID_DISTRIBUTION,
                )
            )
            continue
        passed = float(probability) <= threshold
        checks.append(
            SemanticCheck(
                candidate_id=candidate_id,
                probability=float(probability),
                threshold=threshold,
                request_id=outcome.request_id,
                model=outcome.model,
                passed=passed,
                reason_code=None if passed else ReasonCode.POSTCHECK_ABOVE_THRESHOLD,
            )
        )
    return checks
