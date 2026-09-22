"""Deterministic profiling, constraint checks and value parsing.

Every calculation that has an exact answer (counts, parses, ranges, grouping)
happens here in Python. The model is never asked whether a value parses or
whether two rows are byte-identical. Detection only describes facts; candidate
values are computed by :mod:`idac.candidates`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import pandas as pd

from .config import NUMERIC_TYPES, ColumnConfig, TableConfig
from .models import (
    DatasetState,
    Evidence,
    Issue,
    IssueCategory,
    IssueStatus,
    Operation,
    Phase,
    ReasonCode,
    Selection,
    fingerprint,
)

PARSE_CATEGORIES = {
    IssueCategory.NUMERIC_FORMAT,
    IssueCategory.DATE_FORMAT,
    IssueCategory.PARSE_FAILURE,
    IssueCategory.AMBIGUOUS_PARSE,
}


class ValueParseError(ValueError):
    """The raw value does not follow the configured format."""


class AmbiguousValueError(ValueParseError):
    """The raw value parses to more than one distinct logical value."""


def _strip(text: str) -> str:
    return text.strip()


def parse_numeric(raw: str, column: ColumnConfig) -> Decimal:
    """Parse a numeric cell strictly against the configured decimal/thousands format."""
    text = _strip(raw)
    if not text:
        raise ValueParseError("empty numeric value")
    sign = ""
    if text[0] in "+-":
        sign, text = text[0], text[1:]
    decimal_separator = column.decimal_separator
    if decimal_separator in text:
        integer_part, _, fraction_part = text.partition(decimal_separator)
        if decimal_separator in fraction_part:
            raise ValueParseError(f"multiple decimal separators in {raw!r}")
    else:
        integer_part, fraction_part = text, ""
    separator = column.thousands_separator
    if separator is not None:
        groups = integer_part.split(separator)
        if len(groups) > 1 and not (
            1 <= len(groups[0]) <= 3 and all(len(group) == 3 for group in groups[1:])
        ):
            raise ValueParseError(f"invalid thousands grouping in {raw!r}")
        integer_part = "".join(groups)
    if integer_part and not integer_part.isdigit():
        raise ValueParseError(f"invalid numeric digits in {raw!r}")
    if fraction_part and not fraction_part.isdigit():
        raise ValueParseError(f"invalid fractional digits in {raw!r}")
    if not integer_part and not fraction_part:
        raise ValueParseError(f"no digits in {raw!r}")
    try:
        value = Decimal(f"{sign}{integer_part or '0'}.{fraction_part or '0'}")
    except InvalidOperation as error:
        raise ValueParseError(f"invalid decimal value {raw!r}") from error
    if not value.is_finite():
        raise ValueParseError(f"non-finite decimal value {raw!r}")
    if column.type == "integer" and value != value.to_integral_value():
        raise ValueParseError(f"integer column rejects fractional value {raw!r}")
    if column.decimal_places is not None and -value.as_tuple().exponent > column.decimal_places:
        raise ValueParseError(f"too many decimal places in {raw!r}")
    return value


def format_numeric(value: Decimal, column: ColumnConfig) -> str:
    """Canonical numeric string: no thousands separator, no trailing zeros."""
    if column.type == "integer":
        return str(int(value))
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def parse_date(raw: str, column: ColumnConfig) -> date:
    """Parse a date with the declared formats; more than one distinct result is ambiguous."""
    text = _strip(raw)
    if not text:
        raise ValueParseError("empty date value")
    results: set[date] = set()
    for date_format in column.date_formats:
        try:
            results.add(datetime.strptime(text, date_format).date())
        except ValueError:
            continue
    if not results:
        try:
            return date.fromisoformat(text)
        except ValueError as error:
            raise ValueParseError(f"no configured format parses {raw!r}") from error
    if len(results) > 1:
        raise AmbiguousValueError(f"{raw!r} parses to multiple dates under the declared formats")
    return results.pop()


def format_date(value: date) -> str:
    return value.isoformat()


def numeric_in_range(value: Decimal, column: ColumnConfig) -> bool:
    return not (
        (column.min_value is not None and value < column.min_value)
        or (column.max_value is not None and value > column.max_value)
    )


def cell_is_resolved(raw: str | None, column: ColumnConfig) -> bool:
    """True when a compare column cell already has a definite logical value."""
    if raw is None:
        return True
    if column.type in NUMERIC_TYPES:
        try:
            return format_numeric(parse_numeric(raw, column), column) == raw
        except ValueParseError:
            return False
    if column.type == "date":
        try:
            return format_date(parse_date(raw, column)) == raw
        except ValueParseError:
            return False
    if column.type == "categorical":
        return raw in column.allowed_values
    return True


def canonical_key(raw: str | None, column: ColumnConfig) -> tuple[str, object]:
    """Canonical logical comparison key used by duplicate detection."""
    if raw is None:
        return ("null", "")
    if column.type in NUMERIC_TYPES:
        try:
            return ("num", parse_numeric(raw, column))
        except ValueParseError:
            return ("raw", raw)
    if column.type == "date":
        try:
            return ("date", parse_date(raw, column))
        except ValueParseError:
            return ("raw", raw)
    if column.type == "categorical":
        return ("str", raw)
    return ("str", raw)


def is_numeric_outlier_iqr(values: list[Decimal], raw: str, column: ColumnConfig) -> bool:
    """1.5 x IQR outlier test with pandas linear-interpolation quantiles."""
    if len(values) < 4:
        return False
    series = pd.Series([float(value) for value in values], dtype="float64")
    q1 = float(series.quantile(0.25, interpolation="linear"))
    q3 = float(series.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    try:
        value = float(parse_numeric(raw, column))
    except ValueParseError:
        return False
    return value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr


def median_decimal(values: list[Decimal], column: ColumnConfig) -> Decimal:
    """Deterministic median; integer columns round half up on the Decimal value."""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    result = (
        ordered[middle]
        if count % 2 == 1
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    if column.type == "integer":
        return result.quantize(Decimal(1), rounding=ROUND_HALF_UP)
    if column.decimal_places is not None:
        quantum = Decimal(1).scaleb(-column.decimal_places)
        return result.quantize(quantum, rounding=ROUND_HALF_UP)
    return result


def mode_value(values: list[str]) -> str:
    """Most frequent value; ties resolve by Unicode lexicographic order."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def _make_selection(state: DatasetState, column: str | None, row_ids: list[str]) -> Selection:
    ordered = [row_id for row_id in state.row_ids if row_id in set(row_ids)]
    return Selection(
        id="sel-" + fingerprint({"column": column, "row_ids": ordered})[:16],
        version_id=state.version_id,
        row_ids=ordered,
        column=column,
        fingerprint=fingerprint({"column": column, "row_ids": ordered}),
    )


def _make_issue(
    category: IssueCategory,
    column: str | None,
    selection: Selection,
    evidence: Evidence,
    status: IssueStatus,
    reason_codes: list[ReasonCode],
) -> Issue:
    return Issue(
        id="iss-" + fingerprint({"category": category.value, "selection": selection.id})[:16],
        category=category,
        column=column,
        selection_id=selection.id,
        evidence_ids=[evidence.id],
        status=status,
        reason_codes=reason_codes,
    )


def _make_evidence(
    state: DatasetState, check_name: str, counts: dict[str, int | float], summary: str, details: list[dict]
) -> Evidence:
    payload = {"check": check_name, "counts": counts, "details": details}
    return Evidence(
        id="ev-" + fingerprint(payload)[:16],
        version_id=state.version_id,
        check_name=check_name,
        counts=counts,
        summary=summary,
        details=details[:8],
    )


@dataclass
class PhaseDetection:
    """Everything a phase discovered in the current snapshot."""

    phase: Phase
    issues: list[Issue] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    selections: dict[str, Selection] = field(default_factory=dict)
    candidates_data: list[dict] = field(default_factory=list)


def _unresolved_columns(state: DatasetState, categories: set[IssueCategory]) -> dict[str, set[str]]:
    """Map column -> blocked row ids for unresolved earlier-phase issues."""
    blocked: dict[str, set[str]] = {}
    for issue in state.issues:
        if issue.status != IssueStatus.UNRESOLVED or issue.category not in categories:
            continue
        if issue.column is None or issue.selection_id is None:
            continue
        selection = state.selection(issue.selection_id)
        rows = selection.row_ids if selection else []
        blocked.setdefault(issue.column, set()).update(rows)
    return blocked


def column_has_unresolved(state: DatasetState, column: str, categories: set[IssueCategory]) -> bool:
    return any(
        issue.status == IssueStatus.UNRESOLVED and issue.category in categories and issue.column == column
        for issue in state.issues
    )


def _column_values(state: DatasetState, column: str) -> list[tuple[str, str]]:
    index = state.column_index()[column]
    return [(row_id, state.rows[row][index]) for row, row_id in enumerate(state.row_ids)]


def _detect_trim(state: DatasetState, config: TableConfig) -> PhaseDetection:
    detection = PhaseDetection(phase=Phase.TRIM)
    for column_name, column in config.columns.items():
        if not column.allows(Operation.TRIM_WHITESPACE):
            continue
        targets = [
            (row_id, raw)
            for row_id, raw in _column_values(state, column_name)
            if raw is not None and raw != raw.strip()
        ]
        if not targets:
            continue
        selection = _make_selection(state, column_name, [row_id for row_id, _ in targets])
        evidence = _make_evidence(
            state,
            "trim_whitespace",
            {"target_cells": len(targets)},
            f"{len(targets)} cells in {column_name!r} have leading or trailing whitespace",
            [{"row_id": row_id, "before": raw, "after": raw.strip()} for row_id, raw in targets[:8]],
        )
        issue = _make_issue(
            IssueCategory.WHITESPACE, column_name, selection, evidence, IssueStatus.OPEN, []
        )
        detection.issues.append(issue)
        detection.evidence.append(evidence)
        detection.selections[selection.id] = selection
        detection.candidates_data.append(
            {
                "operation": Operation.TRIM_WHITESPACE,
                "column": column_name,
                "selection": selection,
                "issue_ids": [issue.id],
                "evidence_ids": [evidence.id],
                "parameters": {},
                "targets": targets,
            }
        )
    return detection


def _detect_null_tokens(state: DatasetState, config: TableConfig) -> PhaseDetection:
    detection = PhaseDetection(phase=Phase.NULL_NORMALIZATION)
    for column_name, column in config.columns.items():
        if not column.allows(Operation.NORMALIZE_NULL_TOKENS):
            continue
        tokens = set(column.null_tokens)
        targets = [
            (row_id, raw)
            for row_id, raw in _column_values(state, column_name)
            if raw is not None and raw in tokens
        ]
        if not targets:
            continue
        selection = _make_selection(state, column_name, [row_id for row_id, _ in targets])
        evidence = _make_evidence(
            state,
            "normalize_null_tokens",
            {"target_cells": len(targets), "distinct_tokens": len({raw for _, raw in targets})},
            f"{len(targets)} cells in {column_name!r} match declared null tokens",
            [{"row_id": row_id, "before": raw, "after": None} for row_id, raw in targets[:8]],
        )
        issue = _make_issue(
            IssueCategory.NULL_TOKEN, column_name, selection, evidence, IssueStatus.OPEN, []
        )
        detection.issues.append(issue)
        detection.evidence.append(evidence)
        detection.selections[selection.id] = selection
        detection.candidates_data.append(
            {
                "operation": Operation.NORMALIZE_NULL_TOKENS,
                "column": column_name,
                "selection": selection,
                "issue_ids": [issue.id],
                "evidence_ids": [evidence.id],
                "parameters": {},
                "targets": targets,
            }
        )
    return detection


def _detect_numeric_cast(state: DatasetState, config: TableConfig) -> PhaseDetection:
    detection = PhaseDetection(phase=Phase.NUMERIC_CAST)
    for column_name, column in config.columns.items():
        if column.type not in NUMERIC_TYPES or not column.allows(Operation.CAST_NUMERIC):
            continue
        castable: list[tuple[str, str, str]] = []
        failures: list[tuple[str, str]] = []
        for row_id, raw in _column_values(state, column_name):
            if raw is None:
                continue
            try:
                canonical = format_numeric(parse_numeric(raw, column), column)
            except ValueParseError:
                failures.append((row_id, raw))
                continue
            if canonical != raw:
                castable.append((row_id, raw, canonical))
        if castable:
            selection = _make_selection(state, column_name, [row_id for row_id, _, _ in castable])
            evidence = _make_evidence(
                state,
                "cast_numeric",
                {"target_cells": len(castable)},
                f"{len(castable)} cells in {column_name!r} are parseable but not canonical",
                [
                    {"row_id": row_id, "before": raw, "after": canonical}
                    for row_id, raw, canonical in castable[:8]
                ],
            )
            issue = _make_issue(
                IssueCategory.NUMERIC_FORMAT, column_name, selection, evidence, IssueStatus.OPEN, []
            )
            detection.issues.append(issue)
            detection.evidence.append(evidence)
            detection.selections[selection.id] = selection
            detection.candidates_data.append(
                {
                    "operation": Operation.CAST_NUMERIC,
                    "column": column_name,
                    "selection": selection,
                    "issue_ids": [issue.id],
                    "evidence_ids": [evidence.id],
                    "parameters": {},
                    "targets": [(row_id, raw) for row_id, raw, _ in castable],
                    "canonical": {row_id: canonical for row_id, _, canonical in castable},
                }
            )
        if failures:
            selection = _make_selection(state, column_name, [row_id for row_id, _ in failures])
            evidence = _make_evidence(
                state,
                "numeric_parse_failure",
                {"target_cells": len(failures)},
                f"{len(failures)} cells in {column_name!r} cannot be parsed under the configured format",
                [{"row_id": row_id, "value": raw} for row_id, raw in failures[:8]],
            )
            issue = _make_issue(
                IssueCategory.PARSE_FAILURE,
                column_name,
                selection,
                evidence,
                IssueStatus.UNRESOLVED,
                [ReasonCode.PARSE_FAILURE, ReasonCode.CANDIDATE_UNAVAILABLE],
            )
            detection.issues.append(issue)
            detection.evidence.append(evidence)
            detection.selections[selection.id] = selection
    return detection


def _detect_date_parsing(state: DatasetState, config: TableConfig) -> PhaseDetection:
    detection = PhaseDetection(phase=Phase.DATE_PARSING)
    for column_name, column in config.columns.items():
        if column.type != "date" or not column.allows(Operation.PARSE_DATETIME):
            continue
        parseable: list[tuple[str, str, str]] = []
        ambiguous: list[tuple[str, str]] = []
        failures: list[tuple[str, str]] = []
        for row_id, raw in _column_values(state, column_name):
            if raw is None:
                continue
            try:
                canonical = format_date(parse_date(raw, column))
            except AmbiguousValueError:
                ambiguous.append((row_id, raw))
                continue
            except ValueParseError:
                failures.append((row_id, raw))
                continue
            if canonical != raw:
                parseable.append((row_id, raw, canonical))
        if parseable:
            selection = _make_selection(state, column_name, [row_id for row_id, _, _ in parseable])
            evidence = _make_evidence(
                state,
                "parse_datetime",
                {"target_cells": len(parseable)},
                f"{len(parseable)} cells in {column_name!r} parse to a unique date but are not ISO",
                [
                    {"row_id": row_id, "before": raw, "after": canonical}
                    for row_id, raw, canonical in parseable[:8]
                ],
            )
            issue = _make_issue(
                IssueCategory.DATE_FORMAT, column_name, selection, evidence, IssueStatus.OPEN, []
            )
            detection.issues.append(issue)
            detection.evidence.append(evidence)
            detection.selections[selection.id] = selection
            detection.candidates_data.append(
                {
                    "operation": Operation.PARSE_DATETIME,
                    "column": column_name,
                    "selection": selection,
                    "issue_ids": [issue.id],
                    "evidence_ids": [evidence.id],
                    "parameters": {},
                    "targets": [(row_id, raw) for row_id, raw, _ in parseable],
                    "canonical": {row_id: canonical for row_id, _, canonical in parseable},
                }
            )
        if ambiguous:
            selection = _make_selection(state, column_name, [row_id for row_id, _ in ambiguous])
            evidence = _make_evidence(
                state,
                "date_ambiguity",
                {"target_cells": len(ambiguous)},
                f"{len(ambiguous)} cells in {column_name!r} parse to more than one date",
                [{"row_id": row_id, "value": raw} for row_id, raw in ambiguous[:8]],
            )
            issue = _make_issue(
                IssueCategory.AMBIGUOUS_PARSE,
                column_name,
                selection,
                evidence,
                IssueStatus.UNRESOLVED,
                [ReasonCode.PARSE_AMBIGUOUS, ReasonCode.CANDIDATE_UNAVAILABLE],
            )
            detection.issues.append(issue)
            detection.evidence.append(evidence)
            detection.selections[selection.id] = selection
        if failures:
            selection = _make_selection(state, column_name, [row_id for row_id, _ in failures])
            evidence = _make_evidence(
                state,
                "date_parse_failure",
                {"target_cells": len(failures)},
                f"{len(failures)} cells in {column_name!r} match no declared date format",
                [{"row_id": row_id, "value": raw} for row_id, raw in failures[:8]],
            )
            issue = _make_issue(
                IssueCategory.PARSE_FAILURE,
                column_name,
                selection,
                evidence,
                IssueStatus.UNRESOLVED,
                [ReasonCode.PARSE_FAILURE, ReasonCode.CANDIDATE_UNAVAILABLE],
            )
            detection.issues.append(issue)
            detection.evidence.append(evidence)
            detection.selections[selection.id] = selection
    return detection


def _detect_duplicates(state: DatasetState, config: TableConfig) -> PhaseDetection:
    detection = PhaseDetection(phase=Phase.DUPLICATES)
    if not config.duplicates.enabled:
        return detection
    compare_columns = config.duplicates.compare_columns
    blocked = _unresolved_columns(state, PARSE_CATEGORIES)
    blocked_rows: set[str] = set()
    for column_name in compare_columns:
        blocked_rows.update(blocked.get(column_name, set()))
    if blocked_rows:
        selection = _make_selection(state, None, sorted(blocked_rows, key=state.row_ids.index))
        evidence = _make_evidence(
            state,
            "duplicate_dependency",
            {"blocked_rows": len(blocked_rows)},
            "rows with unresolved compare-column issues were excluded from duplicate grouping",
            [{"row_id": row_id} for row_id in list(selection.row_ids)[:8]],
        )
        issue = _make_issue(
            IssueCategory.DEPENDENCY_BLOCKED,
            None,
            selection,
            evidence,
            IssueStatus.UNRESOLVED,
            [ReasonCode.DEPENDENCY_BLOCKED],
        )
        detection.issues.append(issue)
        detection.evidence.append(evidence)
        detection.selections[selection.id] = selection

    groups: dict[tuple, list[str]] = {}
    for row_id, row in zip(state.row_ids, state.rows, strict=True):
        if row_id in blocked_rows:
            continue
        key = tuple(canonical_key(row[state.column_index()[name]], config.columns[name]) for name in compare_columns)
        groups.setdefault(key, []).append(row_id)
    duplicate_groups = {key: ids for key, ids in groups.items() if len(ids) > 1}
    if not duplicate_groups:
        return detection
    kept: list[str] = []
    removed: list[str] = []
    group_details: list[dict] = []
    order = state.row_index()
    for ids in duplicate_groups.values():
        ordered = sorted(ids, key=lambda row_id: order[row_id])
        kept.append(ordered[0])
        removed.extend(ordered[1:])
        group_details.append({"kept_row_id": ordered[0], "removed_row_ids": ordered[1:]})
    selection = _make_selection(state, None, removed)
    evidence = _make_evidence(
        state,
        "remove_exact_duplicates",
        {
            "duplicate_groups": len(duplicate_groups),
            "rows_to_remove": len(removed),
            "rows_to_keep": len(kept),
        },
        f"{len(duplicate_groups)} exact duplicate groups contain {len(removed)} redundant rows",
        group_details,
    )
    issue = _make_issue(
        IssueCategory.DUPLICATE_RECORD, None, selection, evidence, IssueStatus.OPEN, []
    )
    detection.issues.append(issue)
    detection.evidence.append(evidence)
    detection.selections[selection.id] = selection
    detection.candidates_data.append(
        {
            "operation": Operation.REMOVE_EXACT_DUPLICATES,
            "column": None,
            "selection": selection,
            "issue_ids": [issue.id],
            "evidence_ids": [evidence.id],
            "parameters": {
                "compare_columns": list(compare_columns),
                "kept_row_ids": sorted(kept, key=lambda item: order[item]),
                "removed_to_kept": {
                    row_id: group["kept_row_id"]
                    for group in group_details
                    for row_id in group["removed_row_ids"]
                },
            },
            "targets": [(row_id, None) for row_id in selection.row_ids],
        }
    )
    return detection


def _detect_out_of_range(state: DatasetState, config: TableConfig) -> PhaseDetection:
    detection = PhaseDetection(phase=Phase.OUT_OF_RANGE)
    for column_name, column in config.columns.items():
        if column.type not in NUMERIC_TYPES:
            continue
        blocked = column_has_unresolved(state, column_name, PARSE_CATEGORIES)
        valid_values: list[Decimal] = []
        parsed: list[tuple[str, str, Decimal]] = []
        for row_id, raw in _column_values(state, column_name):
            if raw is None:
                continue
            try:
                value = parse_numeric(raw, column)
            except ValueParseError:
                continue
            parsed.append((row_id, raw, value))
            if numeric_in_range(value, column):
                valid_values.append(value)
        if blocked:
            if column.allows(Operation.INVALIDATE_OUT_OF_RANGE) or column.allows(Operation.IMPUTE_MISSING):
                non_null_rows = [
                    row_id for row_id, raw in _column_values(state, column_name) if raw is not None
                ]
                selection = _make_selection(state, column_name, non_null_rows)
                evidence = _make_evidence(
                    state,
                    "range_dependency",
                    {"blocked_column": 1},
                    f"{column_name!r} has unresolved parse issues; range and imputation checks are blocked",
                    [{"column": column_name}],
                )
                issue = _make_issue(
                    IssueCategory.DEPENDENCY_BLOCKED,
                    column_name,
                    selection,
                    evidence,
                    IssueStatus.UNRESOLVED,
                    [ReasonCode.DEPENDENCY_BLOCKED],
                )
                detection.issues.append(issue)
                detection.evidence.append(evidence)
                detection.selections[selection.id] = selection
            continue
        if column.allows(Operation.INVALIDATE_OUT_OF_RANGE):
            targets = [
                (row_id, raw)
                for row_id, raw, value in parsed
                if not numeric_in_range(value, column)
            ]
            if targets:
                selection = _make_selection(state, column_name, [row_id for row_id, _ in targets])
                evidence = _make_evidence(
                    state,
                    "invalidate_out_of_range",
                    {"target_cells": len(targets)},
                    f"{len(targets)} cells in {column_name!r} violate the declared numeric range",
                    [
                        {
                            "row_id": row_id,
                            "before": raw,
                            "min": str(column.min_value),
                            "max": str(column.max_value),
                        }
                        for row_id, raw in targets[:8]
                    ],
                )
                issue = _make_issue(
                    IssueCategory.OUT_OF_RANGE, column_name, selection, evidence, IssueStatus.OPEN, []
                )
                detection.issues.append(issue)
                detection.evidence.append(evidence)
                detection.selections[selection.id] = selection
                detection.candidates_data.append(
                    {
                        "operation": Operation.INVALIDATE_OUT_OF_RANGE,
                        "column": column_name,
                        "selection": selection,
                        "issue_ids": [issue.id],
                        "evidence_ids": [evidence.id],
                        "parameters": {},
                        "targets": targets,

                    }
                )
        if len(valid_values) >= 4:
            outliers = [
                (row_id, raw)
                for row_id, raw, _ in parsed
                if numeric_in_range(parse_numeric(raw, column), column)
                and is_numeric_outlier_iqr(valid_values, raw, column)
            ]
            if outliers:
                selection = _make_selection(state, column_name, [row_id for row_id, _ in outliers])
                evidence = _make_evidence(
                    state,
                    "iqr_outlier_informational",
                    {"target_cells": len(outliers), "valid_values": len(valid_values)},
                    f"{len(outliers)} in-range values in {column_name!r} are IQR outliers; no rewrite is authorized",
                    [{"row_id": row_id, "value": raw} for row_id, raw in outliers[:8]],
                )
                issue = _make_issue(
                    IssueCategory.IQR_OUTLIER,
                    column_name,
                    selection,
                    evidence,
                    IssueStatus.INFORMATIONAL,
                    [ReasonCode.INFORMATIONAL_IQR],
                )
                detection.issues.append(issue)
                detection.evidence.append(evidence)
                detection.selections[selection.id] = selection
    return detection


def _detect_imputation(state: DatasetState, config: TableConfig) -> PhaseDetection:
    detection = PhaseDetection(phase=Phase.IMPUTATION)
    for column_name, column in config.columns.items():
        if not column.allows(Operation.IMPUTE_MISSING):
            continue
        blocked = column_has_unresolved(
            state, column_name, PARSE_CATEGORIES | {IssueCategory.OUT_OF_RANGE}
        )
        if blocked:
            selection = _make_selection(state, column_name, [])
            evidence = _make_evidence(
                state,
                "imputation_dependency",
                {"blocked_column": 1},
                f"{column_name!r} still has unresolved parse or range issues; imputation is blocked",
                [{"column": column_name}],
            )
            issue = _make_issue(
                IssueCategory.DEPENDENCY_BLOCKED,
                column_name,
                selection,
                evidence,
                IssueStatus.UNRESOLVED,
                [ReasonCode.DEPENDENCY_BLOCKED],
            )
            detection.issues.append(issue)
            detection.evidence.append(evidence)
            detection.selections[selection.id] = selection
            continue
        index = state.column_index()[column_name]
        targets = [
            (row_id, state.rows[row][index])
            for row, row_id in enumerate(state.row_ids)
            if state.rows[row][index] is None
        ]
        if not targets:
            continue
        selection = _make_selection(state, column_name, [row_id for row_id, _ in targets])
        if column.type == "categorical":
            donor_values_raw = [
                value
                for value in (state.rows[row][index] for row in range(len(state.row_ids)))
                if value is not None and value in column.allowed_values
            ]
            fill_value = mode_value(donor_values_raw) if donor_values_raw else None
            method = "mode"
            donor_count = len(donor_values_raw)
        else:
            donor_values: list[Decimal] = []
            for row in range(len(state.row_ids)):
                value = state.rows[row][index]
                if value is None:
                    continue
                try:
                    parsed = parse_numeric(value, column)
                except ValueParseError:
                    continue
                if numeric_in_range(parsed, column):
                    donor_values.append(parsed)
            if not donor_values:
                fill_value = None
            else:
                fill_value = format_numeric(median_decimal(donor_values, column), column)
            method = "median"
            donor_count = len(donor_values)
        if fill_value is None:
            reason = ReasonCode.ALL_VALUES_NULL if donor_count == 0 else ReasonCode.NO_VALID_DONORS
            evidence = _make_evidence(
                state,
                "impute_missing",
                {"target_cells": len(targets), "valid_donors": donor_count},
                f"{column_name!r} has {len(targets)} missing cells but no valid donor values",
                [{"row_id": row_id} for row_id, _ in targets[:8]],
            )
            issue = _make_issue(
                IssueCategory.MISSING_VALUE,
                column_name,
                selection,
                evidence,
                IssueStatus.UNRESOLVED,
                [reason, ReasonCode.CANDIDATE_UNAVAILABLE],
            )
            detection.issues.append(issue)
            detection.evidence.append(evidence)
            detection.selections[selection.id] = selection
            continue
        evidence = _make_evidence(
            state,
            "impute_missing",
            {"target_cells": len(targets), "valid_donors": donor_count},
            f"{len(targets)} missing cells in {column_name!r} can use {method} value {fill_value!r}",
            [{"row_id": row_id, "before": None, "after": fill_value} for row_id, _ in targets[:8]],
        )
        issue = _make_issue(
            IssueCategory.MISSING_VALUE, column_name, selection, evidence, IssueStatus.OPEN, []
        )
        detection.issues.append(issue)
        detection.evidence.append(evidence)
        detection.selections[selection.id] = selection
        detection.candidates_data.append(
            {
                "operation": Operation.IMPUTE_MISSING,
                "column": column_name,
                "selection": selection,
                "issue_ids": [issue.id],
                "evidence_ids": [evidence.id],
                "parameters": {
                    "method": method,
                    "fill_value": fill_value,
                    "donor_count": donor_count,
                },
                "targets": targets,
            }
        )
    return detection


_DETECTORS = {
    Phase.TRIM: _detect_trim,
    Phase.NULL_NORMALIZATION: _detect_null_tokens,
    Phase.NUMERIC_CAST: _detect_numeric_cast,
    Phase.DATE_PARSING: _detect_date_parsing,
    Phase.DUPLICATES: _detect_duplicates,
    Phase.OUT_OF_RANGE: _detect_out_of_range,
    Phase.IMPUTATION: _detect_imputation,
}


def detect_phase(state: DatasetState, config: TableConfig, phase: Phase) -> PhaseDetection:
    """Run the deterministic checks for one sub-phase on the current snapshot."""
    return _DETECTORS[phase](state, config)
