"""Evaluation: quality counts, corruption repair coverage and integrity metrics.

The ground truth and corruption manifest are used only by this module; they
never enter candidate generation or the model state. Metrics are bound to the
run's current version so a rollback cannot keep showing stale numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import NUMERIC_TYPES, TableConfig
from .models import DatasetState, IssueCategory, IssueStatus, ModelStatus
from .policy import replay_decision, verify_request_evidence
from .profiling import (
    ValueParseError,
    format_date,
    format_numeric,
    numeric_in_range,
    parse_date,
    parse_numeric,
)
from .storage import RunStore, load_truth_csv


def logical_value(raw: str | None, column) -> Any:
    if raw is None:
        return None
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
    return ("str", raw)


def _is_canonical(raw: str, column) -> bool:
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


def quality_counts(state: DatasetState, config: TableConfig) -> dict[str, int]:
    """Missing, parse failure, format, range and exact-duplicate counts."""
    missing = 0
    parse_failures = 0
    format_issues = 0
    range_violations = 0
    index = state.column_index()
    for row in state.rows:
        for column_name, column in config.columns.items():
            raw = row[index[column_name]]
            if raw is None:
                missing += 1
                continue
            if column.type in NUMERIC_TYPES:
                try:
                    value = parse_numeric(raw, column)
                except ValueParseError:
                    parse_failures += 1
                    continue
                if not _is_canonical(raw, column):
                    format_issues += 1
                if not numeric_in_range(value, column):
                    range_violations += 1
            elif column.type == "date":
                try:
                    value = parse_date(raw, column)
                except ValueParseError:
                    parse_failures += 1
                    continue
                if format_date(value) != raw:
                    format_issues += 1
            elif column.type == "categorical" and raw not in column.allowed_values:
                parse_failures += 1
    duplicates = 0
    if config.duplicates.enabled:
        compare = config.duplicates.compare_columns
        seen: set[tuple] = set()
        for row in state.rows:
            key = tuple(logical_value(row[index[name]], config.columns[name]) for name in compare)
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
    return {
        "missing_cells": missing,
        "parse_failures": parse_failures,
        "format_issues": format_issues,
        "range_violations": range_violations,
        "exact_duplicate_rows": duplicates,
    }


def _corruption_path(truth_path: Path) -> Path | None:
    candidate = truth_path.parent / "corruption.json"
    return candidate if candidate.is_file() else None


def _valid_in_range(raw: str, column) -> bool:
    try:
        return numeric_in_range(parse_numeric(raw, column), column)
    except ValueParseError:
        return False


def _value_at(state: DatasetState, row_id: str, column: str) -> str | None:
    index = state.row_index().get(row_id)
    if index is None:
        return None
    return state.rows[index][state.column_index()[column]]


def _valid_imputed_value(raw: str | None, column) -> bool:
    """A retained null token or invalid value is not a successful fill."""
    if raw is None or raw in column.null_tokens:
        return False
    if column.type in NUMERIC_TYPES:
        return _valid_in_range(raw, column)
    if column.type == "categorical":
        return raw in column.allowed_values
    return False


def evaluate_run(
    store: RunStore,
    config: TableConfig,
    truth_path: str | Path,
    baseline_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute fixed comparison metrics and write evaluation_<version>.json."""
    truth_path = Path(truth_path)
    truth = load_truth_csv(truth_path)
    corruption_file = _corruption_path(truth_path)
    corruption = (
        json.loads(corruption_file.read_text(encoding="utf-8")) if corruption_file else {}
    )
    manifest = store.manifest()
    version = manifest["current_version"]
    original = store.load_snapshot("v000")
    cleaned = store.load_snapshot(version)
    truth_rows = truth.values.tolist()
    truth_columns = [str(column) for column in truth.columns]
    truth_index = {column: position for position, column in enumerate(truth_columns)}

    metrics: dict[str, Any] = {
        "run": manifest["run_id"],
        "version": version,
        "model_identity": manifest.get("model_identity", "real"),
        "quality": {
            "original": quality_counts(original, config),
            "cleaned": quality_counts(cleaned, config),
        },
        "corruption": _corruption_metrics(corruption, original, cleaned, truth_rows, truth_index, config),
        "duplicates": _duplicate_metrics(corruption, cleaned),
        "api": _api_metrics(store),
        "integrity": integrity_metrics(store),
        "baseline": None,
    }
    baseline = Path(baseline_dir) if baseline_dir else _default_baseline_dir(store)
    if baseline and (baseline / "cleaned.csv").is_file():
        metrics["baseline"] = _baseline_metrics(
            baseline,
            config,
            original,
            corruption,
            truth_rows,
            truth_index,
        )
    store.write_evaluation(version, metrics)
    return metrics


def _default_baseline_dir(store: RunStore) -> Path | None:
    candidate = store.run_dir.parent / "baseline"
    return candidate if candidate.is_dir() else None


def _corruption_metrics(
    corruption: dict[str, Any],
    original: DatasetState,
    cleaned: DatasetState,
    truth_rows: list[list[str]],
    truth_index: dict[str, int],
    config: TableConfig,
) -> dict[str, Any]:
    entries = corruption.get("entries", [])
    duplicate_sources = corruption.get("duplicate_sources", {})
    removed = {removed.row_id for removed in cleaned.removed_rows}
    repaired = 0
    repairable = 0
    truth_matches = 0
    unrepaired_reasons: dict[str, int] = {}
    wrong_modifications = 0
    numeric_missing_total = 0
    numeric_missing_filled = 0
    numeric_errors: list[float] = []
    categorical_missing_total = 0
    categorical_missing_filled = 0
    categorical_correct = 0
    truth_lookup: dict[str, list[str]] = {}
    for row_id, row in zip(original.row_ids[: len(truth_rows)], truth_rows, strict=True):
        truth_lookup[row_id] = row
    polluted: set[tuple[str, str]] = set()
    for entry in entries:
        row_id = entry["row_id"]
        column = entry["column"]
        polluted.add((row_id, column))
        truth_row = truth_lookup.get(row_id)
        if truth_row is None:
            continue
        truth_value = truth_row[truth_index[column]]
        cleaned_row_id = row_id if row_id not in removed else duplicate_sources.get(row_id, row_id)
        cleaned_value = _value_at(cleaned, cleaned_row_id, column)
        column_config = config.columns[column]
        truth_match = logical_value(cleaned_value, column_config) == logical_value(
            truth_value, column_config
        )
        corruption_type = entry.get("corruption_type")
        if corruption_type in ("numeric_missing", "categorical_missing"):
            matches = _valid_imputed_value(cleaned_value, column_config)
        elif corruption_type == "null_token":
            matches = cleaned_value != entry.get("dirty_value")
        elif corruption_type == "out_of_range":
            matches = cleaned_value is None or _valid_in_range(cleaned_value, column_config)
        else:
            matches = truth_match
        if entry.get("recoverable_by_rule", True):
            repairable += 1
            truth_matches += int(truth_match)
            if matches:
                repaired += 1
            else:
                reason = corruption_type or "unknown"
                unrepaired_reasons[reason] = unrepaired_reasons.get(reason, 0) + 1
        if entry.get("corruption_type") == "numeric_missing":
            numeric_missing_total += 1
            if _valid_imputed_value(cleaned_value, column_config):
                numeric_missing_filled += 1
                try:
                    cleaned_number = parse_numeric(cleaned_value, column_config)
                    truth_number = parse_numeric(truth_value, column_config)
                    numeric_errors.append(abs(float(cleaned_number - truth_number)))
                except ValueParseError:
                    pass
        if entry.get("corruption_type") == "categorical_missing":
            categorical_missing_total += 1
            if _valid_imputed_value(cleaned_value, column_config):
                categorical_missing_filled += 1
                if cleaned_value == truth_value:
                    categorical_correct += 1
    for row_id, truth_row in truth_lookup.items():
        if row_id not in original.row_ids:
            continue
        for column_name, position in truth_index.items():
            if column_name not in config.columns:
                continue
            if (row_id, column_name) in polluted:
                continue
            original_value = _value_at(original, row_id, column_name)
            cleaned_row_id = row_id if row_id not in removed else duplicate_sources.get(row_id, row_id)
            cleaned_value = _value_at(cleaned, cleaned_row_id, column_name)
            column_config = config.columns[column_name]
            if logical_value(original_value, column_config) == logical_value(
                truth_row[position], column_config
            ) and logical_value(cleaned_value, column_config) != logical_value(
                truth_row[position], column_config
            ):
                wrong_modifications += 1
    mae = sum(numeric_errors) / len(numeric_errors) if numeric_errors else None
    return {
        "repairable_entries": repairable,
        "repaired_entries": repaired,
        "repair_coverage": (repaired / repairable) if repairable else None,
        "truth_match_entries": truth_matches,
        "truth_match_rate": (truth_matches / repairable) if repairable else None,
        "unrepaired_by_type": unrepaired_reasons,
        "wrong_modifications": wrong_modifications,
        "numeric_imputation": {
            "masked_total": numeric_missing_total,
            "filled": numeric_missing_filled,
            "coverage": (numeric_missing_filled / numeric_missing_total)
            if numeric_missing_total
            else None,
            "mae": mae,
            "mae_denominator": len(numeric_errors),
        },
        "categorical_imputation": {
            "masked_total": categorical_missing_total,
            "filled": categorical_missing_filled,
            "coverage": (categorical_missing_filled / categorical_missing_total)
            if categorical_missing_total
            else None,
            "correct": categorical_correct,
            "accuracy": (categorical_correct / categorical_missing_filled)
            if categorical_missing_filled
            else None,
        },
    }


def _duplicate_metrics(corruption: dict[str, Any], cleaned: DatasetState) -> dict[str, Any]:
    expected_removed = set(corruption.get("duplicate_sources", {}))
    removed = {removed.row_id for removed in cleaned.removed_rows}
    return {
        "expected_removed": len(expected_removed),
        "removed": len(removed),
        "wrong_deletions": len(removed - expected_removed),
        "missed_deletions": len(expected_removed - removed),
    }


def _api_metrics(store: RunStore) -> dict[str, Any]:
    requests = store.read_requests()
    retries = sum(max(0, len(request.get("attempts", [])) - 1) for request in requests)
    known_tokens = 0
    unknown = 0
    elapsed = 0.0
    for request in requests:
        usage = request.get("usage") or {}
        if usage.get("known") and usage.get("input_tokens") is not None:
            known_tokens += int(usage["input_tokens"])
        elif request.get("model_status") == "ok":
            unknown += 1
        elapsed += float(request.get("elapsed_seconds") or 0.0)
    manifest = store.manifest()
    audit = store.read_audit()
    run_finished = next(
        (event for event in reversed(audit) if event["event_type"] == "run_finished"), None
    )
    budget = (run_finished or {}).get("details", {}).get("budget", {})
    return {
        "requests": len(requests),
        "decision_requests": sum(1 for request in requests if request.get("kind") == "decision"),
        "postcheck_requests": sum(1 for request in requests if request.get("kind") == "postcheck"),
        "retries": retries,
        "known_input_tokens": known_tokens,
        "unknown_usage_requests": unknown,
        "request_elapsed_seconds": round(elapsed, 3),
        "run_elapsed_seconds": budget.get("elapsed_seconds"),
        "budget_attempts": budget.get("attempts"),
        "stop_reason": manifest.get("stop_reason"),
    }


def integrity_metrics(store: RunStore) -> dict[str, Any]:
    decisions = store.read_decisions()
    traces = {trace["decision_id"]: trace for trace in store.read_traces()}
    valid = [record for record in decisions if record.get("model_status") == ModelStatus.OK.value]
    complete = [
        record
        for record in valid
        if record.get("choice_assessment") is not None and record.get("risk_assessment") is not None
    ]
    traceable = [
        record
        for record in decisions
        if record.get("evidence_ids") and record.get("policy_version") and record["id"] in traces
        and verify_request_evidence(record, store.run_dir)
    ]
    replayable = 0
    replay_consistent = 0
    for record in valid:
        result = replay_decision(record, store.run_dir)
        if result.replayable:
            replayable += 1
            if result.consistent:
                replay_consistent += 1
    return {
        "distribution_save_rate": _rate(len(complete), len(valid)),
        "distribution_save_numerator": len(complete),
        "distribution_save_denominator": len(valid),
        "traceability_rate": _rate(len(traceable), len(decisions)),
        "traceability_numerator": len(traceable),
        "traceability_denominator": len(decisions),
        "gate_replay_rate": _rate(replay_consistent, len(valid)),
        "gate_replay_numerator": replay_consistent,
        "gate_replay_denominator": len(valid),
        "gate_replay_unavailable": len(valid) - replayable,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _baseline_metrics(
    baseline_dir: Path,
    config: TableConfig,
    original: DatasetState,
    corruption: dict[str, Any],
    truth_rows: list[list[str]],
    truth_index: dict[str, int],
) -> dict[str, Any]:
    from .models import RemovedRow
    from .storage import load_csv

    baseline_state = load_csv(baseline_dir / "cleaned.csv", config)
    removed_path = baseline_dir / "removed_rows.csv"
    if removed_path.is_file():
        import csv as csv_module

        with removed_path.open(encoding="utf-8", newline="") as handle:
            reader = csv_module.DictReader(handle)
            baseline_state.removed_rows = [
                RemovedRow(
                    row_id=row["row_id"],
                    values=[row.get(column) or None for column in baseline_state.ordered_columns],
                    reason=row.get("reason", ""),
                    candidate_id=row.get("candidate_id") or None,
                )
                for row in reader
            ]
    return {
        "kind": "rule_baseline",
        "cleaned": quality_counts(baseline_state, config),
        "corruption": _corruption_metrics(
            corruption, original, baseline_state, truth_rows, truth_index, config
        ),
        "duplicates": _duplicate_metrics(corruption, baseline_state),
        "removed_rows": len(baseline_state.removed_rows),
        "model_distribution_metrics": "N/A - rule baseline makes no probability calls",
    }


def unresolved_summary(state: DatasetState) -> dict[str, int]:
    summary: dict[str, int] = {}
    for issue in state.issues:
        if issue.status != IssueStatus.UNRESOLVED:
            continue
        for reason in issue.reason_codes:
            summary[reason.value] = summary.get(reason.value, 0) + 1
        if not issue.reason_codes:
            summary[issue.category.value] = summary.get(issue.category.value, 0) + 1
    return summary


def actionable_issue_count(state: DatasetState) -> int:
    return sum(1 for issue in state.issues if issue.status == IssueStatus.OPEN)


def informational_issue_count(state: DatasetState) -> int:
    return sum(
        1
        for issue in state.issues
        if issue.status == IssueStatus.INFORMATIONAL
        or issue.category == IssueCategory.IQR_OUTLIER
    )
