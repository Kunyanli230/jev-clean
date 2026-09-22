"""Acceptance 1-4: parsing, imputation, IQR/range phases and exact duplicates."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fake_client import run_fake_pipeline

from idac.config import ColumnConfig, load_config
from idac.models import IssueStatus, Phase
from idac.planner import AcceptedCandidate, build_plan
from idac.profiling import (
    AmbiguousValueError,
    ValueParseError,
    detect_phase,
    median_decimal,
    mode_value,
    parse_date,
    parse_numeric,
)
from idac.storage import load_csv

DEMO_CONFIG = load_config("configs/demo.yaml")


def numeric_column(**overrides) -> ColumnConfig:
    payload = {
        "type": "decimal",
        "description": "amount",
        "null_tokens": [""],
        "allowed_operations": ["cast_numeric"],
    }
    payload.update(overrides)
    return ColumnConfig.model_validate(payload)


def test_numeric_thousands_and_decimal_separator() -> None:
    column = numeric_column(thousands_separator=",", decimal_places=2)
    assert parse_numeric("1,234.50", column) == Decimal("1234.50")
    assert parse_numeric("  1,234.50 ", column) == Decimal("1234.50")
    assert parse_numeric("12.5", column) == Decimal("12.5")

    european = numeric_column(decimal_separator=",", thousands_separator=".", decimal_places=2)
    assert parse_numeric("1.234,50", european) == Decimal("1234.50")

    with pytest.raises(ValueParseError):
        parse_numeric("12,34.5", column)
    with pytest.raises(ValueParseError):
        parse_numeric("1,234.567", column)
    with pytest.raises(ValueParseError):
        parse_numeric("nan", column)
    with pytest.raises(ValueParseError):
        parse_numeric("inf", column)
    with pytest.raises(ValueParseError):
        parse_numeric("", column)


def test_integer_precision_rejects_fractions() -> None:
    column = ColumnConfig.model_validate(
        {
            "type": "integer",
            "description": "count",
            "null_tokens": [""],
            "allowed_operations": ["cast_numeric"],
        }
    )
    assert parse_numeric("0042", column) == Decimal(42)
    with pytest.raises(ValueParseError):
        parse_numeric("42.5", column)


def test_date_ambiguity_and_unique_parse() -> None:
    column = ColumnConfig.model_validate(
        {
            "type": "date",
            "description": "date",
            "null_tokens": [""],
            "allowed_operations": ["parse_datetime"],
            "date_formats": ["%d/%m/%Y", "%m/%d/%Y"],
        }
    )
    assert parse_date("25/12/2023", column).isoformat() == "2023-12-25"
    assert parse_date("12/25/2023", column).isoformat() == "2023-12-25"
    assert parse_date("2024-04-03", column).isoformat() == "2024-04-03"
    with pytest.raises(AmbiguousValueError):
        parse_date("03/04/2024", column)
    with pytest.raises(ValueParseError):
        parse_date("31/02/2024", column)


SMALL_CONFIG = """
table_description: "small demo table"
record_grain: "one row per record"
columns:
  id:
    type: string
    description: "protected identifier with leading zeros"
    protected: true
    null_tokens: [""]
    allowed_operations: []
  age:
    type: integer
    description: "age"
    min_value: 18
    max_value: 100
    null_tokens: ["", "NA"]
    allowed_operations: [trim_whitespace, normalize_null_tokens, cast_numeric, invalidate_out_of_range, impute_missing]
  city:
    type: categorical
    description: "city"
    allowed_values: ["Berlin", "Hamburg", "Munich"]
    null_tokens: ["", "unknown"]
    allowed_operations: [trim_whitespace, normalize_null_tokens, impute_missing]
  score:
    type: decimal
    description: "score"
    decimal_places: 2
    min_value: 0
    max_value: 1000
    null_tokens: [""]
    allowed_operations: [trim_whitespace, normalize_null_tokens, cast_numeric, invalidate_out_of_range, impute_missing]
duplicates:
  enabled: true
  compare_columns: [id, age, city, score]
"""


def test_leading_zero_protected_id_and_whitespace(tmp_path) -> None:
    rows = [
        [" 001 ", " 21 ", " Berlin ", "10.5"],
        ["002", "22", "Hamburg", "20.5"],
    ]
    result, store, _ = run_fake_pipeline(tmp_path, SMALL_CONFIG, ["id", "age", "city", "score"], rows)
    cleaned = store.current_state()
    assert cleaned.rows[0][0] == " 001 "  # protected: never rewritten
    assert cleaned.rows[0][1] == "21"
    assert cleaned.rows[0][2] == "Berlin"
    assert result.outcome.value in {"completed", "partial"}


def test_median_round_half_up_and_mode_tie() -> None:
    integer_column = ColumnConfig.model_validate(
        {
            "type": "integer",
            "description": "n",
            "null_tokens": [""],
            "allowed_operations": ["impute_missing"],
        }
    )
    assert median_decimal([Decimal(1), Decimal(2)], integer_column) == Decimal(2)
    assert median_decimal([Decimal(1), Decimal(4)], integer_column) == Decimal(3)
    decimal_column = numeric_column(decimal_places=2)
    assert median_decimal([Decimal("1.005"), Decimal("1.005")], decimal_column) == Decimal("1.01")
    assert mode_value(["b", "a", "b", "a"]) == "a"


def test_all_null_column_is_not_imputed(tmp_path) -> None:
    config_text = SMALL_CONFIG
    rows = [["001", "NA", "Berlin", "10"], ["002", "NA", "Hamburg", "20"]]
    result, store, _ = run_fake_pipeline(tmp_path, config_text, ["id", "age", "city", "score"], rows)
    issues = [issue for issue in store.current_state().issues if issue.column == "age"]
    assert any("ALL_VALUES_NULL" in [code.value for code in issue.reason_codes] for issue in issues)
    assert store.current_state().rows[0][1] is None  # normalized, then left missing
    assert result.outcome.value == "partial"


def test_invalid_donors_are_excluded_from_imputation() -> None:
    config = load_config(
        __import__("pathlib").Path(__file__).resolve().parent / ".." / "configs" / "demo.yaml"
    )
    rows = [
        ["001", "Alice", "3", "Berlin", "10.00", "2020-01-01"],
        ["002", "Bob", "40", "Berlin", "20.00", "2020-01-02"],
        ["003", "Cara", "", "Berlin", "30.00", "2020-01-03"],
    ]
    # Build state directly to inspect donor selection.
    import csv
    import pathlib
    import tempfile

    from idac.storage import load_csv

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "input.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["customer_id", "name", "age", "city", "income", "signup_date"])
            writer.writerows(rows)
        state = load_csv(path, config)
    state.rows[2][state.column_index()["age"]] = None  # normalized missing value
    detection = detect_phase(state, config, Phase.IMPUTATION)
    candidate_data = [data for data in detection.candidates_data if data["column"] == "age"]
    assert candidate_data
    # The out-of-range age 3 is excluded from the donor set, so the median is 40.
    assert candidate_data[0]["parameters"]["fill_value"] == "40"
    assert candidate_data[0]["parameters"]["donor_count"] == 1


def test_iqr_only_value_is_not_rewritten(tmp_path) -> None:
    # 95 is inside the declared 18-100 range but a 1.5xIQR outlier: informational only.
    rows = [["001", str(age), "Berlin", "10"] for age in [20, 21, 22, 23, 24, 25, 26, 95]]
    result, store, _ = run_fake_pipeline(
        tmp_path, SMALL_CONFIG, ["id", "age", "city", "score"], rows
    )
    cleaned = store.current_state()
    assert cleaned.rows[-1][1] == "95"
    iqr_issues = [issue for issue in cleaned.issues if issue.category.value == "iqr_outlier"]
    assert iqr_issues and iqr_issues[0].status == IssueStatus.INFORMATIONAL
    config = load_config(tmp_path / "config.yaml")
    detection = detect_phase(cleaned, config, Phase.OUT_OF_RANGE)
    assert not [
        data for data in detection.candidates_data if data["operation"].value == "invalidate_out_of_range"
    ]
    assert result is not None


def test_out_of_range_then_imputed(tmp_path) -> None:
    rows = [["001", str(age), "Berlin", "10"] for age in [20, 30, 40, 50, 999]]
    _, store, _ = run_fake_pipeline(
        tmp_path, SMALL_CONFIG, ["id", "age", "city", "score"], rows
    )
    cleaned = store.current_state()
    assert cleaned.rows[-1][1] == "35"  # 999 invalidated, then median of [20,30,40,50] imputed
    assert cleaned.rows[-1][1] != "999"


def test_exact_duplicates_keep_earliest_row_id(tmp_path) -> None:
    rows = [
        ["001", "20", "Berlin", "10"],
        ["001", "20", "Berlin", "10"],  # exact duplicate of row 0: removed
        ["001", "20", "Berlin", "11"],  # same id, different compare values: kept
        ["003", "21", "Hamburg", "12"],
    ]
    _, store, _ = run_fake_pipeline(
        tmp_path, SMALL_CONFIG, ["id", "age", "city", "score"], rows
    )
    cleaned = store.current_state()
    assert cleaned.row_ids == ["row-000000", "row-000002", "row-000003"]
    assert [removed.row_id for removed in cleaned.removed_rows] == ["row-000001"]


def test_unauthorized_duplicate_deletion_is_not_proposed(tmp_path) -> None:
    config_text = SMALL_CONFIG.replace(
        "duplicates:\n  enabled: true\n  compare_columns: [id, age, city, score]",
        "duplicates:\n  enabled: false\n  compare_columns: []",
    )
    rows = [["001", "20", "Berlin", "10"], ["001", "20", "Berlin", "10"]]
    _, store, _ = run_fake_pipeline(
        tmp_path, config_text, ["id", "age", "city", "score"], rows
    )
    assert len(store.current_state().rows) == 2
    assert not store.current_state().removed_rows


def test_imputation_statistics_are_recomputed_after_dedup(tmp_path) -> None:
    rows = [
        ["001", "20", "Berlin", "10"],
        ["001", "20", "Berlin", "10"],  # removed as a duplicate before imputation
        ["002", "30", "Berlin", "20"],
        ["003", "", "Berlin", "30"],
    ]
    _, store, client = run_fake_pipeline(
        tmp_path, SMALL_CONFIG, ["id", "age", "city", "score"], rows
    )
    contexts = [
        context
        for call in client.calls
        if call["kind"] == "decision"
        for context in call["state"]["candidates"]
        if context["operation"] == "impute_missing" and context["column"] == "age"
    ]
    assert contexts
    assert contexts[0]["proposed_value"]["donor_count"] == 2  # 20 and 30, not the duplicate
    assert contexts[0]["proposed_value"]["fill_value"] == "25"
    assert store.current_state().rows[-1][1] == "25"


def test_duplicate_dependency_blocking(tmp_path) -> None:
    config_text = SMALL_CONFIG.replace(
        "duplicates:\n  enabled: true\n  compare_columns: [id, age, city, score]",
        "duplicates:\n  enabled: true\n  compare_columns: [id, age, city, score]",
    ).replace(
        'score:\n    type: decimal',
        'score:\n    type: decimal\n    thousands_separator: ","',
    )
    rows = [
        ["001", "20", "Berlin", "10"],
        ["001", "20", "Berlin", "not-a-number"],
        ["001", "20", "Berlin", "not-a-number"],
    ]
    _, store, _ = run_fake_pipeline(
        tmp_path, config_text, ["id", "age", "city", "score"], rows
    )
    state = store.current_state()
    blocked = [
        issue
        for issue in state.issues
        if issue.category.value == "dependency_blocked" and issue.status == IssueStatus.UNRESOLVED
    ]
    assert blocked
    assert len(state.rows) == 3  # unresolved compare values block deletion


def test_plan_rejects_candidate_without_selection(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("id,age,city,score\n001,20,Berlin,10\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(SMALL_CONFIG, encoding="utf-8")
    config = load_config(config_path)
    state = load_csv(input_path, config)
    from idac.models import Operation, RepairCandidate

    orphan = RepairCandidate(
        id="cand-orphan",
        version_id=state.version_id,
        issue_ids=[],
        operation=Operation.TRIM_WHITESPACE,
        selection_id="sel-missing",
        parameters={"column": "id", "after_values": {"row-000000": "001"}},
        preview=[],
        evidence_ids=[],
        fingerprint="fp",
    )
    plan = build_plan(
        state, config, {}, [AcceptedCandidate(candidate=orphan, decision_id="dec")], "plan-0001"
    )
    assert plan.ordered_operations == []
    assert plan.discarded
    assert plan.discarded[0]["reason_code"] == "STALE_VERSION"
