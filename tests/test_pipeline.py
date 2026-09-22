"""Acceptance 11, 13, 17-19: fake integration, baseline parity and reports."""

from __future__ import annotations

import json

from fake_client import DEMO_CONFIG, DEMO_DATA, FakeDecisionClient
from typer.testing import CliRunner

from idac.cli import app
from idac.config import load_config
from idac.evaluation import evaluate_run, quality_counts
from idac.explanation import render_expected_examples
from idac.models import RunOutcome, decision_from_log
from idac.orchestrator import run_clean
from idac.policy import replay_decision
from idac.storage import RunStore

CLEAN_CONFIG = """
table_description: "clean pipeline test"
record_grain: "one row per record"
columns:
  id:
    type: string
    description: "identifier"
    protected: true
    null_tokens: [""]
    allowed_operations: []
  age:
    type: integer
    description: "age"
    min_value: 18
    max_value: 100
    null_tokens: [""]
    allowed_operations: [trim_whitespace, normalize_null_tokens, cast_numeric, invalidate_out_of_range, impute_missing]
duplicates:
  enabled: false
  compare_columns: []
"""


def test_demo_fake_pipeline_commits_and_evaluates(tmp_path) -> None:
    client = FakeDecisionClient()
    output = tmp_path / "demo"
    result = run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    assert result.outcome == RunOutcome.PARTIAL  # the ambiguous date is intentionally unresolved
    store = RunStore(output)
    state = store.current_state()
    assert len(state.rows) == 200
    assert len(state.removed_rows) == 10
    unresolved = [issue for issue in state.issues if issue.status.value == "unresolved"]
    assert any("PARSE_AMBIGUOUS" in [code.value for code in issue.reason_codes] for issue in unresolved)
    iqr = [issue for issue in state.issues if issue.category.value == "iqr_outlier"]
    assert iqr and iqr[0].status.value == "informational"

    config = load_config(DEMO_CONFIG)
    metrics = evaluate_run(store, config, DEMO_DATA / "ground_truth.csv")
    assert metrics["corruption"]["repair_coverage"] == 1.0
    assert metrics["corruption"]["wrong_modifications"] == 0
    assert metrics["duplicates"]["wrong_deletions"] == 0
    assert metrics["duplicates"]["missed_deletions"] == 0
    assert metrics["integrity"]["distribution_save_rate"] == 1.0
    assert metrics["integrity"]["traceability_rate"] == 1.0
    assert metrics["integrity"]["gate_replay_rate"] == 1.0
    assert metrics["baseline"] is None


def test_all_abstain_does_not_change_data(tmp_path) -> None:
    client = FakeDecisionClient(approve=False)
    output = tmp_path / "demo"
    result = run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    assert result.outcome == RunOutcome.PARTIAL
    assert result.current_version == "v000"
    store = RunStore(output)
    state = store.current_state()
    assert len(state.rows) == 210
    assert state.rows[0][1] == "  Ulf Dorn "  # original value untouched
    reasons = {
        code for record in store.read_decisions() for code in record["reason_codes"]
    }
    assert "MODEL_ABSTAIN" in reasons


def test_no_repeat_sampling_after_rejection(tmp_path) -> None:
    client = FakeDecisionClient(approve=False)
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, tmp_path / "demo", client=client)
    asked: list[str] = []
    for request in client.calls:
        if request["kind"] != "decision":
            continue
        for context in request["state"]["candidates"]:
            asked.append(context["candidate_id"])
    assert len(asked) == len(set(asked)), "a rejected candidate must not be re-sampled"
    assert len(asked) >= 5  # dependency blocking reduces the remaining phases
    # With every candidate approved, the demo produces exactly one candidate per issue.
    approving = FakeDecisionClient()
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, tmp_path / "approved", client=approving)
    approved_ids = [
        context["candidate_id"]
        for request in approving.calls
        if request["kind"] == "decision"
        for context in request["state"]["candidates"]
    ]
    assert len(approved_ids) == 12
    assert len(approved_ids) == len(set(approved_ids))


def test_truth_never_enters_the_model_payload(tmp_path) -> None:
    import csv

    client = FakeDecisionClient()
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, tmp_path / "demo", client=client)
    with (DEMO_DATA / "dirty.csv").open(encoding="utf-8", newline="") as handle:
        dirty_rows = list(csv.DictReader(handle))
    payloads = [
        json.dumps(call["state"], ensure_ascii=False) + json.dumps(call["questions"], default=str)
        for call in client.calls
    ]
    joined = "\n".join(payloads)
    for forbidden in ("ground_truth", "truth_value", "corruption.json", "2024-04-03"):
        assert forbidden not in joined
    # Every sampled before-value must be the dirty cell, never the ground truth.
    sampled = 0
    for call in client.calls:
        if call["kind"] != "decision":
            continue
        for context in call["state"]["candidates"]:
            for sample in context["samples"]:
                if sample["before"] is None or sample["column"] not in dirty_rows[0]:
                    continue
                row_index = int(sample["row_id"].split("-")[1])
                assert sample["before"] == dirty_rows[row_index][sample["column"]]
                sampled += 1
    assert sampled > 0


def test_completed_outcome_when_nothing_is_actionable(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    config_path = write_config(tmp_path, CLEAN_CONFIG)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", "21"], ["002", "22"]])
    client = FakeDecisionClient()
    result = run_clean(input_path, config_path, tmp_path / "run", client=client)
    assert result.outcome == RunOutcome.COMPLETED
    assert result.stop_reason == "no_actionable_issues"
    assert not client.calls


def test_integrity_metrics_show_na_for_zero_denominators(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    from idac.evaluation import integrity_metrics

    config_path = write_config(tmp_path, CLEAN_CONFIG)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", "21"], ["002", "22"]])
    run_clean(input_path, config_path, tmp_path / "run", client=FakeDecisionClient())
    metrics = integrity_metrics(RunStore(tmp_path / "run"))
    assert metrics["distribution_save_rate"] is None
    assert metrics["distribution_save_denominator"] == 0
    assert metrics["gate_replay_rate"] is None
    assert metrics["gate_replay_denominator"] == 0


def test_no_model_call_when_no_issues(tmp_path) -> None:
    from fake_client import write_config, write_csv_file

    config_path = write_config(tmp_path, CLEAN_CONFIG)
    input_path = write_csv_file(tmp_path, ["id", "age"], [["001", "21"], ["002", "22"]])
    client = FakeDecisionClient()
    run_clean(input_path, config_path, tmp_path / "run", client=client)
    assert client.calls == []


def test_baseline_and_pipeline_share_quality_checks(tmp_path) -> None:
    from examples.run_baseline import BaselineRunner

    config = load_config(DEMO_CONFIG)
    from idac.storage import load_csv

    state = load_csv(DEMO_DATA / "dirty.csv", config)
    runner = BaselineRunner(config, state)
    final = runner.run()
    assert runner.hard_failures == []
    baseline_counts = quality_counts(final, config)
    assert baseline_counts["exact_duplicate_rows"] == 0
    assert baseline_counts["parse_failures"] == 1  # the ambiguous date remains unresolved
    client = FakeDecisionClient()
    output = tmp_path / "demo"
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    store = RunStore(output)
    idac_counts = quality_counts(store.current_state(), config)
    assert set(baseline_counts) == set(idac_counts)
    assert baseline_counts["parse_failures"] == idac_counts["parse_failures"]
    assert baseline_counts["range_violations"] == idac_counts["range_violations"]
    assert baseline_counts["exact_duplicate_rows"] == idac_counts["exact_duplicate_rows"]

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    from idac.storage import write_cleaned_csv

    write_cleaned_csv(final, baseline_dir / "cleaned.csv")
    metrics = evaluate_run(store, config, DEMO_DATA / "ground_truth.csv", baseline_dir=baseline_dir)
    assert metrics["baseline"] is not None
    assert metrics["baseline"]["model_distribution_metrics"].startswith("N/A")


def test_replay_all_decisions_offline(tmp_path, monkeypatch) -> None:
    client = FakeDecisionClient()
    output = tmp_path / "demo"
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    store = RunStore(output)

    def explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("replay must not call the model")

    monkeypatch.setattr(FakeDecisionClient, "evaluate", explode)
    records = [decision_from_log(entry) for entry in store.read_decisions()]
    valid = [record for record in records if record.model_status.value == "ok"]
    assert valid
    for record in valid:
        result = replay_decision(record, store.run_dir)
        assert result.replayable and result.consistent, result.detail


def test_report_and_cards_contain_required_exhibits(tmp_path) -> None:
    client = FakeDecisionClient()
    output = tmp_path / "demo"
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "## Decision table" in report
    assert "P(L=2)" in report
    assert "MIN_CHOICE_CONFIDENCE" in report
    assert "known input tokens" in report
    assert "Synthetic illustration" in report
    assert "model identity" in report
    assert render_expected_examples().splitlines()[0] in report
    cards = (output / "decision_cards.md").read_text(encoding="utf-8")
    decisions = [line for line in (output / "decisions.jsonl").read_text().splitlines() if line]
    assert cards.count("### Decision card") == len(decisions)
    assert "Expected level expansion" in cards


def test_decision_request_references_resolve_to_saved_payloads(tmp_path) -> None:
    import json

    client = FakeDecisionClient()
    output = tmp_path / "demo"
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    from idac.models import fingerprint

    checked = 0
    for line in (output / "decisions.jsonl").read_text().splitlines():
        record = json.loads(line)
        for key in ("context_ref", "questions_ref"):
            saved = json.loads((output / record[key]).read_text(encoding="utf-8"))
            assert saved["request_id"] == record["request_id"]
            assert saved["state"]["candidates"]
            assert saved["questions"]
            index = next(
                position
                for position, context in enumerate(saved["state"]["candidates"])
                if context["candidate_id"] == record["candidate_id"]
            )
            subset = {
                name: question
                for name, question in saved["questions"].items()
                if name.startswith(f"c{index}_")
            }
            assert record["context_hash"] == fingerprint(saved["state"]["candidates"][index])
            assert record["questions_hash"] == fingerprint(subset)
            checked += 1
    assert checked == 24  # 12 decisions with context and question references


def test_inspect_cli_shows_card_and_replay(tmp_path) -> None:
    client = FakeDecisionClient()
    output = tmp_path / "demo"
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    candidate_id = json.loads((output / "decisions.jsonl").read_text().splitlines()[0])["candidate_id"]
    runner = CliRunner()
    summary = runner.invoke(app, ["inspect", "--run", str(output)])
    assert summary.exit_code == 0
    assert "offline policy replay" in summary.output
    card = runner.invoke(app, ["inspect", "--run", str(output), "--candidate-id", candidate_id])
    assert card.exit_code == 0
    assert "Decision card" in card.output
    assert "Expected level expansion" in card.output


def test_evaluate_cli_refreshes_report_with_metrics(tmp_path) -> None:
    client = FakeDecisionClient()
    output = tmp_path / "demo"
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    report_before = (output / "report.md").read_text(encoding="utf-8")
    assert "No evaluation is bound" in report_before
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["evaluate", "--run", str(output), "--truth", str(DEMO_DATA / "ground_truth.csv")],
    )
    assert result.exit_code == 0
    report_after = (output / "report.md").read_text(encoding="utf-8")
    assert "No evaluation is bound" not in report_after
    assert "corruption repair coverage" in report_after
    assert "gate replay consistency rate" in report_after


def test_evaluate_and_rollback_cli(tmp_path) -> None:
    client = FakeDecisionClient()
    output = tmp_path / "demo"
    run_clean(DEMO_DATA / "dirty.csv", DEMO_CONFIG, output, client=client)
    runner = CliRunner()
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            "--run",
            str(output),
            "--truth",
            str(DEMO_DATA / "ground_truth.csv"),
        ],
    )
    assert evaluated.exit_code == 0
    assert "corruption repair coverage" in evaluated.output
    rolled = runner.invoke(app, ["rollback", "--run", str(output), "--to-version", "v000"])
    assert rolled.exit_code == 0
    assert "rolled back to v000" in rolled.output
    assert RunStore(output).manifest()["current_version"] == "v000"
