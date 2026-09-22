"""Markdown run report and decision-card export.

The report lists before/after quality counts, model decisions, unresolved
reasons, rejections, request/usage accounting, fixed thresholds and the explicit
decision table. Only modifications on the current version's ancestor chain count
as effective; rejected and rolled-back history is shown separately.
"""

from __future__ import annotations

from typing import Any

from .config import TableConfig
from .evaluation import quality_counts, unresolved_summary
from .explanation import (
    DISPLAY_NOTE,
    decision_card,
    render_card_markdown,
    render_decision_table,
    render_expected_examples,
)
from .models import (
    DecisionRecord,
    DecisionTrace,
    ModelStatus,
    ValidationResult,
    decision_from_log,
)
from .settings import (
    MAX_HIGH_RISK_PROBABILITY,
    MAX_INPUT_TOKENS,
    MAX_POSTCHECK_ERROR_PROBABILITY,
    MAX_SDK_REQUEST_ATTEMPTS,
    MAX_SEMANTIC_RISK_SCORE,
    MIN_APPLICABLE_PROBABILITY,
    MIN_CHOICE_CONFIDENCE,
    MODEL_NAME,
    POLICY_VERSION,
    QUESTION_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    RUN_TIME_LIMIT_SECONDS,
)
from .storage import RUN_FILES, RunStore


def write_run_reports(
    store: RunStore,
    config: TableConfig,
    validation_results: list[ValidationResult] | None = None,
) -> None:
    decisions = [decision_from_log(entry) for entry in store.read_decisions()]
    traces = [DecisionTrace.model_validate(entry) for entry in store.read_traces()]
    markdown = build_report_markdown(store, config, decisions, traces)
    (store.run_dir / RUN_FILES["report"]).write_text(markdown + "\n", encoding="utf-8")
    cards = build_cards_markdown(store, decisions, traces)
    (store.run_dir / RUN_FILES["cards"]).write_text(cards + "\n", encoding="utf-8")


def build_cards_markdown(
    store: RunStore,
    decisions: list[DecisionRecord],
    traces: list[DecisionTrace],
) -> str:
    chain = set(store.ancestor_chain())
    trace_map = {trace.decision_id: trace for trace in traces}
    lines = [
        "# Decision cards",
        "",
        (
            "Every card is built from the stored record; no value is re-sampled and no model "
            "call is made while rendering. Cards from versions outside the current ancestor "
            "chain describe history that is no longer in effect."
        ),
        "",
    ]
    for record in decisions:
        trace = trace_map.get(record.id)
        card = decision_card(record, trace)
        if trace and trace.committed_version and trace.committed_version not in chain:
            lines.append(
                f"> Historical: committed version `{trace.committed_version}` is not on the "
                "current ancestor chain (rolled back)."
            )
            lines.append("")
        lines.append(render_card_markdown(card))
    if not decisions:
        lines.append("_No model decisions were recorded in this run._")
    return "\n".join(lines)


def build_report_markdown(
    store: RunStore,
    config: TableConfig,
    decisions: list[DecisionRecord],
    traces: list[DecisionTrace],
) -> str:
    manifest = store.manifest()
    current = store.current_state()
    original = store.load_snapshot("v000")
    chain = store.ancestor_chain()
    chain_set = set(chain)
    changes = store.read_changes()
    effective_changes = [
        entry
        for entry in changes
        if entry.get("committed") and entry.get("committed_version") in chain_set
    ]
    rejected_changes = [entry for entry in changes if not entry.get("committed")]
    rolled_back = [
        entry
        for entry in changes
        if entry.get("committed") and entry.get("committed_version") not in chain_set
    ]
    trace_map = {trace.decision_id: trace for trace in traces}
    committed_decisions = [
        record
        for record in decisions
        if trace_map.get(record.id) and trace_map[record.id].commit_status == "committed"
    ]
    abstained = [record for record in decisions if not record.gate_accepted]
    model_failures = [
        record for record in decisions if record.model_status != ModelStatus.OK
    ]
    hard_failures = [result for result in store.read_validation_results() if not result.get("passed")]
    postcheck_failures = [
        check
        for result in store.read_validation_results()
        for check in result.get("semantic_checks", [])
        if not check.get("passed")
    ]
    audit = store.read_audit()
    rollback_events = [event for event in audit if event["event_type"] == "rollback"]
    evaluation = store.read_evaluation(manifest["current_version"])

    lines: list[str] = []
    lines.append(f"# IDAC run report: `{manifest['run_id']}`")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- outcome: **{manifest.get('outcome')}**; stop reason: `{manifest.get('stop_reason')}`")
    lines.append(f"- current version: `{manifest['current_version']}`; ancestor chain: {', '.join(chain)}")
    lines.append(f"- model: `{manifest.get('model')}` (fixed {MODEL_NAME})")
    lines.append(
        f"- model identity: **{manifest.get('model_identity')}**"
        + (
            " (FakeDecisionClient fixture; not a real Jev result)"
            if manifest.get("model_identity") == "fake"
            else " (real TypeSafe API calls)"
        )
    )
    lines.append(f"- policy version: `{POLICY_VERSION}`; question version: `{QUESTION_VERSION}`")
    lines.append(f"- input hash: `{manifest.get('input_hash')}`; config hash: `{manifest.get('config_hash')}`")
    lines.append("")

    lines.append("## Quality counts (original -> current)")
    lines.append("")
    before = quality_counts(original, config)
    after = quality_counts(current, config)
    lines.append("| Metric | Original | Current |")
    lines.append("|---|---:|---:|")
    for key in before:
        lines.append(f"| {key} | {before[key]} | {after[key]} |")
    lines.append("")
    lines.append(
        "Non-missing format issues (parseable but non-canonical) are counted separately from "
        "missing cells. IQR-only informational findings never produce rewrites."
    )
    lines.append("")

    lines.append("## Model decisions")
    lines.append("")
    lines.append(f"- decision records: {len(decisions)}")
    lines.append(f"- committed decisions: {len(committed_decisions)}")
    lines.append(f"- gate abstentions: {len(abstained)}")
    lines.append(f"- model failures / invalid responses / not called: {len(model_failures)}")
    lines.append(f"- effective change entries: {len(effective_changes)}")
    lines.append(f"- rejected plans (hard or semantic): {len(hard_failures)}")
    lines.append(f"- failed postcheck checks: {len(postcheck_failures)}")
    lines.append("")

    unresolved = unresolved_summary(current)
    lines.append("## Unresolved issues")
    lines.append("")
    if unresolved:
        lines.append("| Reason | Count |")
        lines.append("|---|---:|")
        for reason, count in sorted(unresolved.items()):
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("_No unresolved issues._")
    lines.append("")

    lines.append("## Decision table")
    lines.append("")
    lines.append(render_decision_table(decisions, traces))
    lines.append("")
    lines.append(
        f"Full cards for every decision are in [{RUN_FILES['cards']}]({RUN_FILES['cards']})."
    )
    lines.append("")

    lines.append("## Committed modification samples (current ancestor chain)")
    lines.append("")
    effective_cells = [
        change for entry in effective_changes for change in entry.get("changes", [])
    ]
    if effective_cells:
        lines.append("| Row | Column | Before | After | Candidate |")
        lines.append("|---|---|---|---|---|")
        for change in effective_cells[:8]:
            lines.append(
                f"| `{change['row_id']}` | {change['column']} | "
                f"{_cell(change.get('before'))} | {_cell(change.get('after'))} | "
                f"`{change.get('candidate_id') or ''}` |"
            )
        lines.append("")
        lines.append(
            f"Showing {min(8, len(effective_cells))} of {len(effective_cells)} effective cell "
            "changes; removed rows are listed in `removed_rows.csv`."
        )
    else:
        lines.append("_No committed cell modifications._")
    lines.append("")

    lines.append("## Rejected and historical work")
    lines.append("")
    lines.append(f"- rejected candidate change entries: {len(rejected_changes)}")
    for entry in rejected_changes[:8]:
        lines.append(
            f"  - plan `{entry['plan_id']}` candidates {entry['candidate_ids']}: "
            f"{len(entry['changes'])} computed changes were never committed"
        )
    lines.append(f"- rolled-back change entries (outside current chain): {len(rolled_back)}")
    if rollback_events:
        lines.append(
            f"- rollback events: {', '.join(event['details'].get('to_version', '?') for event in rollback_events)}"
        )
    lines.append("")

    lines.append("## Requests, usage and timing")
    lines.append("")
    api = store.read_requests()
    retries = sum(max(0, len(request.get("attempts", [])) - 1) for request in api)
    known_tokens = sum(
        int((request.get("usage") or {}).get("input_tokens") or 0)
        for request in api
        if (request.get("usage") or {}).get("known")
    )
    unknown_usage = sum(
        1
        for request in api
        if request.get("model_status") == "ok"
        and not (request.get("usage") or {}).get("known")
    )
    lines.append(f"- actual SDK request attempts recorded: {sum(len(r.get('attempts', [])) for r in api)}")
    lines.append(f"- decision requests: {sum(1 for r in api if r.get('kind') == 'decision')}")
    lines.append(f"- postcheck requests: {sum(1 for r in api if r.get('kind') == 'postcheck')}")
    lines.append(f"- retries: {retries}")
    lines.append(f"- known input tokens: {known_tokens}")
    lines.append(
        f"- requests with unknown usage: {unknown_usage} (unknown usage is never counted as zero)"
    )
    run_finished = next(
        (event for event in reversed(audit) if event["event_type"] == "run_finished"), None
    )
    elapsed = (run_finished or {}).get("details", {}).get("budget", {}).get("elapsed_seconds")
    lines.append(f"- elapsed seconds: {elapsed}")
    lines.append("")

    lines.append("## Fixed thresholds and budgets")
    lines.append("")
    lines.append("| Constant | Value |")
    lines.append("|---|---:|")
    lines.append(f"| MIN_CHOICE_CONFIDENCE | {MIN_CHOICE_CONFIDENCE} |")
    lines.append(f"| MIN_APPLICABLE_PROBABILITY | {MIN_APPLICABLE_PROBABILITY} |")
    lines.append(f"| MAX_SEMANTIC_RISK_SCORE | {MAX_SEMANTIC_RISK_SCORE} |")
    lines.append(f"| MAX_HIGH_RISK_PROBABILITY | {MAX_HIGH_RISK_PROBABILITY} |")
    lines.append(f"| MAX_POSTCHECK_ERROR_PROBABILITY | {MAX_POSTCHECK_ERROR_PROBABILITY} |")
    lines.append(f"| MAX_SDK_REQUEST_ATTEMPTS | {MAX_SDK_REQUEST_ATTEMPTS} |")
    lines.append(f"| MAX_INPUT_TOKENS | {MAX_INPUT_TOKENS} |")
    lines.append(f"| REQUEST_TIMEOUT_SECONDS | {REQUEST_TIMEOUT_SECONDS} |")
    lines.append(f"| RUN_TIME_LIMIT_SECONDS | {RUN_TIME_LIMIT_SECONDS} |")
    lines.append("")
    lines.append("These thresholds are course-project policy choices, not calibrated accuracy claims.")
    lines.append("")

    if evaluation:
        lines.append("## Evaluation summary (current version)")
        lines.append("")
        integrity = evaluation.get("integrity", {})
        corruption = evaluation.get("corruption", {})
        lines.append(
            f"- corruption repair coverage: {_fmt_rate(corruption.get('repair_coverage'))} "
            f"({corruption.get('repaired_entries')}/{corruption.get('repairable_entries')})"
        )
        lines.append(
            f"- distribution/rubric save rate: {_fmt_rate(integrity.get('distribution_save_rate'))} "
            f"({integrity.get('distribution_save_numerator')}/{integrity.get('distribution_save_denominator')})"
        )
        lines.append(
            f"- evidence/policy/final-state traceability rate: {_fmt_rate(integrity.get('traceability_rate'))} "
            f"({integrity.get('traceability_numerator')}/{integrity.get('traceability_denominator')})"
        )
        lines.append(
            f"- gate replay consistency rate: {_fmt_rate(integrity.get('gate_replay_rate'))} "
            f"({integrity.get('gate_replay_numerator')}/{integrity.get('gate_replay_denominator')})"
        )
        if evaluation.get("baseline"):
            lines.append(
                f"- rule baseline quality counts: {evaluation['baseline']['cleaned']} "
                f"({evaluation['baseline']['model_distribution_metrics']})"
            )
        lines.append("")
    else:
        lines.append("## Evaluation summary")
        lines.append("")
        lines.append(
            f"_No evaluation is bound to the current version `{manifest['current_version']}`. "
            "Run `idac evaluate` after cleaning; after a rollback, stale metrics are not shown._"
        )
        lines.append("")

    lines.append(render_expected_examples())
    lines.append("")
    lines.append(f"_{DISPLAY_NOTE}_")
    lines.append("")
    lines.append(
        "This report describes policy and record completeness; it does not claim that the model "
        "is correct on every cell or that the thresholds are calibrated."
    )
    return "\n".join(lines)


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _cell(value: Any) -> str:
    if value is None:
        return "(null)"
    return f"`{value}`"
