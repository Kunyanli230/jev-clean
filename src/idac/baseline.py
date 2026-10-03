"""Explicit rule-only cleaning, without model judgments or probability records."""

from __future__ import annotations

import json
from pathlib import Path

from .candidates import build_candidates
from .config import TableConfig, config_hash, load_config
from .executor import ExecutionError, execute
from .models import (
    PHASE_ORDER,
    DatasetState,
    IssueStatus,
    Phase,
    ReasonCode,
    sha256_bytes,
)
from .planner import AcceptedCandidate, build_plan
from .profiling import PhaseDetection, detect_phase
from .storage import (
    StorageError,
    atomic_write_text,
    load_csv,
    write_cleaned_csv,
    write_issues,
    write_removed_csv,
)
from .validator import confirmed_repaired, validate_hard


class BaselineRunner:
    def __init__(self, config: TableConfig, state: DatasetState) -> None:
        self.config = config
        self.state = state
        self.evidence: dict = {}
        self.version = 0
        self.plan = 0
        self.hard_failures: list[dict] = []

    def run(self) -> DatasetState:
        for phase in PHASE_ORDER:
            self._run_phase(phase)
        return self.state

    def _run_phase(self, phase: Phase) -> None:
        while True:
            detection = detect_phase(self.state, self.config, phase)
            self._merge(detection)
            status = {issue.id: issue.status for issue in self.state.issues}
            candidates = [
                candidate
                for candidate in build_candidates(self.state, self.config, detection)
                if all(status.get(issue_id) == IssueStatus.OPEN for issue_id in candidate.issue_ids)
            ]
            if not candidates:
                return
            self.plan += 1
            accepted = [AcceptedCandidate(candidate=candidate, decision_id="baseline") for candidate in candidates]
            plan = build_plan(
                self.state, self.config, self.evidence, accepted, f"baseline-plan-{self.plan:04d}"
            )
            if not plan.ordered_operations:
                return
            next_version = self.version + 1
            try:
                execution = execute(self.state, plan, self.config, f"v{next_version:03d}")
            except ExecutionError as error:
                self.hard_failures.append({"plan": plan.id, "error": str(error)})
                return
            hard = validate_hard(self.state, execution, plan, self.config)
            if not hard.passed:
                self.hard_failures.append(
                    {
                        "plan": plan.id,
                        "failed_checks": [
                            check.rule_id
                            for check in hard.hard_checks
                            if check.status.value == "failed"
                        ],
                    }
                )
                return
            self.version = next_version
            self._mark_repaired(phase, candidates, execution.state)
            self.state = execution.state

    def _merge(self, detection: PhaseDetection) -> None:
        for evidence in detection.evidence:
            self.evidence[evidence.id] = evidence
        issues = {issue.id: issue for issue in self.state.issues}
        for issue in detection.issues:
            issues.setdefault(issue.id, issue)
        selections = dict(self.state.selections)
        selections.update(detection.selections)
        self.state = self.state.model_copy(
            update={"issues": list(issues.values()), "selections": selections}
        )

    def _mark_repaired(self, phase: Phase, candidates, new_state: DatasetState) -> None:
        issue_map = {issue.id: issue for issue in new_state.issues}
        for candidate in candidates:
            issues = [issue_map[issue_id] for issue_id in candidate.issue_ids if issue_id in issue_map]
            selections = {}
            for issue in issues:
                selection = new_state.selection(issue.selection_id or "")
                if selection is not None:
                    selections[issue.selection_id or ""] = selection
            repaired = confirmed_repaired(new_state, self.config, phase, issues, selections)
            for issue in issues:
                status = IssueStatus.REPAIRED if issue.id in repaired else IssueStatus.UNRESOLVED
                reasons = (
                    issue.reason_codes
                    if issue.id in repaired
                    else [*issue.reason_codes, ReasonCode.HARD_VALIDATION_FAILED]
                )
                issue_map[issue.id] = issue.model_copy(
                    update={"status": status, "reason_codes": reasons}
                )
        new_state.issues = list(issue_map.values())


def run_baseline(
    input_path: str | Path, config_path: str | Path, output_path: str | Path
) -> dict:
    """Apply hard-eligible local repairs and save an independently checkable snapshot."""
    output = Path(output_path)
    if output.exists():
        raise StorageError(f"output directory already exists: {output}")
    config = load_config(config_path)
    state = load_csv(input_path, config)
    runner = BaselineRunner(config, state)
    final = runner.run()
    output.mkdir(parents=True)
    write_cleaned_csv(final, output / "cleaned.csv")
    write_removed_csv(final, output / "removed_rows.csv")
    write_issues(output / "issues.json", final.issues)
    snapshot = final.model_dump_json(indent=2) + "\n"
    atomic_write_text(output / "snapshot.json", snapshot)
    summary = {
        "kind": "rule_baseline",
        "artifact_version": 2,
        "model_identity": "none",
        "model_calls": 0,
        "model_distribution_metrics": "N/A - rule baseline makes no probability calls",
        "input_hash": state.input_hash,
        "config_hash": config_hash(config),
        "snapshot_hash": sha256_bytes((output / "snapshot.json").read_bytes()),
        "versions": runner.version,
        "rows": len(final.rows),
        "removed_rows": len(final.removed_rows),
        "hard_failures": runner.hard_failures,
    }
    atomic_write_text(output / "baseline.json", json.dumps(summary, indent=2) + "\n")
    return summary
