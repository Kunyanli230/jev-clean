"""Fixed rule baseline: apply every hard-eligible candidate directly, no model calls.

The baseline reuses the same configuration, checkers, candidate builders, seven
sub-phases and hard validator as the full pipeline, but skips the decision gate
and semantic verifier. It is an independent demonstration script, not a fallback
of ``idac clean``, and needs no API key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idac.candidates import build_candidates
from idac.config import TableConfig, config_hash, load_config
from idac.executor import ExecutionError, execute
from idac.models import (
    PHASE_ORDER,
    DatasetState,
    IssueStatus,
    Phase,
    ReasonCode,
)
from idac.planner import AcceptedCandidate, build_plan
from idac.profiling import PhaseDetection, detect_phase
from idac.storage import (
    load_csv,
    write_cleaned_csv,
    write_issues,
    write_removed_csv,
)
from idac.validator import confirmed_repaired, validate_hard


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
            self.version += 1
            try:
                execution = execute(self.state, plan, self.config, f"v{self.version:03d}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed rule baseline (no API calls).")
    parser.add_argument("--input", default="examples/data/dirty.csv")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output", default="runs/baseline")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    config = load_config(args.config)
    state = load_csv(args.input, config)
    runner = BaselineRunner(config, state)
    final = runner.run()
    output.mkdir(parents=True)
    write_cleaned_csv(final, output / "cleaned.csv")
    write_removed_csv(final, output / "removed_rows.csv")
    write_issues(output / "issues.json", final.issues)
    summary = {
        "kind": "rule_baseline",
        "model_identity": "none",
        "model_calls": 0,
        "model_distribution_metrics": "N/A - rule baseline makes no probability calls",
        "config_hash": config_hash(config),
        "versions": runner.version,
        "removed_rows": len(final.removed_rows),
        "hard_failures": runner.hard_failures,
    }
    (output / "baseline.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output / 'cleaned.csv'} ({len(final.rows)} rows, {runner.version} commits)")
    print("model calls: 0; probability distributions: N/A")


if __name__ == "__main__":
    main()
